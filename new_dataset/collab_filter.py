
import os
import pickle
import pandas as pd
import numpy as np
import faiss
from scipy.sparse import coo_matrix, csr_matrix
import implicit

MODEL_PATH = "als_model.pkl"  # File to save the ALS model


def save_model(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")


def load_model(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return model
    return None


def preprocess_ratings(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=['userId', 'movieId', 'rating'])
    df = df[(df['rating'] >= 0.5) & (df['rating'] <= 5.0)]
    return df


def build_mappings(ratings_df):
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()
    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    movie2idx = {movie: idx for idx, movie in enumerate(unique_movies)}
    idx2user = {idx: user for user, idx in user2idx.items()}
    idx2movie = {idx: movie for movie, idx in movie2idx.items()}
    return user2idx, movie2idx, idx2user, idx2movie


def build_sparse_matrix(ratings_df, user2idx, movie2idx):
    rows = ratings_df['userId'].map(user2idx)
    cols = ratings_df['movieId'].map(movie2idx)
    data = ratings_df['rating'].values
    n_users = len(user2idx)
    n_movies = len(movie2idx)
    matrix = coo_matrix((data, (rows, cols)), shape=(n_users, n_movies))
    return matrix.tocsr()


def train_als(matrix, alpha=15, factors=50, regularization=0.1, iterations=10):
    confidence = matrix.copy()
    confidence.data = 1.0 + alpha * confidence.data

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        calculate_training_loss=True
    )
    model.fit(confidence)
    return model, confidence


def build_faiss_index(item_factors):
    item_factors = item_factors.copy()
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_item_factors = item_factors / norms

    d = normalized_item_factors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(normalized_item_factors.astype('float32'))
    return index, normalized_item_factors


def recommend_for_existing_user(user_id, user2idx, idx2movie, model, confidence, faiss_index, N=5):
    if user_id not in user2idx:
        print(f"User {user_id} not found.")
        return []
    user_idx = user2idx[user_id]
    user_vector = model.user_factors[user_idx]
    norm = np.linalg.norm(user_vector)
    if norm == 0:
        norm = 1.0
    normalized_user_vector = (user_vector / norm).astype('float32').reshape(1, -1)

    D, I = faiss_index.search(normalized_user_vector, N + 20)
    user_items = set(confidence[user_idx].indices)
    recommendations = []
    for item_idx in I[0]:
        if item_idx not in user_items:
            recommendations.append(idx2movie[item_idx])
        if len(recommendations) >= N:
            break
    return recommendations


def recommend_for_new_user(new_user_ratings, movie2idx, idx2movie, model, faiss_index, alpha=15, N=5):
    """
    Recommend movies for a new user by:
      1. Building a temporary confidence vector from the provided (movieId, rating) pairs.
      2. Inferring the new user's latent factor using the trained ALS model.
      3. Querying the FAISS index to find top-N recommendations.
    """
    n_items = len(movie2idx)
    user_conf = np.zeros(n_items, dtype=np.float32)
    for movie, rating in new_user_ratings:
        if movie in movie2idx:
            idx = movie2idx[movie]
            user_conf[idx] = 1.0 + alpha * rating

    # Create a sparse representation of the user's interactions.
    new_user_items = csr_matrix(user_conf.reshape(1, -1))  # Shape: (1, n_items)

    # Infer the new user's latent factor using the sparse representation.
    new_user_factor = model.recalculate_user(0, new_user_items)  # using 0 as a dummy user_id

    # Normalize the new user's latent vector.
    norm = np.linalg.norm(new_user_factor)
    if norm == 0:
        norm = 1.0
    normalized_user_vector = (new_user_factor / norm).astype('float32').reshape(1, -1)

    # Retrieve top-N recommendations via FAISS.
    D, I = faiss_index.search(normalized_user_vector, N)
    recommended = [idx2movie[idx] for idx in I[0]]
    return recommended


def main():
    # -------------------------
    # 1. Preprocessing
    # -------------------------
    ratings_df = preprocess_ratings("ratings.csv")
    print("Ratings data shape:", ratings_df.shape)

    user2idx, movie2idx, idx2user, idx2movie = build_mappings(ratings_df)
    print("Number of unique users:", len(user2idx))
    print("Number of unique movies:", len(movie2idx))

    matrix = build_sparse_matrix(ratings_df, user2idx, movie2idx)
    print("User-item matrix shape:", matrix.shape)

    # Report matrix density and sparsity.
    n_users, n_items = matrix.shape
    density = matrix.nnz / (n_users * n_items)
    sparsity = 1 - density
    print(f"Matrix density: {density * 100:.4f}%")
    print(f"Matrix sparsity: {sparsity * 100:.4f}%")

    # -------------------------
    # 2. Train or Load ALS Model
    # -------------------------
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        if model:
            print("Loaded from existing model")
        # Since confidence is derived from the ratings matrix, recreate it:
        confidence = matrix.copy()
        confidence.data = 1.0 + 15 * confidence.data
    else:
        model, confidence = train_als(matrix, alpha=15, factors=50, regularization=0.1, iterations=10)
        print("ALS model trained.")
        save_model(model, MODEL_PATH)

    # Report latent factor statistics.
    user_norms = np.linalg.norm(model.user_factors, axis=1)
    item_norms = np.linalg.norm(model.item_factors, axis=1)
    print(f"User factors: mean norm = {np.mean(user_norms):.4f}, std = {np.std(user_norms):.4f}")
    print(f"Item factors: mean norm = {np.mean(item_norms):.4f}, std = {np.std(item_norms):.4f}")

    # -------------------------
    # 3. Build FAISS Index on Item Factors
    # -------------------------
    faiss_index, normalized_item_factors = build_faiss_index(model.item_factors)
    print("FAISS index built with", faiss_index.ntotal, "items.")
    print("Sample item latent vector (first 5):")
    print(normalized_item_factors[:5])

    # -------------------------
    # 4. Recommendations for an Existing User
    # -------------------------
    example_user_id = 1  # Change this to a valid userId from your dataset
    rec_existing = recommend_for_existing_user(
        example_user_id, user2idx, idx2movie, model, confidence, faiss_index, N=5)
    print(f"Recommendations for existing user {example_user_id}: {rec_existing}")

    # -------------------------
    # 5. Recommendations for a New User
    # -------------------------
    new_user_ratings = [
        (110, 4.0),  # (movieId, rating)
        (858, 5.0)
    ]
    rec_new = recommend_for_new_user(
        new_user_ratings, movie2idx, idx2movie, model, faiss_index, alpha=15, N=5)
    print(f"Recommendations for new user (with ratings {new_user_ratings}): {rec_new}")


if __name__ == "__main__":
    main()
