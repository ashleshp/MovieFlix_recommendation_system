
import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import normalize

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
ratings = pd.read_csv("ratings_small.csv")        # Expected columns: userId, movie_id, rating, ...
movies = pd.read_csv("filtered_movies_data.csv")    # Expected columns: movie_id, title, genres, ...
data = pd.merge(ratings, movies, on="movie_id", how="inner")

# ---------------------------
# 2. Load BERT Embeddings
# ---------------------------
with open("..\\distilbert_movie_embeddings.pkl", "rb") as f:
    movie_embeddings = pickle.load(f)

# Ensure each embedding is 1D and keys are strings
for m_id, emb in movie_embeddings.items():
    movie_embeddings[m_id] = emb.squeeze()

# Normalize embeddings (L2 normalization)
all_embs = np.array(list(movie_embeddings.values()))
all_embs_norm = normalize(all_embs, axis=1)
movie_embeddings = dict(zip(movie_embeddings.keys(), all_embs_norm))

# ---------------------------
# 3. Prepare Training Data from Historical Ratings
# ---------------------------
def label_rating(r):
    return 1 if r >= 4.0 else 0

data['label'] = data['rating'].apply(label_rating)

X = []
y = []
movie_ids = []  # track movie_ids corresponding to each row

for idx, row in data.iterrows():
    m_id = str(row['movie_id'])
    if m_id in movie_embeddings:
        X.append(movie_embeddings[m_id])
        y.append(row['label'])
        movie_ids.append(m_id)

X = np.array(X)
y = np.array(y)
print("Total training samples:", X.shape[0])

# ---------------------------
# 4. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 5. Hyperparameter Tuning with GridSearchCV
# ---------------------------
# Exclude 'hinge' since it does not support predict_proba.
# Also add a fixed eta0 (e.g., 0.01) for learning_rate options that require it.
param_grid = {
    'loss': ['log_loss', 'modified_huber'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['optimal', 'constant', 'invscaling'],
    'eta0': [0.01]  # Ensure eta0 > 0 for constant/invscaling learning rates
}

sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='f1', n_jobs=-1, error_score='raise',verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters from GridSearch:", grid_search.best_params_)
print("Best CV F1 score:", grid_search.best_score_)

# Use the best estimator
best_clf = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_clf.predict(X_test)
y_proba = best_clf.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Best Model on Test Set - Accuracy: {acc:.3f}, F1: {f1:.3f}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Best Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Model")
plt.legend()
plt.show()

# ---------------------------
# 6. Simulate a New User and Update the Model
# ---------------------------
# New user now provides only movie_id and rating.
new_user_preferences = {
    'userId': [9999, 9999, 9999, 9999, 9999],
    'movie_id': [1930, 1724, 414, 1924, 414],
    'rating': [5.0, 5.0, 4.5, 5.0, 4.7]
}
new_user_df = pd.DataFrame(new_user_preferences)
new_user_df['label'] = new_user_df['rating'].apply(label_rating)

# Merge new user info with movies to fetch genres
new_user_df['movie_id'] = new_user_df['movie_id'].astype(int)
movies['movie_id'] = movies['movie_id'].astype(int)
new_user_df = pd.merge(new_user_df, movies[['movie_id', 'genres']], on="movie_id", how="left")
print("\nNew user preferences with genres:")
print(new_user_df)

# Build feature matrix for new user using embeddings
X_new = []
y_new = []
for idx, row in new_user_df.iterrows():
    m_id = str(row['movie_id'])
    if m_id in movie_embeddings:
        X_new.append(movie_embeddings[m_id])
        y_new.append(row['label'])
    else:
        print(f"Warning: Movie {m_id} not found in embeddings.")
X_new = np.array(X_new)
y_new = np.array(y_new)

# Update the best model using incremental learning
best_clf.partial_fit(X_new, y_new, classes=np.array([0, 1]))

# ---------------------------
# 7. Generate Personalized Recommendations
# ---------------------------
# Predict preference scores for all movies using embeddings
all_movie_ids = list(movie_embeddings.keys())
all_embeddings = np.array([movie_embeddings[m_id] for m_id in all_movie_ids])
all_proba = best_clf.predict_proba(all_embeddings)[:, 1]

# Create a DataFrame for recommendations
rec_df = pd.DataFrame({
    'movie_id': all_movie_ids,
    'predicted_score': all_proba
})
rec_df['movie_id'] = rec_df['movie_id'].astype(int)
# Exclude movies already rated by the new user
rec_df = rec_df[~rec_df['movie_id'].isin(new_user_df['movie_id'])]
rec_df = rec_df.merge(movies, on="movie_id", how="left")

# ---------------------------
# Genre Bonus Adjustment for Personalization
# ---------------------------
def extract_genres(genres_str):
    if pd.isna(genres_str):
        return []
    return [g.strip() for g in genres_str.split(",")]

# Extract genres from movies the new user liked (rating>=4.0)
preferred_genres = []
for idx, row in new_user_df.iterrows():
    if row['label'] == 1:
        preferred_genres.extend(extract_genres(row['genres']))
user_preferred_genres = set(preferred_genres)
print("\nUser Preferred Genres:", user_preferred_genres)

def compute_genre_bonus(genres_str, preferred_genres):
    if pd.isna(genres_str):
        return 0
    genre_list = extract_genres(genres_str)
    bonus = sum(1 for g in genre_list if g in preferred_genres)
    return bonus

alpha = 0.05  # Weighting factor for genre bonus
rec_df['genre_bonus'] = rec_df['genres'].apply(lambda x: compute_genre_bonus(x, user_preferred_genres))
rec_df['adjusted_score'] = rec_df['predicted_score'] + alpha * rec_df['genre_bonus']

# Sort recommendations by adjusted score descending
rec_df = rec_df.sort_values(by='adjusted_score', ascending=False)

print("\nTop 10 Personalized Recommendations for New User:")
print(rec_df[['movie_id', 'title', 'genres', 'predicted_score', 'genre_bonus', 'adjusted_score']].head(10))

# ---------------------------
# 8. Plot Distribution of Adjusted Scores
# ---------------------------
plt.figure(figsize=(8, 4))
sns.histplot(rec_df['adjusted_score'], bins=30, kde=True)
plt.title("Distribution of Adjusted Preference Scores (New User)")
plt.xlabel("Adjusted Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
