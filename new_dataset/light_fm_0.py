
# import pandas as pd
# import numpy as np
# import pickle
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     confusion_matrix,
#     roc_curve,
#     auc
# )
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize

# # ---------------------------
# # 1. Load and Preprocess Data
# # ---------------------------
# # Load ratings and movie metadata
# ratings = pd.read_csv("ratings_small.csv")        # Expected columns: userId, movie_id, rating, ...
# movies = pd.read_csv("filtered_movies_data.csv")    # Expected columns: movie_id, title, genres, ...

# # Merge on movie_id
# data = pd.merge(ratings, movies, on="movie_id", how="inner")

# # ---------------------------
# # 2. Load BERT Embeddings
# # ---------------------------
# # Load precomputed BERT embeddings (dictionary: {movie_id (as string): embedding_vector})
# with open("..\\distilbert_movie_embeddings.pkl", "rb") as f:
#     movie_embeddings = pickle.load(f)

# # Ensure each embedding is 1D and keys are strings
# for m_id, emb in movie_embeddings.items():
#     movie_embeddings[m_id] = emb.squeeze()

# # Normalize embeddings (L2 normalization)
# all_embs = np.array(list(movie_embeddings.values()))
# all_embs_norm = normalize(all_embs, axis=1)
# movie_embeddings = dict(zip(movie_embeddings.keys(), all_embs_norm))

# # ---------------------------
# # 3. Prepare Training Data from Historical Ratings
# # ---------------------------
# # Convert ratings to binary labels: 1 if rating >= 4.0, else 0
# def label_rating(r):
#     return 1 if r >= 4.0 else 0

# data['label'] = data['rating'].apply(label_rating)

# # Build training feature matrix X and target vector y using movies with embeddings
# X = []
# y = []
# movie_ids = []

# for idx, row in data.iterrows():
#     m_id = str(row['movie_id'])
#     if m_id in movie_embeddings:
#         X.append(movie_embeddings[m_id])
#         y.append(row['label'])
#         movie_ids.append(m_id)

# X = np.array(X)
# y = np.array(y)
# print("Total training samples:", X.shape[0])

# # ---------------------------
# # 4. Train-Test Split for Evaluation
# # ---------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ---------------------------
# # 5. Initialize and Train the Global Model
# # ---------------------------
# clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
# clf.fit(X_train, y_train)

# # Evaluate on test set
# y_pred = clf.predict(X_test)
# y_proba = clf.predict_proba(X_test)[:, 1]

# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print(f"Global Model - Accuracy: {acc:.3f}, F1: {f1:.3f}")

# # Plot Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title("Confusion Matrix - Global Model")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # Plot ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve - Global Model")
# plt.legend()
# plt.show()

# # ---------------------------
# # 6. Simulate a New User and Update the Model
# # ---------------------------
# # New user provides only movie_id and rating (userId repeated)
# new_user_preferences = {
#     'userId': [9999, 9999, 9999, 9999],
#     'movie_id': [12110, 12158, 456101, 48959],
#     'rating': [5.0, 5.0, 4.5, 5.0]
# }
# new_user_df = pd.DataFrame(new_user_preferences)
# new_user_df['label'] = new_user_df['rating'].apply(label_rating)

# # Merge with movies to fetch genres for each movie the new user rated
# # Ensure movie_id is the same type in both dataframes
# new_user_df['movie_id'] = new_user_df['movie_id'].astype(int)
# movies['movie_id'] = movies['movie_id'].astype(int)
# new_user_df = pd.merge(new_user_df, movies[['movie_id', 'genres']], on="movie_id", how="left")

# # Print new user info with genres
# print("\nNew user preferences with genres:")
# print(new_user_df)

# # Build feature matrix for new user feedback from embeddings
# X_new = []
# y_new = []
# for idx, row in new_user_df.iterrows():
#     m_id = str(row['movie_id'])
#     if m_id in movie_embeddings:
#         X_new.append(movie_embeddings[m_id])
#         y_new.append(row['label'])
#     else:
#         print(f"Warning: Movie {m_id} not found in embeddings.")

# X_new = np.array(X_new)
# y_new = np.array(y_new)

# # Update the model using incremental learning (partial_fit)
# clf.partial_fit(X_new, y_new, classes=np.array([0, 1]))

# # ---------------------------
# # 7. Generate Personalized Recommendations
# # ---------------------------
# # Predict preference scores for all movies (using movies with embeddings)
# all_movie_ids = list(movie_embeddings.keys())
# all_embeddings = np.array([movie_embeddings[m_id] for m_id in all_movie_ids])
# all_proba = clf.predict_proba(all_embeddings)[:, 1]

# # Create a DataFrame with movie IDs (as strings) and predicted scores
# rec_df = pd.DataFrame({
#     'movie_id': all_movie_ids,
#     'predicted_score': all_proba
# })

# # Convert movie_id to int and remove movies that the new user rated
# rec_df['movie_id'] = rec_df['movie_id'].astype(int)
# rec_df = rec_df[~rec_df['movie_id'].isin(new_user_df['movie_id'])]

# # Merge with movies metadata to get titles and genres
# rec_df = rec_df.merge(movies, on="movie_id", how="left")

# # ---------------------------
# # Tuning: Adjust Predicted Scores by Genre Matching
# # ---------------------------
# # Extract preferred genres from the new user ratings
# # Since each movie can have multiple genres, split the 'genres' field and collect them into a set.
# def extract_genres(genres_str):
#     if pd.isna(genres_str):
#         return []
#     return [g.strip() for g in genres_str.split(",")]

# # Get a list of genres from all movies the new user liked (rating >= 4.0)
# preferred_genres = []
# for idx, row in new_user_df.iterrows():
#     if row['label'] == 1:
#         preferred_genres.extend(extract_genres(row['genres']))
# user_preferred_genres = set(preferred_genres)
# print("\nUser Preferred Genres:", user_preferred_genres)

# def compute_genre_bonus(genres_str, preferred_genres):
#     if pd.isna(genres_str):
#         return 0
#     genre_list = extract_genres(genres_str)
#     bonus = sum(1 for g in genre_list if g in preferred_genres)
#     return bonus

# # Set weighting factor alpha for genre bonus
# alpha = 0.05  # Adjust this value as needed

# rec_df['genre_bonus'] = rec_df['genres'].apply(lambda x: compute_genre_bonus(x, user_preferred_genres))
# rec_df['adjusted_score'] = rec_df['predicted_score'] + alpha * rec_df['genre_bonus']

# # Sort recommendations by adjusted score descending
# rec_df = rec_df.sort_values(by='adjusted_score', ascending=False)

# print("\nTop 10 Personalized Recommendations for New User:")
# print(rec_df[['movie_id', 'title', 'genres', 'predicted_score', 'genre_bonus', 'adjusted_score']].head(20).sort_values(by=['genre_bonus','adjusted_score'],ascending=False))


# # ---------------------------
# # 8. Plot Distribution of Adjusted Scores
# # ---------------------------
# plt.figure(figsize=(8, 4))
# sns.histplot(rec_df['adjusted_score'], bins=30, kde=True)
# plt.title("Distribution of Adjusted Preference Scores (New User)")
# plt.xlabel("Adjusted Score")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import normalize

# Optionally, check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#####################################
# 1. Load and Preprocess Data
#####################################
# Load ratings and movie metadata
ratings = pd.read_csv("ratings_small.csv")        # Expected: userId, movie_id, rating, ...
movies = pd.read_csv("filtered_movies_data.csv")    # Expected: movie_id, title, genres, ...
data = pd.merge(ratings, movies, on="movie_id", how="inner")

# Convert ratings to binary labels: 1 if rating >= 4.0, else 0.
def label_rating(r):
    return 1 if r >= 4.0 else 0

data['label'] = data['rating'].apply(label_rating)

#####################################
# 2. Load and Normalize BERT Embeddings
#####################################
with open("..\\distilbert_movie_embeddings.pkl", "rb") as f:
    movie_embeddings = pickle.load(f)

# Ensure each embedding is 1D and keys are strings
for m_id, emb in movie_embeddings.items():
    movie_embeddings[m_id] = emb.squeeze()

# Normalize embeddings (L2 normalization)
all_embs = np.array(list(movie_embeddings.values()))
all_embs_norm = normalize(all_embs, axis=1)
movie_embeddings = dict(zip(movie_embeddings.keys(), all_embs_norm))

#####################################
# 3. Build User History Sequences (for training)
#####################################
# Fixed sequence length for user history
max_seq_len = 5
user_history = {}

# Build a history for each user using positively rated movies only
for idx, row in data.iterrows():
    uid = row['userId']
    m_id = str(row['movie_id'])
    if row['label'] == 1 and m_id in movie_embeddings:
        user_history.setdefault(uid, []).append(movie_embeddings[m_id])
        
# Pad or truncate each user’s history to max_seq_len
for uid, seq in user_history.items():
    if len(seq) < max_seq_len:
        pad = [np.zeros((768,))] * (max_seq_len - len(seq))
        user_history[uid] = seq + pad
    else:
        user_history[uid] = seq[-max_seq_len:]

#####################################
# 4. Create a PyTorch Dataset for Training
#####################################
class RecommendationDataset(Dataset):
    def __init__(self, data, user_history, movie_embeddings):
        self.samples = []
        for idx, row in data.iterrows():
            uid = row['userId']
            m_id = str(row['movie_id'])
            if m_id in movie_embeddings and uid in user_history:
                self.samples.append((uid, m_id, row['label']))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        uid, m_id, label = self.samples[idx]
        # User history as a sequence: shape [max_seq_len, 768]
        user_hist = torch.tensor(user_history[uid], dtype=torch.float)
        # Candidate movie embedding: shape [768]
        movie_emb = torch.tensor(movie_embeddings[m_id], dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return user_hist, movie_emb, label

dataset = RecommendationDataset(data, user_history, movie_embeddings)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#####################################
# 5. Define the Two-Tower LSTM Model
#####################################
class TwoTowerLSTM(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=128):
        super(TwoTowerLSTM, self).__init__()
        # User tower: LSTM to encode the user's history.
        self.user_lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        # Item tower: Feed-forward network to encode the candidate movie.
        self.item_fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU()
        )
    def forward(self, user_hist, item_emb):
        # user_hist: [batch, seq_len, emb_dim]; item_emb: [batch, emb_dim]
        _, (h_n, _) = self.user_lstm(user_hist)  # h_n: [1, batch, hidden_dim]
        user_repr = h_n.squeeze(0)                # [batch, hidden_dim]
        item_repr = self.item_fc(item_emb)          # [batch, hidden_dim]
        # Dot product similarity
        score = (user_repr * item_repr).sum(dim=1)   # [batch]
        prob = torch.sigmoid(score)                  # [batch]
        return prob

model = TwoTowerLSTM(emb_dim=768, hidden_dim=128).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#####################################
# 6. Training Loop
#####################################
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for user_hist, item_emb, label in train_loader:
        user_hist = user_hist.to(device)
        item_emb = item_emb.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(user_hist, item_emb)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * user_hist.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

#####################################
# 7. Evaluation on Test Set
#####################################
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for user_hist, item_emb, label in test_loader:
        user_hist = user_hist.to(device)
        item_emb = item_emb.to(device)
        label = label.to(device)
        output = model(user_hist, item_emb)
        preds = (output >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

acc_nn = accuracy_score(all_labels, all_preds)
f1_nn = f1_score(all_labels, all_preds)
auc_nn = roc_auc_score(all_labels, all_preds)
print(f"Two-Tower LSTM - Accuracy: {acc_nn:.3f}, F1: {f1_nn:.3f}, AUC: {auc_nn:.3f}")

#####################################
# 8. New User Prediction & Personalized Recommendations
#####################################
# New user input as specified:
new_user_preferences = { 
    'userId': [9999, 9999, 9999, 9999], 
    'movie_id': [557, 558, 19593, 1726], 
    'rating': [5.0, 5.0, 4.5, 5.0] 
}
new_user_df = pd.DataFrame(new_user_preferences)
new_user_df['label'] = new_user_df['rating'].apply(label_rating)

# Merge new user data with movies to fetch genres.
new_user_df['movie_id'] = new_user_df['movie_id'].astype(int)
movies['movie_id'] = movies['movie_id'].astype(int)
new_user_df = pd.merge(new_user_df, movies[['movie_id', 'genres']], on="movie_id", how="left")
print("\nNew user preferences with genres:")
print(new_user_df)

# Build new user history sequence from these interactions.
new_user_history = []
for idx, row in new_user_df.iterrows():
    m_id = str(row['movie_id'])
    if m_id in movie_embeddings:
        new_user_history.append(movie_embeddings[m_id])
if len(new_user_history) < max_seq_len:
    pad = [np.zeros((768,))] * (max_seq_len - len(new_user_history))
    new_user_history = new_user_history + pad
else:
    new_user_history = new_user_history[-max_seq_len:]
new_user_history = torch.tensor(new_user_history, dtype=torch.float).unsqueeze(0).to(device)  # shape [1, max_seq_len, emb_dim]

# For personalized recommendations, we score all candidate movies.
all_movie_ids = list(movie_embeddings.keys())
all_embeddings = np.array([movie_embeddings[m_id] for m_id in all_movie_ids])
all_embeddings = torch.tensor(all_embeddings, dtype=torch.float).to(device)  # shape [num_movies, emb_dim]

model.eval()
with torch.no_grad():
    # Repeat new_user_history for each candidate movie.
    repeated_history = new_user_history.repeat(all_embeddings.size(0), 1, 1)  # shape [num_movies, max_seq_len, emb_dim]
    scores = model(repeated_history, all_embeddings)  # shape [num_movies]
    all_proba = scores.cpu().numpy()

# Create a recommendations DataFrame.
rec_df = pd.DataFrame({
    'movie_id': all_movie_ids,
    'predicted_score': all_proba
})
rec_df['movie_id'] = rec_df['movie_id'].astype(int)
# Exclude movies already rated by the new user.
rated_movie_ids = new_user_df['movie_id'].unique()
rec_df = rec_df[~rec_df['movie_id'].isin(rated_movie_ids)]
rec_df = rec_df.merge(movies, on="movie_id", how="left")

#####################################
# Genre Bonus Adjustment for Personalization
#####################################
def extract_genres(genres_str):
    if pd.isna(genres_str):
        return []
    return [g.strip() for g in genres_str.split(",")]

# Extract preferred genres from new user interactions (only those with positive label).
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

alpha = 0.05  # Weight for genre bonus.
rec_df['genre_bonus'] = rec_df['genres'].apply(lambda x: compute_genre_bonus(x, user_preferred_genres))
rec_df['adjusted_score'] = rec_df['predicted_score'] + alpha * rec_df['genre_bonus']

rec_df = rec_df.sort_values(by='adjusted_score', ascending=False)

print("\nTop 10 Personalized Recommendations for New User:")
print(rec_df[['movie_id', 'title', 'genres', 'predicted_score', 'genre_bonus', 'adjusted_score']].head(20).sort_values(by=['genre_bonus'], ascending=False))

plt.figure(figsize=(8, 4))
sns.histplot(rec_df['adjusted_score'], bins=30, kde=True)
plt.title("Distribution of Adjusted Preference Scores (New User)")
plt.xlabel("Adjusted Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
def save_model(model, movie_embeddings, movies, model_path="two_tower_model.pth",
               embeddings_path="movie_embeddings_updated.pkl", movies_path="movies_updated.pkl"):
    """
    Save the trained model's state dictionary, the updated movie embeddings, and the movies DataFrame.
    """
    torch.save(model.state_dict(), model_path)
    with open(embeddings_path, "wb") as f:
        pickle.dump(movie_embeddings, f)
    with open(movies_path, "wb") as f:
        pickle.dump(movies, f)
    print("Model and data saved.")


save_model(model,movie_embeddings,movies)    
