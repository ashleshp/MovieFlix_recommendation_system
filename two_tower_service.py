
import json
import os
import sys
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline, AutoTokenizer
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import normalize
import zmq

service_status = "stopped"
# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################
# TwoTowerModel Class
###############################
class TwoTowerModel:
    def __init__(self):
        # Load historical data
        self.ratings = pd.read_csv("new_dataset\\ratings_small.csv")      # userId, movie_id, rating, ...
        self.movies = pd.read_csv("filtered_movies_data.csv")  # movie_id, title, genres, ...
        self.data = pd.merge(self.ratings, self.movies, on="movie_id", how="inner")
        self.data['label'] = self.data['rating'].apply(lambda r: 1 if r >= 4.0 else 0)
        
        # Initialize the embedding model separately.
        self.embedding_model = pipeline("feature-extraction", model="distilbert-base-uncased", device=-1)  # force CPU
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Load and normalize precomputed BERT embeddings
        with open("distilbert_movie_embeddings.pkl", "rb") as f:
            movie_embeds = pickle.load(f)
        for m_id, emb in movie_embeds.items():
            movie_embeds[m_id] = emb.squeeze()
        all_embs = np.array(list(movie_embeds.values()))
        all_embs_norm = normalize(all_embs, axis=1)
        self.movie_embeddings = dict(zip(movie_embeds.keys(), all_embs_norm))

        # Build user history sequences (for training) using positive ratings only.
        self.max_seq_len = 5
        self.user_history = {}
        for idx, row in self.data.iterrows():
            uid = row['userId']
            m_id = str(row['movie_id'])
            if row['label'] == 1 and m_id in self.movie_embeddings:
                self.user_history.setdefault(uid, []).append(self.movie_embeddings[m_id])
        for uid, seq in self.user_history.items():
            if len(seq) < self.max_seq_len:
                pad = [np.zeros((768,))] * (self.max_seq_len - len(seq))
                self.user_history[uid] = seq + pad
            else:
                self.user_history[uid] = seq[-self.max_seq_len:]

        # Initialize TwoTowerLSTM model, loss, and optimizer for recommendations.
        self.model = TwoTowerLSTM(emb_dim=768, hidden_dim=128).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if os.path.exists("two_tower_model.pth") and os.path.exists("movie_embeddings_updated.pkl") and os.path.exists("movies_updated.pkl"):
            print("Saved model found. Loading model and data...")
            self.load_model()
        else:
            print("No saved model found. Initializing new model.")
        global service_status
        service_status = "running"

    def train(self, num_epochs=5):
        # Create Dataset and DataLoader
        dataset = RecommendationDataset(self.data, self.user_history, self.movie_embeddings)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, _ = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for user_hist, item_emb, label in train_loader:
                user_hist = user_hist.to(device)
                item_emb = item_emb.to(device)
                label = label.to(device)
                self.optimizer.zero_grad()
                output = self.model(user_hist, item_emb)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * user_hist.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        self.save_model()
        return {"status": "success", "message": "Model trained and saved."}

    def _get_movie_by_title(self, title):
        matches = self.movies[self.movies['title'].astype(str).str.lower() == title.lower()]
        if not matches.empty:
            return matches.iloc[0]
        else:
            print(f"Movie titled '{title}' not found in the DataFrame.")
            return None
        
    def _get_movie_id_by_title(self, title):
        """
        Helper function to return the movie_id for a given title.
        Assumes self.movies DataFrame has a 'title' column.
        """
        row = self.movies[self.movies['title'].str.lower() == title.lower()]
        if not row.empty:
            return int(row.iloc[0]['movie_id'])
        else:
            return None
        
    def _get_movie_image(self, movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyNWRjNWQ2YzY2ODNiOTliZTUzZjBmZjk0Yzc2MGQxZCIsIm5iZiI6MTczNzQ5NTkyMi40MjUsInN1YiI6IjY3OTAxNTcyZjNiYTAxOGI3MWYwOTZhZiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.yCWhQlmsX8P5J8zK1Mtytoo8UvvPQbzvUCSJRHVI5gQ"
        }
        try:
            response = requests.get(url, headers=headers, timeout=1)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            if 'posters' in data and len(data['posters']) > 0:
                poster_path = data['posters'][0]['file_path']
                full_path = f"https://image.tmdb.org/t/p/w185{poster_path}"
                return full_path
            else:
                return None
        except requests.RequestException as e:
            print(f"Error fetching image for movie ID {movie_id}: {e}")
            return None
        
    def recommend(self, new_user_ratings, N=5):
        """
        new_user_ratings: dict with keys that may include:
           'userId': [...],
           'movie_id': [557, 558, ...] or 'title': [<movie titles>],
           'rating': [...]
        Returns recommendations as a dictionary keyed by movie_id.
        """
        new_user_df = pd.DataFrame(new_user_ratings)
        new_user_df['label'] = new_user_df['rating'].apply(lambda r: 1 if r >= 4.0 else 0)
        
        # If 'movie_id' is missing but 'title' exists, convert title to movie_id.
        if 'movie_id' not in new_user_df.columns and 'title' in new_user_df.columns:
            new_user_df['movie_id'] = new_user_df['title'].apply(self._get_movie_id_by_title)
        else:
            new_user_df['movie_id'] = new_user_df['movie_id'].astype(int)
        
        self.movies['movie_id'] = self.movies['movie_id'].astype(int)
        new_user_df = pd.merge(new_user_df, self.movies[['movie_id', 'genres']], on="movie_id", how="left")
        print("New user preferences with genres:")
        print(new_user_df)
        
        # Build new user history sequence from these interactions.
        new_user_history = []
        for idx, row in new_user_df.iterrows():
            m_id = str(row['movie_id'])
            if m_id in self.movie_embeddings:
                new_user_history.append(self.movie_embeddings[m_id])
        if len(new_user_history) < self.max_seq_len:
            pad = [np.zeros((768,))] * (self.max_seq_len - len(new_user_history))
            new_user_history = new_user_history + pad
        else:
            new_user_history = new_user_history[-self.max_seq_len:]
        new_user_history = torch.tensor(new_user_history, dtype=torch.float).unsqueeze(0).to(device)
        
        # Score all candidate movies.
        all_movie_ids = list(self.movie_embeddings.keys())
        all_embeddings = np.array([self.movie_embeddings[m_id] for m_id in all_movie_ids])
        all_embeddings = torch.tensor(all_embeddings, dtype=torch.float).to(device)
        self.model.eval()
        with torch.no_grad():
            repeated_history = new_user_history.repeat(all_embeddings.size(0), 1, 1)
            scores = self.model(repeated_history, all_embeddings)
            all_proba = scores.cpu().numpy()
        rec_df = pd.DataFrame({
            'movie_id': all_movie_ids,
            'predicted_score': all_proba
        })
        rec_df['movie_id'] = rec_df['movie_id'].astype(int)
        rated_movie_ids = new_user_df['movie_id'].unique()
        rec_df = rec_df[~rec_df['movie_id'].isin(rated_movie_ids)]
        rec_df = rec_df.merge(self.movies, on="movie_id", how="left")
        
        # Genre bonus adjustment.
        def extract_genres(genres_str):
            if pd.isna(genres_str):
                return []
            return [g.strip() for g in genres_str.split(",")]
        preferred_genres = []
        for idx, row in new_user_df.iterrows():
            if row['label'] == 1:
                preferred_genres.extend(extract_genres(row['genres']))
        user_preferred_genres = set(preferred_genres)
        print("User Preferred Genres:", user_preferred_genres)
        def compute_genre_bonus(genres_str, preferred_genres):
            if pd.isna(genres_str):
                return 0
            genre_list = extract_genres(genres_str)
            bonus = sum(1 for g in genre_list if g in preferred_genres)
            return bonus
        alpha = 0.05
        rec_df['genre_bonus'] = rec_df['genres'].apply(lambda x: compute_genre_bonus(x, user_preferred_genres))
        rec_df['adjusted_score'] = rec_df['predicted_score'] + alpha * rec_df['genre_bonus']
        
        rec_df = rec_df.sort_values(by='adjusted_score', ascending=False)
        rec_df_unique = rec_df.drop_duplicates(subset=['movie_id'])
        top_N = rec_df_unique.head(N)
        top_N = top_N.sort_values(by='genre_bonus', ascending=False)
        rec_dict = top_N.set_index('movie_id').to_dict(orient="index")
        for id, movie in rec_dict.items():
            movie['poster_path'] = self._get_movie_image(int(id))
        return {"status": "success", "recommendations": {"recommendations": rec_dict}}

    def save_model(self):
        torch.save(self.model.state_dict(), "two_tower_model.pth")
        with open("movie_embeddings_updated.pkl", "wb") as f:
            pickle.dump(self.movie_embeddings, f)
        with open("movies_updated.pkl", "wb") as f:
            pickle.dump(self.movies, f)
        print("Model and data saved.")
    
    def load_model(self):
        if os.path.exists("two_tower_model.pth"):
            self.model.load_state_dict(torch.load("two_tower_model.pth", map_location=device))
            with open("movie_embeddings_updated.pkl", "rb") as f:
                self.movie_embeddings = pickle.load(f)
            with open("movies_updated.pkl", "rb") as f:
                self.movies = pickle.load(f)
            print("Model and data loaded.")
            return {"status": "success", "message": "Model loaded."}
        else:
            return {"error": "Saved model not found."}
    
    def generate_movie_embeddings(self, movies):
        """
        Generate embeddings for a list of movies.
        Each movie is expected to have keys: 'movie_id', 'title', 'genres', 'tag' (or similar).
        """
        print("Generating movie embeddings")
        for movie in movies:
            movie_description = movie['title'] + ", " + movie['genres'] + ", " + movie.get('tag', '')
            embedding = self.embedding_model(movie_description, truncation=True, max_length=512)
            embedding_tensor = torch.tensor(embedding).mean(dim=1).detach().numpy()
            self.movie_embeddings[str(movie['movie_id'])] = embedding_tensor
        print("Generate completed")    
        self.save_movie_embeddings()
        print("Saved embeddings")

    def save_movie_embeddings(self):
        with open("movie_embeddings_updated.pkl", "wb") as f:
            pickle.dump(self.movie_embeddings, f)
        print("Movie embeddings saved.")

    def standardize_embedding(self,emb, target_dim=768):
        emb = np.array(emb).flatten()  # flatten in case it's multidimensional
        if emb.shape[0] < target_dim:
            # Pad with zeros if embedding is shorter than expected
            pad_width = target_dim - emb.shape[0]
            emb = np.pad(emb, (0, pad_width), 'constant')
        elif emb.shape[0] > target_dim:
            # Truncate if embedding is longer than expected
            emb = emb[:target_dim]
        return emb


    def add_new_movies_and_update(self):
        if not os.path.exists("new_movies.csv"):
            return {"error": "new_movies.csv not found."}
        new_movies = pd.read_csv("new_movies.csv")
        movies_list = new_movies.to_dict(orient='records')

        if hasattr(self, 'embedding_model'):
            self.generate_movie_embeddings(movies_list)
        else:
            def generate_dummy_embedding(movie):
                vec = np.random.rand(768)
                return vec / np.linalg.norm(vec)
            for movie in movies_list:
                m_id = str(movie['movie_id'])
                self.movie_embeddings[m_id] = generate_dummy_embedding(movie)

        self.movies = pd.concat([self.movies, new_movies], ignore_index=True)
        # Then, when stacking the embeddings:
        embeddings_list = [self.standardize_embedding(emb) for emb in self.movie_embeddings.values()]
        all_embs = np.stack(embeddings_list)
        all_embs_norm = normalize(all_embs, axis=1)
        self.movie_embeddings = dict(zip(self.movie_embeddings.keys(), all_embs_norm))
        self.save_model()
        print(f"New movies added. Total movies: {len(self.movie_embeddings)}")
        return {"status": "success", "message": "New movies added and model saved."}

#####################################
# TwoTowerLSTM Model and RecommendationDataset definitions
#####################################
class TwoTowerLSTM(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=128):
        super(TwoTowerLSTM, self).__init__()
        self.user_lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.item_fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU()
        )
    def forward(self, user_hist, item_emb):
        _, (h_n, _) = self.user_lstm(user_hist)
        user_repr = h_n.squeeze(0)
        item_repr = self.item_fc(item_emb)
        score = (user_repr * item_repr).sum(dim=1)
        prob = torch.sigmoid(score)
        return prob

class RecommendationDataset(Dataset):
    def __init__(self, data, user_history, movie_embeddings):
        self.samples = []
        self.user_history = user_history
        self.movie_embeddings = movie_embeddings
        for idx, row in data.iterrows():
            uid = row['userId']
            m_id = str(row['movie_id'])
            if m_id in self.movie_embeddings and uid in self.user_history:
                self.samples.append((uid, m_id, row['label']))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        uid, m_id, label = self.samples[idx]
        user_hist = torch.tensor(self.user_history[uid], dtype=torch.float)
        movie_emb = torch.tensor(self.movie_embeddings[m_id], dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return user_hist, movie_emb, label

#####################################
# TwoTowerService Class
#####################################
class TwoTowerService:
    def __init__(self):
        self.model_obj = TwoTowerModel()
    
    def handle_message(self, message):
        req_type = message.get("request_type")
        if req_type == "train":
            num_epochs = message.get("num_epochs", 10)
            return self.model_obj.train(num_epochs=num_epochs)
        elif req_type == "recommend":
            new_user_input = message.get("new_user_preferences")
            if not new_user_input:
                return {"error": "New user preferences not provided."}
            return self.model_obj.recommend(new_user_input)
        elif req_type == "save_model":
            self.model_obj.save_model()
            return {"status": "success", "message": "Model saved."}
        elif req_type == "load_model":
            return self.model_obj.load_model()
        elif req_type == "add_new_movies":
            return self.model_obj.add_new_movies_and_update()
        elif req_type == "status":
            return {"status": "running"}
        elif req_type == "terminate":
            global service_status
            service_status = "stopped"
            return {"status": "stopped"}
        else:
            return {"error": "Unknown request type."}
    
    def run(self):
        PORT = 12352
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://127.0.0.1:{PORT}")
        print("TwoTowerService is running on port", PORT)
        executor = ThreadPoolExecutor(max_workers=2)
        global service_status
        while service_status == "running":
            try:
                msg = socket.recv_json()
                message = json.loads(msg)
                print("Received request:", message)
                future = executor.submit(self.handle_message, message)
                response = future.result()
                socket.send_json(json.dumps(response, ensure_ascii=False))
                sys.stdout.flush()
            except zmq.ZMQError as e:
                print(f"ZeroMQ Error: {e}")
                break
        print("TwoTowerService terminated.")

if __name__ == "__main__":
    service = TwoTowerService()
    service.run()
