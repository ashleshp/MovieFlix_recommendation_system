import json

import pandas as pd
import zmq
import os
import pickle
import torch
from transformers import pipeline,AutoTokenizer
from rapidfuzz import process, fuzz
import time
import faiss
from sentence_transformers import util
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import sys
import requests
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import implicit
import ast
import math
# Define the ports
PORT = 12349  # Recommendation Service Port
PROFILE_GENERATOR_PORT = 12348  # Profile Generator Port

# ZeroMQ Context
context = zmq.Context()

# REP Socket for FastAPI communication
socket = context.socket(zmq.REP)
socket.bind(f"tcp://127.0.0.1:{PORT}")
# Lock for thread-safe access to shared resources
lock = threading.Lock()
# Service Status
service_status = "stopped"

COLLAB_MODEL_PATH = "als_model.pkl"  # File to save the ALS model
class RecommendationModel:
    """
    Handles movie embedding generation and provides recommendations based on user profile.
    Now uses FAISS for fast similarity search.
    """

    def __init__(self, model_name="distilbert-base-uncased",
                 embeddings_file="distilbert_movie_embeddings.pkl",
                 movies_csv="filtered_movies_data.csv"):
        self.model = pipeline("feature-extraction", model=model_name, device=-1)  # force CPU
        # Initialize a tokenizer if needed (for update_profile_vector)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embeddings_file = embeddings_file
        self.movie_embeddings = self.load_movie_embeddings()  # assume keys are movie IDs (strings or ints)

        # Load movie metadata (pandas DataFrame)
        self.movie_df = pd.read_csv(movies_csv)
        self.movie_df = self.movie_df.drop_duplicates(subset=['movie_id'], keep='first')
        self.movie_df.set_index("movie_id", inplace=True)

        # Build FAISS index from the embeddings if available
        if self.movie_embeddings:
            self._build_faiss_index()
            global service_status
            service_status = "running"

        #Collab model loading sequence
        self.ratings_df = self.preprocess_ratings("new_dataset/ratings.csv")
        print("Ratings data shape:", self.ratings_df.shape)

        self.user2idx, self.movie2idx, self.idx2user, self.idx2movie = self.build_mappings(self.ratings_df)
        print("Number of unique users:", len(self.user2idx))
        print("Number of unique movies:", len(self.movie2idx))

        self.matrix = self.build_sparse_matrix(self.ratings_df,self.user2idx, self.movie2idx)
        print("User-item matrix shape:", self.matrix.shape)

        if os.path.exists(COLLAB_MODEL_PATH):
            self.model = self.load_model(COLLAB_MODEL_PATH)
            if self.model:
                print("Loaded from existing model")
            # Since confidence is derived from the ratings matrix, recreate it:
            self.confidence = self.matrix.copy()
            self.confidence.data = 1.0 + 15 * self.confidence.data
        else:
            print("Re-train the collaborative model please")
        self.collab_faiss_index, self.normalized_item_factors = self._build_collab_faiss_index(self.model.item_factors)
        print("FAISS index built with", self.collab_faiss_index.ntotal, "items.")


    def load_movie_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "rb") as file:
                print("Loaded movie embeddings.")
                return pickle.load(file)
        return {}

    def load_model(self,file_path):
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                model = pickle.load(f)
            print(f"Model loaded from {file_path}")
            return model
        return None

    def preprocess_ratings(self,file_path):
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()
        df = df.dropna(subset=['userId', 'movieId', 'rating'])
        df = df[(df['rating'] >= 0.5) & (df['rating'] <= 5.0)]
        return df

    def build_mappings(self,ratings_df):
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        user2idx = {user: idx for idx, user in enumerate(unique_users)}
        movie2idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        idx2user = {idx: user for user, idx in user2idx.items()}
        idx2movie = {idx: movie for movie, idx in movie2idx.items()}
        return user2idx, movie2idx, idx2user, idx2movie

    def build_sparse_matrix(self,ratings_df, user2idx, movie2idx):
        rows = ratings_df['userId'].map(user2idx)
        cols = ratings_df['movieId'].map(movie2idx)
        data = ratings_df['rating'].values
        n_users = len(user2idx)
        n_movies = len(movie2idx)
        matrix = coo_matrix((data, (rows, cols)), shape=(n_users, n_movies))
        return matrix.tocsr()
    

    def _get_movie_by_title(self,title):
        matches = self.movie_df[self.movie_df['title'].astype(str).str.lower() == title.lower()]
        if not matches.empty:
            return matches.iloc[0]
        else:
            print(f"Movie titled '{title}' not found in the DataFrame.")
            return None
        
    def _get_movie_by_id(self, movie_id):
        """
        Retrieve the movie record from the movie DataFrame using the movie ID.
        Returns the movie row as a Series if found, or None if not found.
        """
        try:
            return self.movie_df.loc[int(movie_id)]
        except KeyError:
            return None  
          
    def _build_collab_faiss_index(self,item_factors):
        item_factors = item_factors.copy()
        norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_item_factors = item_factors / norms

        d = normalized_item_factors.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(normalized_item_factors.astype('float32'))
        return index, normalized_item_factors


    def save_movie_embeddings(self):
        with open(self.embeddings_file, "wb") as file:
            pickle.dump(self.movie_embeddings, file)

    def generate_movie_embeddings(self, movies):
        """
        Generate embeddings for a list of movies.
        Each movie is expected to have keys: 'movie_id', 'title', 'genres', 'tag' (or similar).
        """
        for movie in movies:
            # Here we assume a rich description that includes title, genres, and tag.
            movie_description = movie['title'] + ", " + movie['genres'] + ", " + movie.get('tag', '')
            embedding = self.model(movie_description, truncation=True, max_length=512)
            # Average the output tokens to get a single vector
            embedding_tensor = torch.tensor(embedding).mean(dim=1).detach().numpy()
            # Store the embedding using movie_id as the key (ensure consistency with movie_df)
            self.movie_embeddings[str(movie['movie_id'])] = embedding_tensor
        self.save_movie_embeddings()
        # After updating embeddings, rebuild the FAISS index
        self._build_faiss_index()

    def _build_faiss_index(self):
        """
        Build a FAISS index from the movie embeddings.
        Assumes that each embedding is a numpy array.
        """
        self.movie_ids = []
        embeddings_list = []

        # Convert the embeddings dictionary into a list and a corresponding index list.
        for movie_id, embedding in self.movie_embeddings.items():
            self.movie_ids.append(movie_id)
            vec = np.array(embedding)
            # Squeeze any extra dimension (e.g., from shape (1,768) to (768,))
            if len(vec.shape) > 1:
                vec = vec.squeeze(0)
            embeddings_list.append(vec)

        self.embeddings_matrix = np.vstack(embeddings_list).astype('float32')
        print(f"Shape of embeddings matrix: {self.embeddings_matrix.shape}")

        # Normalize the embeddings so that inner product equals cosine similarity.
        faiss.normalize_L2(self.embeddings_matrix)

        # Build a FAISS index using inner product.
        d = self.embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings_matrix)
        print(f"Total number of movies indexed: {self.index.ntotal}")

    def get_movie_image(self, movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyNWRjNWQ2YzY2ODNiOTliZTUzZjBmZjk0Yzc2MGQxZCIsIm5iZiI6MTczNzQ5NTkyMi40MjUsInN1YiI6IjY3OTAxNTcyZjNiYTAxOGI3MWYwOTZhZiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.yCWhQlmsX8P5J8zK1Mtytoo8UvvPQbzvUCSJRHVI5gQ"  # Replace with your actual token
        }

        try:
            response = requests.get(url, headers=headers, timeout=1)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            # Extract poster path
            if 'posters' in data and len(data['posters']) > 0:
                poster_path = data['posters'][0]['file_path']
                full_path = f"https://image.tmdb.org/t/p/w185{poster_path}"
                return full_path
            else:
                return None
        except requests.RequestException as e:
            print(f"Error fetching image for movie ID {movie_id}: {e}")
            return None

    def recommend_collab_for_new_user(self, new_user_ratings, N=5):
        """
        Recommend movies for a new user by:
        1. Building a temporary confidence vector from the provided (movie title, rating) pairs.
            (Liked movies with rating 5 and disliked movies with rating 0.0)
        2. Inferring the new user's latent factor using the trained ALS model.
        3. Querying the FAISS index to find top-N recommendations while filtering out movies the user has already rated.
        """
        alpha = 15
        n_items = len(self.movie2idx)
        user_conf = np.zeros(n_items, dtype=np.float32)
        rated_items = set()

        # Build confidence vector and record rated movie indices.
        for movie, rating in new_user_ratings:
            movie_row = self._get_movie_by_title(movie)
            # Get the movie id from the row (using .name since the DataFrame is indexed by movie_id)
            movie_id = movie_row.name
            print(f"\nTHE movie ID is : {movie_id}")
            if movie_id in self.movie2idx:
                idx = self.movie2idx[movie_id]
                rated_items.add(idx)
                # If the movie is liked (rating > 0), boost the confidence.
                # If the movie is disliked (rating == 0), assign a low confidence.
                if rating > 0:
                    user_conf[idx] = 1.0 + alpha * rating
                else:
                    user_conf[idx] = 0.1  # A very low confidence to signal dislike

        # Create a sparse representation of the user's interactions.
        new_user_items = csr_matrix(user_conf.reshape(1, -1))  # Shape: (1, n_items)

        # Infer the new user's latent factor using the sparse representation.
        new_user_factor = self.model.recalculate_user(0, new_user_items)  # using 0 as a dummy user_id

        # Normalize the new user's latent vector.
        norm = np.linalg.norm(new_user_factor)
        if norm == 0:
            norm = 1.0
        normalized_user_vector = (new_user_factor / norm).astype('float32').reshape(1, -1)

        # Increase candidate pool size to have room for filtering.
        candidate_pool_size = N + len(new_user_ratings)
        D, I = self.collab_faiss_index.search(normalized_user_vector, candidate_pool_size)

        # Filter out movies that are already rated by the user and any duplicates.
        recommended = []
        seen = set()
        for idx in I[0]:
            if idx in rated_items:
                continue
            movie_id = self.idx2movie[idx]
            if movie_id in seen:
                continue
            # Extra check: only add if _get_movie_by_id returns a valid movie record.
            movie_record = self._get_movie_by_id(movie_id)
            if movie_record is None:
                continue
            seen.add(movie_id)
            recommended.append(movie_id)
            if len(recommended) >= N:
                break

        print(f"The recommendations are: {recommended}")

        community_recommended = {}
        for mid in recommended:
            if int(mid) in self.movie_df.index:
                movie_row = self.movie_df.loc[int(mid)]
                movie_row['poster_path'] = self.get_movie_image(int(mid))
                community_recommended[int(mid)] = movie_row.to_dict()
            else:
                print(f"Movie id {mid} not found in movie_df. Skipping.")
        return community_recommended



    def recommend_movies(self, user_vector, liked_movies, disliked_movies, num_recommendations=5):
        """
        Recommend movies by querying the FAISS index with the current (normalized) user profile vector.
        Filters out movies already liked or disliked.
        """
        # Ensure the user_vector is a numpy array of type float32 and reshape to (1, d)
        user_vec = np.array(user_vector, dtype='float32').reshape(1, -1)
        # Normalize the profile vector (same as the movie embeddings)
        faiss.normalize_L2(user_vec)

        # Retrieve more than needed so we can filter out liked/disliked movies.
        k_search = num_recommendations + (len(liked_movies)+len(disliked_movies))
        distances, indices = self.index.search(user_vec, k_search)

        recommendations = []
        for idx in indices[0]:
            movie_id = self.movie_ids[idx]
            # Look up movie details from the DataFrame (convert movie_id to int if necessary)
            try:
                movie_row = self.movie_df.loc[int(movie_id)]
            except Exception as e:
                print(f"Movie ID {movie_id} not found in movie_df: {e}")
                continue
            movie_title = movie_row['title']
            if movie_title not in liked_movies and movie_title not in disliked_movies:
                recommendations.append(movie_id)
            if len(recommendations) == num_recommendations:
                break

        res = {}
        for rec_id in recommendations:
            try:
                movie_row = self.movie_df.loc[int(rec_id)]
                print(f"Recommended: {movie_row['title']}, directed by {movie_row.get('director', 'N/A')}")
                rec_details = movie_row.to_dict()
                rec_details['poster_path'] = self.get_movie_image(rec_id)
                res[rec_id] = rec_details
            except Exception as e:
                print(f"Error retrieving details for movie ID {rec_id}: {e}")
        return res
    



    def clean_data(self,data):
        """
        Recursively replace NaN, None, or Inf values in a dictionary or list with None.
        """
        if isinstance(data, dict):
            return {k: self.clean_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.clean_data(v) for v in data]
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return None  # Convert NaN/Inf to None
            return data
        return data

    

    def search_movies(self, query, num_results=5):
        """
        Search movies using RapidFuzz for title matching.
        """
        if not query:
            return {"error": "Search query cannot be empty."}

        # Get all movie titles from the dataframe
        movie_titles = self.movie_df["title"].astype(str).tolist()

        # Use RapidFuzz to find the best matches
        matches = process.extract(query, movie_titles, scorer=fuzz.ratio, limit=num_results)

        search_results = []
        for match in matches:
            title, score, idx = match  # RapidFuzz returns (title, score, index)
            movie_row = self.movie_df.iloc[idx]

            # Convert to dictionary and fetch the poster image
            movie_data = movie_row.to_dict()
            movie_data["poster_path"] = self.get_movie_image(movie_row.name)
            movie_data["movie_id"] = int(movie_row.name)

            # Clean the data (handle NaN, Inf values)
            search_results.append(self.clean_data(movie_data))
        print(search_results)
        return search_results




class RecommendationService:
    def __init__(self):
        self.recommendation_model = RecommendationModel()
        self.executor = ThreadPoolExecutor(max_workers=4)  # Handle 10 concurrent requests

    def handle_request(self, message):
        global service_status
        request_type = message.get("request_type")

        if request_type == "generate_embeddings":
            movies = message.get("movies", [])
            self.recommendation_model.generate_movie_embeddings(movies)
            return {"status": "success", "message": "Movie embeddings generated."}

        elif request_type == "recommend_movies":
            user_vector = message.get("profile_vector")
            liked_movies = message.get("liked_movies", [])
            disliked_movies = message.get("disliked_movies", [])
            num_recommendations = message.get("num_recommendations", 5)
            if not user_vector:
                return {"error": "User profile vector is missing."}
            recommendations = self.recommendation_model.recommend_movies(
                user_vector, liked_movies, disliked_movies, num_recommendations
            )
            return {"status": "success", "recommendations": recommendations}
        elif request_type == "get_community_picks":
            user_ratings = message.get("user_ratings")
            community_recommendations = self.recommendation_model.recommend_collab_for_new_user(user_ratings)
            print(f"The community based picks are : {community_recommendations}")
            if not community_recommendations:
                return {"error": "Community based picks failed"}
            return {"status": "success", "recommendations": community_recommendations}
        elif request_type == "status":
            return {"status": service_status}

        elif request_type == "terminate":
            service_status = "stopped"
            return {"status": "stopped"}


        elif request_type == "search_movies":
            query = message.get("query", "").strip()
            num_results = message.get("num_results", 5)

            if not query:
                return {"error": "Search query cannot be empty."}

            search_results = self.recommendation_model.search_movies(query, num_results)
            
            return {"status": "success", "movies": search_results}

        else:
            return {"error": "Unknown request type."}

    def run(self):
        global service_status
        print("Recommendation Service is running...")
        while service_status == "running":
            try:
                message = json.loads(socket.recv_json())
                print(f"Received request: {message}")

                future = self.executor.submit(self.handle_request, message)
                response = future.result()
                socket.send_json(json.dumps(response))
                sys.stdout.flush()
            except Exception as e:
                print(f"Error: {e}")
                socket.send_json(json.dumps({"error": "Internal server error"},ensure_ascii=False))

        print("Recommendation Service stopped.")


if __name__ == "__main__":
    service = RecommendationService()
    service.run()
