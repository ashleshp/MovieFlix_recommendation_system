
import json
import sys

import zmq
import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
import threading

# Define Ports
PORT = 12348  # Profile Generator Service Port
USER_SERVICE_PORT = 12345  # Port for User Service

# ZeroMQ Context
context = zmq.Context()

# REP Socket to listen to FastAPI Server or other requests
socket = context.socket(zmq.REP)
socket.bind(f"tcp://127.0.0.1:{PORT}")

# Global Status of the service
service_status = "running"

# Lock for thread safety
lock = threading.Lock()


class ProfileGenerator:
    """
    A microservice responsible for generating and updating user embeddings.
    """
    def __init__(self, model_name="distilbert-base-uncased", embeddings_dir="user_embeddings"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embeddings_dir = embeddings_dir
        os.makedirs(self.embeddings_dir, exist_ok=True)
        self.cache = {}  # In-memory cache for faster access to embeddings

    def load_embedding(self, email):
        """
        Load the profile embedding for a given user email.
        Thread-safe with locking.
        """
        with lock:
            if email in self.cache:
                print(f"Loaded embedding for {email} from cache.")
                return self.cache[email]

            embedding_file = os.path.join(self.embeddings_dir, f"{email}_embedding.pkl")
            if os.path.exists(embedding_file):
                with open(embedding_file, "rb") as file:
                    embedding = pickle.load(file)
                    self.cache[email] = embedding  # Cache the embedding for future use
                    print(f"Loaded embedding for {email}.")
                    return embedding
            print(f"No existing embedding found for {email}. Generating new.")
            return None

    def save_embedding(self, email, embedding):
        """
        Save the profile embedding for a given user email.
        Thread-safe with locking.
        """
        with lock:
            embedding_file = os.path.join(self.embeddings_dir, f"{email}_embedding.pkl")
            with open(embedding_file, "wb") as file:
                pickle.dump(embedding, file)
            self.cache[email] = embedding  # Update cache
            print(f"Saved embedding for {email}.")

    def generate_profile_vector(self, preferred_genres):
        """
        Generate a new profile vector based on preferred genres.
        """
        print("Generating profile vector...")
        input_text = ', '.join(preferred_genres)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        profile_vector = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
        print("Profile vector generated successfully.")
        return profile_vector

    def update_profile_vector(self, current_vector, liked_movies, disliked_movies):
        """
        Update the profile vector based on user feedback.
        """
        print("Updating profile vector...")
        accumulated_vector = torch.tensor(current_vector, dtype=torch.float32)

        # Apply liked movies influence
        for movie in liked_movies:
            movie_input = self.tokenizer(movie, return_tensors="pt", truncation=True, max_length=512)
            movie_output = self.model(**movie_input)
            movie_vector = torch.mean(movie_output.last_hidden_state, dim=1).detach()
            accumulated_vector += 0.1 * movie_vector

        # Apply disliked movies influence
        for movie in disliked_movies:
            movie_input = self.tokenizer(movie, return_tensors="pt", truncation=True, max_length=512)
            movie_output = self.model(**movie_input)
            movie_vector = torch.mean(movie_output.last_hidden_state, dim=1).detach()
            accumulated_vector -= 0.2 * movie_vector

        # Normalize the profile vector
        accumulated_vector = accumulated_vector / torch.norm(accumulated_vector)
        print("Profile vector updated successfully.")
        return accumulated_vector.numpy()


class ProfileGeneratorService:
    def __init__(self):
        self.profile_generator = ProfileGenerator()
        self.executor = ThreadPoolExecutor(max_workers=10)  # Thread pool with 10 workers

    def handle_request(self, message):
        """
        Handle incoming requests for generating or updating profile vectors.
        """
        global service_status
        request_type = message.get("request_type")
        email = message.get("email")

        if request_type == "generate_profile":
            preferred_genres = message.get("preferred_genres", [])
            if not preferred_genres:
                return {"error": "Preferred genres missing."}

            profile_vector = self.profile_generator.generate_profile_vector(preferred_genres)
            self.profile_generator.save_embedding(email, profile_vector)
            return {"status": "success", "message": f"Profile generated for {email}."}

        elif request_type == "get_profile_vector":
            profile_vector = self.profile_generator.load_embedding(email)
            if profile_vector is None:
                return {"error": "No profile vector found for this user."}

            return {"status": "success", "profile_vector": profile_vector.tolist()}

        elif request_type == "update_profile":
            liked_movies = message.get("liked_movies", [])
            disliked_movies = message.get("disliked_movies", [])

            current_vector = self.profile_generator.load_embedding(email)
            if current_vector is None:
                return {"error": "No existing profile vector found. Generate a new profile first."}

            updated_vector = self.profile_generator.update_profile_vector(current_vector, liked_movies, disliked_movies)
            self.profile_generator.save_embedding(email, updated_vector)
            return {"status": "success", "message": f"Profile updated for {email}."}

        elif request_type == "status":
            return {"status": service_status}

        elif request_type == "terminate":
            service_status = "stopped"
            return {"status": "stopped"}

        else:
            return {"error": "Unknown request type."}

    def run(self):
        """
        Main loop for the profile generator service with ThreadPoolExecutor.
        """
        global service_status
        print("Profile Generator Service is running...")

        while service_status == "running":
            try:
                # Wait for a request
                message = json.loads(socket.recv_json())
                print(f"Received request: {message}")

                # Submit request to thread pool
                future = self.executor.submit(self.handle_request, message)
                response = future.result()

                # Send response
                socket.send_json(json.dumps(response))
                sys.stdout.flush()
            except zmq.ZMQError as e:
                print(f"ZeroMQ Error: {e}")
                break

        print("Profile Generator Service stopped.")


if __name__ == "__main__":
    service = ProfileGeneratorService()
    service.run()
