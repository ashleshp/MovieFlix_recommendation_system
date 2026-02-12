
import os
import pickle
import zmq
import time
import json
import concurrent.futures
import threading
import sys
# Define the port for the User Service
PORT = 12345

# ZeroMQ Context
context = zmq.Context()

# Define a REP socket for handling incoming status/terminate requests
socket = context.socket(zmq.REP)
socket.bind(f"tcp://127.0.0.1:{PORT}")

# Status of the service
service_status = "running"

# UserProfile Class
class UserProfile:
    def __init__(self, user_id, email, password, age, country, preferred_genres):
        self.user_id = user_id
        self.email = email
        self.password = password
        self.age = age
        self.country = country
        self.preferred_genres = preferred_genres
        self.profile_vector = None
        self.liked_movies = []
        self.disliked_movies = []


class UserService:
    """
    A microservice responsible for user management: register, login, and update user preferences.
    """
    def __init__(self, user_folder="user_data"):
        self.user_folder = user_folder
        os.makedirs(self.user_folder, exist_ok=True)

    def get_user_file(self, email):
        """Return the file path for a specific user."""
        return os.path.join(self.user_folder, f"{email}.pkl")

    def load_user(self, email):
        """Load user data from a user-specific file."""
        user_file = self.get_user_file(email)
        if os.path.exists(user_file):
            with open(user_file, "rb") as file:
                return pickle.load(file)
        return None

    def save_user(self, user_data):
        """Save user data to a user-specific file."""
        user_file = self.get_user_file(user_data["email"])
        with open(user_file, "wb") as file:
            pickle.dump(user_data, file)

    def register_user(self, message):
        """Register a new user."""
        email = message["email"]
        if self.load_user(email):
            return {"error": f"User {email} already exists."}

        user_id = len(os.listdir(self.user_folder)) + 1
        user_data = {
            "user_id": user_id,
            "email": email,
            "password": message["password"],  # Storing the hashed password as-is
            "age": message["age"],
            "country": message["country"],
            "preferred_genres": message["preferred_genres"],
            "liked_movies": [],
            "disliked_movies": [],
            "profile_vector": None,
        }
        self.save_user(user_data)
        return {"message": f"User {email} registered successfully."}

    def login_user(self, message):
        """Login an existing user."""
        email = message["email"]
        user_data = self.load_user(email)
        if not user_data:
            return {"error": "User not found."}

        # Return the stored hashed password for verification
        return {"message": f"User {email} logged in successfully.", "password": user_data["password"]}

    def update_user(self, message):
        """Update user preferences."""
        email = message["email"]
        user_data = self.load_user(email)
        if not user_data:
            return {"error": "User not found."}
        user_data["liked_movies"] += message.get("liked_movies", [])
        user_data["disliked_movies"] += message.get("disliked_movies", [])
        if "preferred_genres" in message:
            user_data["preferred_genres"] = message.get("preferred_genres", user_data["preferred_genres"])
        #user_data["liked_movies"] = list(set(user_data["liked_movies"]))
        self.save_user(user_data)
        return {"status": "success", "message": f"User {email} updated successfully. with liked movies {user_data['liked_movies']}, dislike movies "
                           f"{user_data['disliked_movies']} "
                           f"preferred genres {user_data['preferred_genres']}"}


class UserServiceMicroservice:
    def __init__(self):
        self.service = UserService()
        self.service = UserService()
        self.lock = threading.Lock()

    def handle_request(self, message):
        """Handle individual requests."""
        global service_status

        request_type = message.get("request_type")
        response = {"status": "error", "message": "Unknown request"}

        try:
            if request_type == "register_user":
                response = self.service.register_user(message)
            elif request_type == "get_user":
                with self.lock:  # Ensure thread-safe access
                    user_data = self.service.load_user(message["email"])
                response = {"status": "success", "user": user_data}
            elif request_type == "login_user":
                response = self.service.login_user(message)
            elif request_type == "update_user":
                response = self.service.update_user(message)
            elif request_type == "status":
                response = {"status": service_status}
            elif request_type == "terminate":
                print("Termination signal received. Shutting down...")
                service_status = "stopped"
                response = {"status": "stopped"}
            else:
                response = {"error": "Invalid command"}
        except Exception as e:
            print(f"Error: {e}")
            response = {"status":"error","error": "Internal server error"}

        return response

    def run(self):
        """Main loop for the User Service with ThreadPoolExecutor."""
        global service_status
        print("User Service is running...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            while service_status == "running":
                try:
                    # Wait for a request
                    message = json.loads(socket.recv_json())
                    print(f"Received request: {message}")

                    # Submit the request to the thread pool
                    future = executor.submit(self.handle_request, message)

                    # Send response back to client
                    response = future.result()  # Get the response from the worker thread
                    socket.send_json(json.dumps(response))

                except zmq.ZMQError as e:
                    print(f"ZeroMQ Error: {e}")
                    socket.send_json(json.dumps({"error": "Internal server error"}))
                except Exception as e:
                    print(f"Error: {e}")
                    socket.send_json(json.dumps({"error": "Internal server error"}))
                sys.stdout.flush()
                sys.stderr.flush()
        print("User Service stopped.")


if __name__ == "__main__":
    user_service = UserServiceMicroservice()
    user_service.run()
