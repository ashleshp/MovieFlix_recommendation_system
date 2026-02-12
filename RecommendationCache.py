import json
import zmq
import os
import time
from threading import Thread, Lock

# Define the port for the Caching Service
PORT = 12350  # Caching Service Port

# ZeroMQ Context
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://127.0.0.1:{PORT}")

# Service Status
service_status = "running"
cache_lock = Lock()  # Ensure thread safety for cache operations

class CachingService:
    """
    A microservice responsible for caching and retrieving recommendations.
    """
    def __init__(self, cache_file="recommendation_cache.pkl"):
        self.cache_file = cache_file
        self.recommendation_cache = self.load_cache()

    def load_cache(self):
        """
        Load cached recommendations from a file.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as file:
                print("Loaded existing recommendation cache.")
                return json.load(file)
        return {}

    def save_cache(self):
        """
        Save the current cache to a file.
        """
        with open(self.cache_file, "w") as file:
            json.dump(self.recommendation_cache, file)
        print("Cache saved successfully.")

    def store_cache(self, email, recommendations):
        """
        Store recommendations in the cache.
        """
        with cache_lock:
            self.recommendation_cache[email] = recommendations
            self.save_cache()
        return {"status": "success", "message": f"Cache stored for {email}."}

    def get_cache(self, email):
        """
        Retrieve cached recommendations for a given email.
        """
        with cache_lock:
            recommendations = self.recommendation_cache.get(email)
            if recommendations is None:
                return {"status": "error", "message": f"No cache found for {email}."}
            return {"status": "success", "recommendations": recommendations}


class CachingServiceMicroservice:
    def __init__(self):
        self.service = CachingService()

    def handle_request(self, message):
        """
        Handle incoming requests for caching operations.
        """
        global service_status
        request_type = message.get("request_type")

        if request_type == "store_cache":
            email = message.get("email")
            recommendations = message.get("recommendations", [])
            return self.service.store_cache(email, recommendations)

        elif request_type == "get_cache":
            email = message.get("email")
            return self.service.get_cache(email)

        elif request_type == "status":
            return {"status": service_status}

        elif request_type == "terminate":
            service_status = "stopped"
            return {"status": "stopped"}

        else:
            return {"error": "Unknown request type."}

    def run(self):
        """
        Main loop to run the Caching Service.
        """
        global service_status
        print("Caching Service is running...")

        while service_status == "running":
            try:
                message = json.loads(socket.recv_json())
                print(f"Received request: {message}")
                response = self.handle_request(message)
                socket.send_json(json.dumps(response))
            except Exception as e:
                print(f"Error: {e}")
                socket.send_json(json.dumps({"error": "Internal server error"}))

        print("Caching Service stopped.")


if __name__ == "__main__":
    service = CachingServiceMicroservice()
    service.run()
