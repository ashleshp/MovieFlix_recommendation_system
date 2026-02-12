import json

import pandas as pd
import zmq
import os
import time
from threading import Thread
import sys
# Define the port for the Feedback Service
PORT = 12351
PROFILE_GENERATOR_PORT = 12348  # Profile Generator Service
RECOMMENDATION_SERVICE_PORT = 12349  # Recommendation Service
USER_INTERFACE_PORT = 12345  # User Interface Service

# ZeroMQ Context
context = zmq.Context()

# REP Socket for FastAPI communication
socket = context.socket(zmq.REP)
socket.bind(f"tcp://127.0.0.1:{PORT}")

# Service Status
service_status = "running"
class FeedbackService:
    """
    Handles user feedback, updates user profiles, and generates new recommendations.
    """
    def __init__(self):
        # Connect to Profile Generator, Recommendation, and User Interface services
        self.profile_generator_socket = context.socket(zmq.REQ)
        self.profile_generator_socket.connect(f"tcp://127.0.0.1:{PROFILE_GENERATOR_PORT}")

        self.recommendation_socket = context.socket(zmq.REQ)
        self.recommendation_socket.connect(f"tcp://127.0.0.1:{RECOMMENDATION_SERVICE_PORT}")

        self.user_interface_socket = context.socket(zmq.REQ)
        self.user_interface_socket.connect(f"tcp://127.0.0.1:{USER_INTERFACE_PORT}")
        self.movies_df = pd.read_csv("filtered_movies_data.csv")

    def update_recommendations(self, email, feedback):
        """
        Updates recommendations based on user feedback.
        """
        print("fetching user")
        # Step 1: Fetch user data from User Interface Service
        self.user_interface_socket.send_json(json.dumps({
            "request_type": "get_user",
            "email": email
        }))
        user_response = self.user_interface_socket.recv_json()
        print(user_response)
        user_data = json.loads(user_response)

        if "error" in user_data:
            print(user_data["error"])
            return {"status": "error", "message": user_data["error"]}
        print("got user")
        user_profile = user_data["user"]
        liked_movies = []
        disliked_movies = []
        for k, v in feedback.items():
            movie_row = self.movies_df.loc[self.movies_df['movie_id'].astype(str) == str(k)]
            title = movie_row.iloc[0]['title']
            if v == 'like':
                liked_movies.append(title)
            else:
                disliked_movies.append(title)
        print(f"Parsed movie details \n liked : {liked_movies}\n disliked: {disliked_movies}")
        # Step 2: Update profile vector via Profile Generator Service
        self.profile_generator_socket.send_json(json.dumps({
            "request_type": "update_profile",
            "email": email,
            "liked_movies": liked_movies,
            "disliked_movies": disliked_movies
        }))
        profile_response = self.profile_generator_socket.recv_json()
        profile_response = json.loads(profile_response)

        if profile_response["status"] != "success":
            return {"status": "error", "message": "Failed to update profile vector."}
        self.user_interface_socket.send_json(json.dumps({
            "request_type": "update_user",
            "email": email,
            "liked_movies": liked_movies,
            "disliked_movies": disliked_movies,
        }))
        feedback_response = self.user_interface_socket.recv_json()
        feedback_response = json.loads(feedback_response)
        if feedback_response["status"] != "success":
            return {"status": "error", "message": "Failed to update user's liked and disliked movies."}
        print("profile updated")
        # profile_response = self.profile_generator_socket.send_json
        # # Step 3: Request updated recommendations
        # self.recommendation_socket.send_json(json.dumps({
        #     "request_type": "recommend_movies",
        #     "profile_vector": profile_response["profile_vector"],
        #     "liked_movies": user_profile.get("liked_movies", []),
        #     "disliked_movies": user_profile.get("disliked_movies", []),
        #     "num_recommendations": 5
        # }))
        # recommendations_response = self.recommendation_socket.recv_json()
        # recommendations_response = json.loads(recommendations_response)

        return {
            "status": "success",
            "message": f"feedback submitted for {email}"
        }

    def handle_request(self, message):
        """
        Handle incoming requests for the Feedback Service.
        """
        global service_status
        request_type = message.get("request_type")
        if request_type == "update_feedback":
            email = message.get("email")
            feedback = message.get("feedback", {})
            print(email,feedback)
            return self.update_recommendations(email, feedback)

        elif request_type == "status":
            return {"status": service_status}

        elif request_type == "terminate":
            service_status = "stopped"
            return {"status": "stopped"}

        else:
            return {"error": "Unknown request type."}

    def run(self):
        """
        Main loop to run the Feedback Service.
        """
        global service_status
        print("Feedback Service is running...")
        while service_status == "running":
            try:
                message = socket.recv_json()
                #print(f"Received request: {message}")
                response = self.handle_request(json.loads(message))
                socket.send_json(json.dumps(response))
                sys.stdout.flush()
            except Exception as e:
                print(f"Error: {e}")
                sys.stderr.flush()
                socket.send_json(json.dumps({"status": "error", "message": "Internal server error"}))

        print("Feedback Service stopped.")

if __name__ == "__main__":
    service = FeedbackService()
    service.run()
