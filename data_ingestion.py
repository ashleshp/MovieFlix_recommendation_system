import json

import zmq
import pandas as pd
import os
import time

# Define the port for the Data Ingestion microservice
PORT = 12312

# ZeroMQ Context
context = zmq.Context()

# Define a REP socket for handling incoming status/terminate requests
socket = context.socket(zmq.REP)
socket.bind(f"tcp://127.0.0.1:{PORT}")

# Status of the service
service_status = "running"

# Path to dataset
DATASET_PATH = "new_dataset"


class DataIngestion:
    """
    A microservice responsible for loading and preparing data.
    """
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path
        self.ratings_df = None
        self.movies_df = None
        self.tags_df = None

    def load_data(self):
        """
        Loads the dataset into pandas DataFrames.
        """
        try:
            # Define paths for CSV files
            ratings_file = os.path.join(self.dataset_path, "ratings.csv")
            movies_file = os.path.join(self.dataset_path, "movies_metadata.csv")
            tags_file = os.path.join(self.dataset_path, "keywords.csv")

            # Load data into DataFrames
            print("Loading data into DataFrames...")
            self.ratings_df = pd.read_csv(ratings_file)
            self.movies_df = pd.read_csv(movies_file)
            self.tags_df = pd.read_csv(tags_file)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error while loading data: {e}")

    def run(self):
        """
        Run the main loop of the service to load data periodically.
        """
        global service_status
        print("Data Ingestion Service is running...")
        self.load_data()  # Initial data load
        while service_status == "running":
            # Wait for a request
            try:
                message = json.loads(socket.recv_json())
                print(message)
                if message["request_type"] == "status":
                    # Respond with service status
                    socket.send_json(json.dumps({"status": service_status}))
                elif message["request_type"] == "terminate":
                    # Gracefully terminate the service
                    print("Termination signal received. Shutting down...")
                    service_status = "stopped"
                    socket.send_json(json.dumps({"status": "stopped"}))
                    socket.close()
                    break
                elif message["request_type"] == "request_data":
                    resp = {
                        "status": "success",
                        "ratings": self.ratings_df.to_json(orient="records"),
                        "movies": self.movies_df.to_json(orient="records"),
                        "tags": self.tags_df.to_json(orient="records"),
                    }
                    socket.send_json(json.dumps(resp))
                else:
                    socket.send_json(json.dumps({"status":"Invalid command"}))
            except zmq.ZMQError as e:
                print(f"ZeroMQ Error: {e}")
                break

            # Simulate periodic tasks (e.g., checking for new data)
            time.sleep(5)  # This keeps the service alive and avoids high CPU usage

        print("Data Ingestion Service stopped.")


if __name__ == "__main__":
    data_ingestion_service = DataIngestion()
    data_ingestion_service.run()
