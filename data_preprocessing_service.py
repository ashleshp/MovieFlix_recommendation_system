import json
import sys

import zmq
import pandas as pd
import os
import time
from tmdbv3api import TMDb, Movie
from rapidfuzz import process
import requests
import traceback
# Define the port for the Data Preprocessing microservice
PORT = 12346
DATA_INGESTION_PORT = 12312

# ZeroMQ Context
context = zmq.Context()

# Define a REP socket for handling incoming status/terminate requests
socket = context.socket(zmq.REP)
socket.bind(f"tcp://127.0.0.1:{PORT}")

# Status of the service
service_status = "running"
tmdb = TMDb()
tmdb.api_key = '25dc5d6c6683b99be53f0ff94c760d1d'  # Replace with your TMDb API key
tmdb.language = 'en'
movie_api = Movie()
class DataPreprocessing:
    """
    A microservice responsible for preprocessing data received from Data Ingestion service.
    """
    def __init__(self):
        self.final_df = None
        self.unique_genres = set()
        self.ratings_df = None
        self.movies_df = None
        self.tags_df = None
        self.new_df = pd.read_csv("filtered_movies_data.csv")

    def preprocess_data(self, ratings_df, movies_df, tags_df):
        """
        Preprocesses the data to create a final DataFrame with:
        movie title, movie genres (separated by commas), tags (separated by commas), and average rating.
        """
        print("Preprocessing data...")

        # Calculate average rating for each movie
        avg_ratings_df = ratings_df.groupby('movieId')['rating'].mean().reset_index()
        avg_ratings_df.rename(columns={'rating': 'average_rating'}, inplace=True)

        # Merge movies with average ratings
        movies_with_ratings_df = pd.merge(movies_df, avg_ratings_df, on='movieId', how='left')

        # Merge tags, concatenate distinct tags for each movie
        tags_df['tag'] = tags_df['tag'].fillna('')
        tags_grouped = tags_df.groupby('movieId')['tag'].apply(lambda x: ', '.join(sorted(set(x)))).reset_index()

        # Merge movies with tags
        movies_with_tags_df = pd.merge(movies_with_ratings_df, tags_grouped, on='movieId', how='left')
        movies_with_tags_df['tag'] = movies_with_tags_df['tag'].fillna('')

        # Replace '|' with ',' in genres
        movies_with_tags_df['genres'] = movies_with_tags_df['genres'].str.replace('|', ', ')

        # Select final columns
        final_df = movies_with_tags_df[['title', 'genres', 'tag', 'average_rating']]
        self.final_df = final_df

        # Extract available genres
        unique_genres = set()
        for genres in self.final_df['genres']:
            unique_genres.update(genres.split(','))
        self.unique_genres = [genre.strip() for genre in unique_genres]
        print(f"Available genres: {', '.join(unique_genres)}")
        print(f"Shape of the data frame: {final_df.shape} \n and df info {final_df.info()}")
        print("Data preprocessing completed.")
        return final_df

    import pandas as pd


    # def GetDirectorBased(self,chosen_movie):
    #     # Check if the movie exists in the DataFrame
    #     # return recommended_movies
    #     best_match, score, _ = process.extractOne(chosen_movie, self.new_df['title'].values)
    #
    #     # If no match is close enough (e.g., score < threshold), return an error
    #     if score < 60:  # Threshold for similarity (adjust as needed)
    #         return {"error": f"No close match found for '{chosen_movie}' in the dataset."}
    #
    #     # Find the director of the closest match
    #     director = self.new_df.loc[self.new_df['title'] == best_match, 'director'].values[0]
    #
    #     # Filter movies by the same director, excluding the matched movie
    #     movies_by_director = self.new_df[(self.new_df['director'] == director) & (self.new_df['title'] != best_match)]
    #
    #     # Get up to max_movies titles
    #     recommended_movies = movies_by_director['title'].head(5).tolist()
    #
    #     return {
    #         "matched_movie": best_match,
    #         "director": director,
    #         "recommendations": recommended_movies
    #     }
    def get_movie_image(self, movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyNWRjNWQ2YzY2ODNiOTliZTUzZjBmZjk0Yzc2MGQxZCIsIm5iZiI6MTczNzQ5NTkyMi40MjUsInN1YiI6IjY3OTAxNTcyZjNiYTAxOGI3MWYwOTZhZiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.yCWhQlmsX8P5J8zK1Mtytoo8UvvPQbzvUCSJRHVI5gQ"
        }

        try:
            response = requests.get(url, headers=headers,timeout=1)
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
    def GetDirectorBased(self, chosen_movie):
        # Check if the movie exists in the DataFrame
        # Use fuzzy matching to find the best match
        best_match, score, _ = process.extractOne(chosen_movie, self.new_df['title'].values)

        # If no match is close enough (e.g., score < threshold), return an error
        if score < 60:  # Threshold for similarity (adjust as needed)
            return {"error": f"No close match found for '{chosen_movie}' in the dataset."}

        # Find the director of the closest match
        director = self.new_df.loc[self.new_df['title'] == best_match, 'director'].values[0]

        # Filter movies by the same director, excluding the matched movie
        movies_by_director = self.new_df[(self.new_df['director'] == director) & (self.new_df['title'] != best_match)]

        # Get up to max_movies rows
        recommended_movies = movies_by_director.head(5).to_dict(orient='records')
        for id in recommended_movies:
            id['poster_path'] = self.get_movie_image(id['movie_id'])
            print(id['poster_path'])

        return {
            "matched_movie": best_match,
            "director": director,
            "recommendations": recommended_movies  # Full rows as dictionaries
        }

class DataPreprocessingService:
    def __init__(self):
        self.preprocessor = DataPreprocessing()

    def request_data_from_ingestion(self):
        """
        Request data from the Data Ingestion service.
        """
        ingestion_socket = context.socket(zmq.REQ)
        try:
            ingestion_socket.connect(f"tcp://127.0.0.1:{DATA_INGESTION_PORT}")
            ingestion_socket.send_json(json.dumps({"request_type":"request_data"}))
            ingestion_socket.RCVTIMEO = 10000
            response = json.loads(ingestion_socket.recv_json())
            if response["status"] == "success":
                ratings_df = pd.DataFrame(json.loads(response["ratings"]))
                movies_df = pd.DataFrame(json.loads(response["movies"]))
                tags_df = pd.DataFrame(json.loads(response["tags"]))
                return ratings_df, movies_df, tags_df
            else:
                print(f"Error from Data Ingestion service: {response['message']}")
                return None, None, None
        except zmq.ZMQError as e:
            print(f"Error communicating with Data Ingestion service: {e}")
            return None, None, None
        finally:
            ingestion_socket.close()

    def run(self):
        """
        Main loop for the data preprocessing service.
        """
        global service_status
        print("Data Preprocessing Service is running...")
        response = {"status": "error", "message": "Failed to get data from ingestion service"}
        self.preprocessor.ratings_df, self.preprocessor.movies_df, self.preprocessor.tags_df = self.request_data_from_ingestion()
        if self.preprocessor.ratings_df is not None and self.preprocessor.movies_df is not None and self.preprocessor.tags_df is not None:
            final_df = self.preprocessor.preprocess_data(self.preprocessor.ratings_df, self.preprocessor.movies_df, self.preprocessor.tags_df)
            response = {
                "status": "success",
                "final_df": final_df.to_json(orient="records"),
                "unique_genres": self.preprocessor.unique_genres,
            }
        else:
            response = {"status": "error", "message": "Failed to get data from ingestion service"}
        print(response)
        while service_status == "running":
            try:
                # Wait for a request
                message = json.loads(socket.recv_json())
                if message["request_type"] == "process":
                    print("Okay")
                    # # Request data from Data Ingestion service
                    # ratings_df, movies_df, tags_df = self.request_data_from_ingestion()
                    # if ratings_df is not None and movies_df is not None and tags_df is not None:
                    #     final_df = self.preprocessor.preprocess_data(ratings_df, movies_df, tags_df)
                    #     response = {
                    #         "status": "success",
                    #         "final_df": final_df.to_json(orient="records"),
                    #         "unique_genres": self.preprocessor.unique_genres,
                    #     }
                    # else:
                    #     response = {"status": "error", "message": "Failed to get data from ingestion service"}
                    #
                    # # Send response
                    # socket.send_json(response)
                elif message["request_type"] == "director_based":
                    movies = self.preprocessor.GetDirectorBased(message["movie"])
                    if movies:
                        socket.send_json(json.dumps({"status":"success","recommendations":movies}))
                    socket.send_json(json.dumps({"status":"error","message":"Internal Error"}))
                elif message["request_type"] == "status":
                    # Respond with service status
                    socket.send_json(json.dumps({"status": service_status}))


                elif message["request_type"] == "terminate":
                    # Gracefully terminate the service
                    print("Termination signal received. Shutting down...")
                    service_status = "stopped"
                    socket.send_json(json.dumps({"status": "stopped"}))
                    socket.close()
                    break
                else:
                    socket.send_json({"status":"Invalid command"})
                sys.stdout.flush()
            except zmq.ZMQError as e:
                print(f"ZeroMQ Error: {e}")
                print(traceback.print_exc())

            # Simulate periodic tasks (if needed)
            time.sleep(2)

        print("Data Preprocessing Service stopped.")


if __name__ == "__main__":
    service = DataPreprocessingService()
    service.run()
