
import os
import random
import ast
import zmq
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
from fastapi import FastAPI, HTTPException, Depends, Response
from pydantic import BaseModel
from threading import Thread
import json
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
import requests, csv
# FastAPI App
app = FastAPI(title="FastAPI Server - Recommendation System")
# Prometheus Metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request Latency", ["method", "endpoint"])
MOVIE_RECOMMENDATION_COUNT = Counter("movie_recommendations_total", "Total movie recommendations", ["movie"])
RECOMMENDATION_REQUESTS = Counter("recommendation_requests_total", "Total recommendation requests", ["user"])
# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.middleware("http")
async def prometheus_middleware(request, call_next):
    method = request.method
    endpoint = request.url.path
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()

    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    return response

# JWT Configuration
SECRET_KEY = "f084e6c9bdfbce13bce4cb02fb1a652909b0e59bf95d0cf809721c9572446bfb"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 20

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ZeroMQ Setup for Communication with Microservices
context = zmq.Context()
# user_service_socket = context.socket(zmq.REQ)
# recommendation_service_socket = context.socket(zmq.REQ)
# profile_generator_socket = context.socket(zmq.REQ)
# profile_generator_socket.connect("tcp://127.0.0.1:12348")  # Profile Generator Service Port

# Correct Ports for Microservices
# user_service_socket.connect("tcp://127.0.0.1:12345")  # User Service Port
# recommendation_service_socket.connect("tcp://127.0.0.1:12349")  # Recommendation Service Port

# Define a REP socket for heartbeat/status handling
HEARTBEAT_PORT = 12347
heartbeat_socket = context.socket(zmq.REP)
heartbeat_socket.bind(f"tcp://127.0.0.1:{HEARTBEAT_PORT}")

# Service Status
service_status = "running"

# Pydantic Models
class RegisterUser(BaseModel):
    email: str
    password: str
    age: int
    country: str
    preferred_genres: list

class LoginUser(BaseModel):
    email: str
    password: str

class UpdateUser(BaseModel):
    liked_movies: list = []
    disliked_movies: list = []
    preferred_genres: list = []
    change_genre: bool

class RecommendationRequest(BaseModel):
    num_recommendations: int = 5

class FeedbackRequest(BaseModel):
    feedback: dict  # Example: {"movie1": "like", "movie2": "dislike"}



# Global dictionary to store user session information
user_sessions = {}
user_sessions_lock = Lock()  # To ensure thread safety

# JWT Utility Functions
def hash_password(password: str) -> str:
    """Hash a plain-text password."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Generate a JWT token with an expiration time."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise JWTError
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current user by validating the token."""
    email = decode_access_token(token)
    return email
# Heartbeat Thread
def send_heartbeat():
    """Handles heartbeat/status communication."""
    global service_status
    while service_status == "running":
        try:
            message = json.loads(heartbeat_socket.recv_json())
            if message["request_type"] == "status":
                heartbeat_socket.send_json(json.dumps({"status": service_status}))
            elif message["request_type"] == "terminate":
                service_status = "stopped"
                heartbeat_socket.send_json(json.dumps({"status": "stopped"}))
                break
            else:
                heartbeat_socket.send_json(json.dumps({"error": "Invalid command"}))
        except zmq.ZMQError as e:
            print(f"ZeroMQ Error: {e}")
            break
        time.sleep(2)

heartbeat_thread = Thread(target=send_heartbeat, daemon=True)
heartbeat_thread.start()

# Endpoints
@app.post("/register")
def register_user(user: RegisterUser):
    """Register a new user."""
    hashed_password = hash_password(user.password)
    user_service_socket = context.socket(zmq.REQ)
    user_service_socket.connect("tcp://127.0.0.1:12345")  # User Service Port
    profile_generator_socket = context.socket(zmq.REQ)
    profile_generator_socket.connect("tcp://127.0.0.1:12348")  # Profile Generator Service Port
    user_service_socket.send_json(json.dumps({
        "request_type": "register_user",
        "email": user.email,
        "password": hashed_password,  # Send hashed password
        "age": user.age,
        "country": user.country,
        "preferred_genres": user.preferred_genres
    }))
    response = json.loads(user_service_socket.recv_json())
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])

    # Step 2: Trigger Profile Generator Service to generate profile vector
    profile_generator_socket.send_json(json.dumps({
        "request_type": "generate_profile",
        "email": user.email,
        "preferred_genres": user.preferred_genres
    }))
    profile_response = json.loads(profile_generator_socket.recv_json())
    if "error" in profile_response:
        raise HTTPException(status_code=400, detail="Failed to generate profile vector.")
    user_service_socket.close()
    profile_generator_socket.close()
    # Generate JWT token
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login")
def login_user(user: LoginUser):
    """Login an existing user."""
    user_service_socket = context.socket(zmq.REQ)
    user_service_socket.connect("tcp://127.0.0.1:12345")  # User Service Port
    user_service_socket.send_json(json.dumps({
        "request_type": "login_user",
        "email": user.email,
    }))
    response = json.loads(user_service_socket.recv_json())
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])

    # Verify password
    if not verify_password(user.password, response["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    user_service_socket.close()
    # Generate JWT token
    access_token = create_access_token(data={"sub": user.email})
    with user_sessions_lock:
        user_sessions[user.email] = {
            "model": None,
            "director_recommendation_made": False,
            "token": access_token,
        }
    return {"access_token": access_token, "token_type": "bearer"}

@app.put("/update_profile")
def update_user_profile(user: UpdateUser, current_user: str = Depends(get_current_user)):
    """Update user preferences for the current user."""
    user_service_socket = context.socket(zmq.REQ)
    user_service_socket.connect("tcp://127.0.0.1:12345")  # User Service Port
    profile_generator_socket = context.socket(zmq.REQ)
    profile_generator_socket.connect("tcp://127.0.0.1:12348")  # Profile Generator Service Port
    if user.change_genre:
        os.remove(os.path.join("user_embeddings",f"{current_user}_embedding.pkl"))
        profile_generator_socket.send_json(json.dumps({
            "request_type": "generate_profile",
            "email": current_user,
            "preferred_genres": user.preferred_genres
        }))
        profile_response = json.loads(profile_generator_socket.recv_json())
        if "error" in profile_response:
            raise HTTPException(status_code=400, detail="Failed to generate profile vector on update of genre.")
    user_service_socket.send_json(json.dumps({
        "request_type": "update_user",
        "email": current_user,
        "liked_movies": user.liked_movies,
        "disliked_movies": user.disliked_movies,
        "preferred_genres": user.preferred_genres
    }))
    response = json.loads(user_service_socket.recv_json())
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    user_service_socket.close()
    profile_generator_socket.close()
    return {"message": response["message"]}

def get_director_based_recommendations(liked_movies,disliked_movies):
        if len(liked_movies)<=0:
            return []
        movie = random.choice(liked_movies)
        print(f"Random movie selected is : {movie}")
        result = []
        data_preprocessor_socket = context.socket(zmq.REQ)
        data_preprocessor_socket.connect("tcp://127.0.0.1:12346")
        data_preprocessor_socket.send_json(json.dumps({
            "request_type": "director_based",
            "movie": movie  # Use the email from the token
        }))
        response = json.loads(data_preprocessor_socket.recv_json())
        if response["recommendations"]:
            result = response["recommendations"]
        else:
            return []
        print(f"Director based recommendation :{result}")
        return result
def _get_twotower_based_recs(liked_movies,disliked_movies):
    recommendation_service_socket = context.socket(zmq.REQ)
    recommendation_service_socket.RCVTIMEO = 25000
    recommendation_service_socket.connect("tcp://127.0.0.1:12352")
    user_ratings = {
        'userId': [],
        'title': [],
        'rating': []
    }
    for movie in liked_movies:
        user_ratings['title'].append(movie)
        user_ratings["userId"].append(9999)
        user_ratings['rating'].append(5.0)
    for movie in disliked_movies:
        user_ratings['title'].append(movie)
        user_ratings["userId"].append(9999)
        user_ratings['rating'].append(0.0)   
    recommendation_service_socket.send_json(json.dumps({
        "request_type": "recommend",
        "new_user_preferences": user_ratings,
         "nos": 5   # Use the email from the token
    }))
    recommendation_service_socket_response = json.loads(recommendation_service_socket.recv_json())

    # if "error" in recommendation_service_socket_response:
    #     raise HTTPException(status_code=400, detail=recommendation_service_socket_response["error"])
    # if "success" in recommendation_service_socket_response:
    recommendations = recommendation_service_socket_response.get("recommendations")
    print(f"The community picks are: {recommendations}")
    # else:
    #     return recommendations
    # if not recommendations:
    #     raise HTTPException(status_code=400, detail="Community based recommendation failed")
    recommendation_service_socket.close()
    return recommendations


def get_community_based_picks(liked_movies,disliked_movies):
    recommendation_service_socket = context.socket(zmq.REQ)
    recommendation_service_socket.RCVTIMEO = 25000
    recommendation_service_socket.connect("tcp://127.0.0.1:12349")
    user_ratings = []
    for movie in liked_movies:
        user_ratings.append((movie,5.0))
    for movie in disliked_movies:
        user_ratings.append((movie,0.0))    
    recommendation_service_socket.send_json(json.dumps({
        "request_type": "get_community_picks",
        "user_ratings": user_ratings  # Use the email from the token
    }))
    recommendation_service_socket_response = json.loads(recommendation_service_socket.recv_json())

    # if "error" in recommendation_service_socket_response:
    #     raise HTTPException(status_code=400, detail=recommendation_service_socket_response["error"])
    # if "success" in recommendation_service_socket_response:
    recommendations = recommendation_service_socket_response.get("recommendations")
    print(f"The community picks are: {recommendations}")
    # else:
    #     return recommendations
    # if not recommendations:
    #     raise HTTPException(status_code=400, detail="Community based recommendation failed")
    recommendation_service_socket.close()
    return recommendations


import math

def clean_data(data):
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            print(data)
            return None
        return data
    return data
@app.post("/recommend")
def get_recommendations(request: RecommendationRequest, current_user: str = Depends(get_current_user)):
    """Fetch movie recommendations."""
    # Initialize sockets for microservices
    recommendation_service_socket = context.socket(zmq.REQ)
    recommendation_service_socket.RCVTIMEO = 25000
    recommendation_service_socket.connect("tcp://127.0.0.1:12349")  # Recommendation Service Port
    user_service_socket = context.socket(zmq.REQ)
    user_service_socket.connect("tcp://127.0.0.1:12345")  # User Service Port
    profile_generator_socket = context.socket(zmq.REQ)
    profile_generator_socket.connect("tcp://127.0.0.1:12348")  # Profile Generator Service Port
    recommendation_cache_socket = context.socket(zmq.REQ)
    recommendation_cache_socket.connect("tcp://127.0.0.1:12350")  # Recommendation Cache Service Port

    # Step 1: Retrieve the user profile vector from the Profile Generator Service
    profile_generator_socket.send_json(json.dumps({
        "request_type": "get_profile_vector",
        "email": current_user  # Use the email from the token
    }))
    profile_response = json.loads(profile_generator_socket.recv_json())

    if "error" in profile_response:
        raise HTTPException(status_code=400, detail=profile_response["error"])

    user_vector = profile_response.get("profile_vector")
    if not user_vector:
        raise HTTPException(status_code=400, detail="User profile vector is missing.")

    # Step 2: Retrieve user data from User Service
    user_service_socket.send_json(json.dumps({
        "request_type": "get_user",
        "email": current_user
    }))
    user_response = user_service_socket.recv_json()
    user_data = json.loads(user_response)["user"]

    # Step 3: Check Recommendation Service status
    recommendation_service_socket.send_json(json.dumps({"request_type": "status"}))
    rec_status = {}
    try:
        rec_status = json.loads(recommendation_service_socket.recv_json())
    except zmq.ZMQError as e:
        print(f"Error receiving from Recommendation Service: {e}")
        rec_status = {"status": "error"}
    response = {}    
    if rec_status.get("status") == "running":
        # Recommendation Service is running, request recommendations
            # Check user-specific switch for two-tower model
        with user_sessions_lock:
            use_two_tower = user_sessions.get(current_user, {}).get("use_two_tower", False)
        
        if use_two_tower:
            # Use two-tower based recommendations.
            response = _get_twotower_based_recs(user_data["liked_movies"], user_data["disliked_movies"])
        else:
            recommendation_service_socket.send_json(json.dumps({
                "request_type": "recommend_movies",
                "profile_vector": user_vector,
                "liked_movies": user_data["liked_movies"],
                "disliked_movies": user_data["disliked_movies"],
                "num_recommendations": request.num_recommendations
            }))
            response = json.loads(recommendation_service_socket.recv_json())

            if "error" in response:
                raise HTTPException(status_code=400, detail=response["error"])

            # Step 4: Store recommendations in cache
            recommendation_cache_socket.send_json(json.dumps({
                "request_type": "store_cache",
                "email": current_user,
                "recommendations": response["recommendations"]
            }))
            cache_response = json.loads(recommendation_cache_socket.recv_json())
            if cache_response.get("status") != "success":
                print(f"Warning: Failed to store cache for {current_user}.")
    else:
        # Recommendation Service is not running, fetch from cache
        recommendation_cache_socket.send_json(json.dumps({
            "request_type": "get_cache",
            "email": current_user
        }))
        cache_response = json.loads(recommendation_cache_socket.recv_json())

        if cache_response.get("status") == "success":
            response = {"recommendations": cache_response["recommendations"]}
        else:
            raise HTTPException(status_code=500, detail="Recommendation Service is unavailable and no cache found.")

    # Close sockets
    user_service_socket.close()
    profile_generator_socket.close()
    recommendation_service_socket.close()
    recommendation_cache_socket.close()
    community_based = get_community_based_picks(user_data["liked_movies"],user_data["disliked_movies"])
    director_based = []
    response["recommendations"] = clean_data(response["recommendations"])
    # if len(user_data["liked_movies"])>0 and not user_sessions[current_user]["director_recommendation_made"]:
    #     user_sessions[current_user]["director_recommendation_made"] = True
    director_based = get_director_based_recommendations(user_data["liked_movies"],user_data["disliked_movies"])
    clean_data(director_based)
    # Increment the Prometheus counter for each recommended movie
    for key,value in response["recommendations"].items():
        MOVIE_RECOMMENDATION_COUNT.labels(movie=value['title']).inc()

    RECOMMENDATION_REQUESTS.labels(user=current_user).inc()


    return {"recommendations": response["recommendations"],"director_based":director_based,"community_picks":community_based}

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest, current_user: str = Depends(get_current_user)):
    """Handle user feedback and return updated recommendations."""
    feedback_socket = context.socket(zmq.REQ)
    feedback_socket.connect("tcp://127.0.0.1:12351")  # Feedback Service Port
    print(request.feedback)
    feedback_socket.send_json(json.dumps({
        "request_type": "update_feedback",
        "email": current_user,
        "feedback": request.feedback
    }))
    response = json.loads(feedback_socket.recv_json())
    feedback_socket.close()
    print(response)
    if response["status"] != "success":
        raise HTTPException(status_code=400, detail=response["message"])

    return {"message": "Feedback submitted successfully.", "message": response["message"]}


@app.get("/profile_details")
def profile_details(current_user: str = Depends(get_current_user)):
    user_service_socket = context.socket(zmq.REQ)
    user_service_socket.connect("tcp://127.0.0.1:12345")  # User Service Port
    user_service_socket.send_json(json.dumps({
        "request_type": "get_user",
        "email": current_user
    }))
    # Assuming the response is a JSON string; decode it
    user_response = json.loads(user_service_socket.recv_json())
    user_service_socket.close()
    
    # Extract user details from the response; assuming the response has a "user" key
    user_data = user_response.get("user", {})
    return {
        "User": user_data.get("name", current_user),
        "Age": user_data.get("age", ""),
        "liked_movies": user_data.get("liked_movies", []),
        "disliked_movies": user_data.get("disliked_movies", [])
    }

class SwitchModelRequest(BaseModel):
    use_two_tower: bool  # True for two-tower based recommendations, False for profile vector

@app.post("/switch_model")
def switch_model(request: SwitchModelRequest, current_user: str = Depends(get_current_user)):
    with user_sessions_lock:
        if current_user not in user_sessions:
            user_sessions[current_user] = {}
        user_sessions[current_user]["use_two_tower"] = request.use_two_tower
    return {"message": f"Recommendation model switched to {'TwoTower' if request.use_two_tower else 'ProfileVector'} for user {current_user}"}

@app.post("/logout")
def logout(current_user: str = Depends(get_current_user)):
    with user_sessions_lock:
        if current_user in user_sessions:
            del user_sessions[current_user]
    return {"message": f"User {current_user} has been logged out successfully."}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/status")
def status():
    """Service status for health checks."""
    return {"service": "FastAPI Server", "status": service_status}



class SearchRequest(BaseModel):
    query: str
    num_results: int = 5  # Default to top 5 search results

@app.post("/search_movies")
def search_movies(request: SearchRequest, current_user: str = Depends(get_current_user)):
    """
    Search for movies using FAISS similarity matching.
    """
    recommendation_service_socket = context.socket(zmq.REQ)
    recommendation_service_socket.RCVTIMEO = 25000
    recommendation_service_socket.connect("tcp://127.0.0.1:12349")  # Recommendation Service Port

    # Send search request to Recommendation Service
    recommendation_service_socket.send_json(json.dumps({
        "request_type": "search_movies",
        "query": request.query,
        "num_results": request.num_results
    }))

    try:
        response = json.loads(recommendation_service_socket.recv_json())
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
    except zmq.ZMQError as e:
        print(f"Error receiving from Recommendation Service: {e}")
        raise HTTPException(status_code=500, detail="Search service unavailable.")

    recommendation_service_socket.close()
    return {"status": "success", "results": response["movies"]}




@app.get("/convert_movie_csv/{movie_id}")
def convert_movie_csv_route(movie_id: int, current_user: str = Depends(get_current_user)):

    print(f"Recived new movie request for : {movie_id}")
    # Replace with your actual Bearer token.
    BEARER_TOKEN = r"eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyNWRjNWQ2YzY2ODNiOTliZTUzZjBmZjk0Yzc2MGQxZCIsIm5iZiI6MTczNzQ5NTkyMi40MjUsInN1YiI6IjY3OTAxNTcyZjNiYTAxOGI3MWYwOTZhZiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.yCWhQlmsX8P5J8zK1Mtytoo8UvvPQbzvUCSJRHVI5gQ"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    
    try:
        # Get movie details.
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
        details_response = requests.get(details_url, headers=headers)
        details_response.raise_for_status()
        movie_details = details_response.json()
    
        # Get movie credits.
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US"
        credits_response = requests.get(credits_url, headers=headers)
        credits_response.raise_for_status()
        credits = credits_response.json()
    
        # Extract genres as a comma-separated string.
        genres_list = movie_details.get("genres", [])
        genres = ", ".join([genre["name"] for genre in genres_list])
    
        # Extract cast: take first three cast members.
        cast_list = credits.get("cast", [])
        cast_names = [member["name"] for member in cast_list][:3]
        cast = ", ".join(cast_names)
    
        # Extract director from crew.
        director = ""
        for member in credits.get("crew", []):
            if member.get("job") == "Director":
                director = member.get("name", "")
                break
    
        # For keywords, we'll leave it blank as they're not provided by these endpoints.
        keywords = ""
    
        # Prepare the CSV data.
        csv_data = {
            "adult": movie_details.get("adult", False),
            "genres": genres,
            "movie_id": movie_details.get("id", ""),
            "imdb_id": movie_details.get("imdb_id", ""),
            "original_language": movie_details.get("original_language", ""),
            "original_title": movie_details.get("original_title", ""),
            "overview": movie_details.get("overview", ""),
            "poster_path": movie_details.get("poster_path", ""),
            "title": movie_details.get("title", ""),
            "vote_average": movie_details.get("vote_average", 0),
            "vote_count": movie_details.get("vote_count", 0),
            "cast": cast,
            "director": director,
            "keywords": keywords,
            "release_date": movie_details.get("release_date", "")
        }
    
        # Specify the CSV column order.
        csv_columns = [
            "adult", "genres", "movie_id", "imdb_id", "original_language", 
            "original_title", "overview", "poster_path", "title", "vote_average", 
            "vote_count", "cast", "director", "keywords", "release_date"
        ]
    
        # Write the CSV file.
        filename = "new_movies.csv"
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(csv_data)




        recommendation_service_socket = context.socket(zmq.REQ)
        recommendation_service_socket.RCVTIMEO = 25000
        recommendation_service_socket.connect("tcp://127.0.0.1:12352")
        recommendation_service_socket.send_json(json.dumps({
            "request_type": "add_new_movies",
        }))
        recommendation_service_socket_response = json.loads(recommendation_service_socket.recv_json())

        # if "error" in recommendation_service_socket_response:
        #     raise HTTPException(status_code=400, detail=recommendation_service_socket_response["error"])
        # if "success" in recommendation_service_socket_response:
        recommendations = recommendation_service_socket_response.get("message")
        print(f"Result of New movie: {recommendations}")
        # else:
        #     return recommendations
        # if not recommendations:
        #     raise HTTPException(status_code=400, detail="Community based recommendation failed")
        recommendation_service_socket.close()    
    
        return {"message": f"CSV file saved as {filename} for movie_id {movie_id}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("FastAPI Server is starting...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
