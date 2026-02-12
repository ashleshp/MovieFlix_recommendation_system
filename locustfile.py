from locust import HttpUser, task, between
import random
import string
import uuid

class FastAPITestUser(HttpUser):
    wait_time = between(1, 3)  # Simulate wait time between tasks (1-3 seconds)

    @task
    def register_and_login(self):
        """Simulate user registration and login."""
        # Generate a random email and password for each user
        email = f"user{uuid.uuid4()}@test.com"
        password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

        # Step 1: Register the user
        self.client.post(r"/register", json={
            "email": email,
            "password": password,
            "age": random.randint(18, 60),
            "country": "USA",
            "preferred_genres": ["Horror", "Comedy"]
        })

        # # Step 2: Login to get the JWT token
        # response = self.client.post(r"/login", json={
        #     "email": email,
        #     "password": password
        # })
        #
        # if response.status_code == 200:
        #     token = response.json().get("access_token")
        #     headers = {"Authorization": f"Bearer {token}"}
        #
        #     # Step 3: Request recommendations
        #     self.client.post("/recommend", json={"num_recommendations": 5}, headers=headers)
    #
    # @task
    # def submit_feedback(self):
    #     """Simulate user feedback."""
    #     feedback_data = {"movie1": "like", "movie2": "dislike"}
    #     token = "your_jwt_token_here"  # Replace with a valid token for testing
    #     headers = {"Authorization": f"Bearer {token}"}
    #
    #     self.client.post("/feedback", json={"feedback": feedback_data}, headers=headers)
