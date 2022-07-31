from locust import HttpUser, task

class AppUser(HttpUser):
  
    @task
    def home_page(self):
        self.client.get("/")
