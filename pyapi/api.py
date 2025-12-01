from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from predict_service import predict_new_user

app = FastAPI(title="Mental Wellness Predictor")

# ----- CORS setup -----
# Allow requests from any origin
app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # allows all HTTP methods
    allow_headers=["*"],  # allows all headers
)
# Define request data structure
class UserData(BaseModel):
    age: int
    screen_time_hours: float
    work_screen_hours: float
    leisure_screen_hours: float
    sleep_hours: float
    sleep_quality_1_5: int
    stress_level_0_10: int
    productivity_0_100: int
    exercise_minutes_per_week: int
    social_hours_per_week: int
    gender: str
    occupation: str
    work_mode: str

@app.post("/predict")
def predict(user: UserData):
    user_dict = user.dict()
    wellness_index = predict_new_user(user_dict)
    return {"mental_wellness_index": wellness_index}
