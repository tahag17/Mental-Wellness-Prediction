# main.py
from predict_service import predict_new_user

# Example high wellness user
user_high = {
    'age': 30,
    'screen_time_hours': 5,
    'work_screen_hours': 2,
    'leisure_screen_hours': 3,
    'sleep_hours': 8,
    'sleep_quality_1_5': 5,
    'stress_level_0_10': 1,
    'productivity_0_100': 95,
    'exercise_minutes_per_week': 120,
    'social_hours_per_week': 10,
    'gender': 'male',                 # match training lowercase
    'occupation': 'student',          # match training lowercase
    'work_mode': 'remote'             # match training lowercase
}

# Example low wellness user
user_low = {
    'age': 30,
    'screen_time_hours': 12,
    'work_screen_hours': 8,
    'leisure_screen_hours': 4,
    'sleep_hours': 4,
    'sleep_quality_1_5': 1,
    'stress_level_0_10': 9,
    'productivity_0_100': 40,
    'exercise_minutes_per_week': 20,
    'social_hours_per_week': 2,
    'gender': 'male',
    'occupation': 'student',
    'work_mode': 'remote'
}

print("📈 High wellness user index:", predict_new_user(user_high))
print("📉 Low wellness user index:", predict_new_user(user_low))
