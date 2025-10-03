import csv
from typing import Optional
import uuid

SESSIONS = {}  # session_id -> user_id

def read_users(csv_path="customers.csv"):
    users = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            users[row["user_id"]] = row["password"]
    return users

def authenticate(user_id: str, password: str) -> Optional[str]:
    users = read_users()
    if user_id in users and users[user_id] == password:
        token = str(uuid.uuid4())
        SESSIONS[token] = user_id
        return token
    return None

def is_valid_token(token: str) -> bool:
    return token in SESSIONS
