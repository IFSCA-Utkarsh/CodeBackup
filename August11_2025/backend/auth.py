import csv
from typing import Optional
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config import SETTINGS

router = APIRouter()

SESSIONS = {}  # token -> user_id

class LoginRequest(BaseModel):
    user_id: str
    password: str

CSV_PATH = "customers.csv"


def read_users(csv_path: str = CSV_PATH):
    users = {}
    try:
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                users[row["user_id"]] = row["password"]
    except FileNotFoundError:
        # No CSV found — return empty dict (allow dev/test override)
        return {}
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


@router.post("/login")
def login(req: LoginRequest):
    token = authenticate(req.user_id, req.password)
    if token:
        return {"token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")