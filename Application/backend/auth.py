import csv
import os
from typing import Optional
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Secret key for JWT (keep secure)
SECRET_KEY = os.getenv("JWT_SECRET", "supersecretjwtkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Path to employee credentials CSV
EMPLOYEE_CSV = os.getenv("EMPLOYEE_CSV", "employees.csv")

# Load employees into a dict {id: password}
def load_employees():
    employees = {}
    with open(EMPLOYEE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            employees[row["id"]] = row["password"]
    return employees

EMPLOYEES = load_employees()

# OAuth2 scheme (reads Authorization header)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Token creation
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Token validation
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None or user_id not in EMPLOYEES:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
