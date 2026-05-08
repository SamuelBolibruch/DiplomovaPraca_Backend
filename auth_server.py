import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys

from load_auth_data import download_latest_authentication_attempt
from auth_vector_creation import build_auth_vector_for_user
from AUTHENTICATION import run_authentication

app = FastAPI(title="Behavioral Authentication API")


class AuthRequest(BaseModel):
    uid: str
    auth_type: str  # "general" alebo "personal"


class RegisterRequest(BaseModel):
    uid: str


@app.post("/register")
def register(req: RegisterRequest):
    result = subprocess.run(
        [sys.executable, "run_models_creation_pipeline.py", "--uid", req.uid]
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Pipeline zlyhala pre používateľa {req.uid}.")
    return {"status": "ok", "message": f"Modely pre používateľa {req.uid} úspešne natrénované."}


@app.post("/authenticate")
def authenticate(req: AuthRequest):
    try:
        # 1. stiahni posledné autentifikačné dáta
        download_latest_authentication_attempt(
            uid=req.uid,
            auth_type=req.auth_type,
        )

        # 2. vytvor autentifikačný vektor
        build_auth_vector_for_user(req.uid, req.auth_type)

        # 3. spusti model
        result = run_authentication(
            uid=req.uid,
            auth_type=req.auth_type,
        )

        return {
            "status": "ok",
            "result": result
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))