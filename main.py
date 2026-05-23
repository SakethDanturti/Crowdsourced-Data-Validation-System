from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from validator import analyze_submission


app = FastAPI()


# =====================================
# CORS
# =====================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)


# =====================================
# HOME
# =====================================

@app.get("/")

def home():

    return {

        "message": "AI Validation API Running"

    }


# =====================================
# VALIDATION ENDPOINT
# =====================================

@app.post("/validate")

def validate_submission(data: dict):

    result = analyze_submission(data)

    return result