from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import ml_inference

app = FastAPI(title="Fairwork AI Backend")

# ✅ CORS (safe for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(ml_inference.router)

# ✅ HEALTH CHECK (THIS WAS MISSING)
@app.get("/health")
def health():
    return {"status": "ok"}
