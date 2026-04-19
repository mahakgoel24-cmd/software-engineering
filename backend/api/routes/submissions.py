from fastapi import APIRouter

router = APIRouter(prefix="/submissions", tags=["Submissions"])

@router.get("/health")
def submissions_health():
    return {"submissions": "ok"}
