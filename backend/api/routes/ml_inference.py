from fastapi import APIRouter
from backend.services.ml_service import MLService
from backend.api.schemas.project import ProjectShortlistRequest

router = APIRouter(prefix="/ml", tags=["ML"])

@router.post("/shortlist")
def shortlist_candidates(payload: ProjectShortlistRequest):
    service = MLService()
    return service.run_pipeline(project_id=payload.project_id)


