# backend/api/schemas/project.py

from pydantic import BaseModel
from uuid import UUID

class ProjectShortlistRequest(BaseModel):
    project_id: UUID

