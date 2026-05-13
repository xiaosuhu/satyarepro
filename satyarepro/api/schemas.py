from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AuditStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class AuditRequest(BaseModel):
    query: str = Field(..., min_length=3, description="PMID, DOI, or free-text research question.")
    max_iterations: int = Field(default=10, ge=1, le=20)


class AuditSubmitted(BaseModel):
    job_id: str
    status: AuditStatus
    created_at: datetime


class AuditResult(BaseModel):
    job_id: str
    status: AuditStatus
    query: str
    summary: str | None = None
    tool_calls_made: int = 0
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
