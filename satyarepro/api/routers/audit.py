import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from satyarepro.agent.orchestrator import AuditOrchestrator
from satyarepro.client.claude import ClaudeClient
from satyarepro.tools import create_default_registry

from ..schemas import AuditRequest, AuditResult, AuditStatus, AuditSubmitted

router = APIRouter(prefix="/audit", tags=["audit"])

_jobs: dict[str, AuditResult] = {}


async def _run_audit(job_id: str, query: str, max_iterations: int) -> None:
    _jobs[job_id] = _jobs[job_id].model_copy(update={"status": AuditStatus.running})
    try:
        client = ClaudeClient()
        registry = create_default_registry()
        orchestrator = AuditOrchestrator(client, registry, max_iterations=max_iterations)
        report = await orchestrator.audit(query)
        _jobs[job_id] = _jobs[job_id].model_copy(
            update={
                "status": AuditStatus.completed,
                "summary": report.summary,
                "tool_calls_made": report.tool_calls_made,
                "completed_at": datetime.now(timezone.utc),
            }
        )
    except Exception as exc:
        _jobs[job_id] = _jobs[job_id].model_copy(
            update={
                "status": AuditStatus.failed,
                "error": str(exc),
                "completed_at": datetime.now(timezone.utc),
            }
        )


@router.post("", response_model=AuditSubmitted, status_code=202)
async def submit_audit(
    request: AuditRequest, background_tasks: BackgroundTasks
) -> AuditSubmitted:
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _jobs[job_id] = AuditResult(
        job_id=job_id,
        status=AuditStatus.pending,
        query=request.query,
        created_at=now,
    )
    background_tasks.add_task(_run_audit, job_id, request.query, request.max_iterations)
    return AuditSubmitted(job_id=job_id, status=AuditStatus.pending, created_at=now)


@router.get("/{job_id}", response_model=AuditResult)
async def get_audit(job_id: str) -> AuditResult:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return job
