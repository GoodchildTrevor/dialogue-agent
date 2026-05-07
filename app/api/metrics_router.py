from fastapi import APIRouter
from fastapi.responses import Response
from app.metrics import collect_metrics, get_metrics_output

router = APIRouter(tags=["observability"])

@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint — pulls latest rows from `traces` first."""
    await collect_metrics()
    body, content_type = get_metrics_output()
    return Response(content=body, media_type=content_type)