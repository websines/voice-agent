from fastapi import FastAPI, Request
from voice_agent.utils.logger import configure_logging
import structlog

logger = configure_logging()

app = FastAPI(title="Voice Agent API")

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request.headers.get("X-Request-ID", "unknown"),
        path=request.url.path,
        method=request.method
    )
    
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "request_failed",
            exception=str(exc),
            exc_info=True
        )
        raise
    
    logger.info(
        "request_completed",
        status_code=response.status_code,
        duration_ms=0  # Would calculate actual duration
    )
    return response

@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "ok"}
