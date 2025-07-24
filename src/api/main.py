# ============================================
# File: src/api/main.py
# ============================================
import sys
import time

# from pathlib import Path
from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

try:
    import src.churn_model.predict as prediction_service
    from src.api.endpoints import predict as predict_endpoint
    from src.churn_model.config import AppConfig, load_config
    from src.churn_model.utils import setup_logging
except ImportError as e:
    print(
        f"ERROR: Failed to import necessary modules: {e}.",
        "Check PYTHONPATH and project structure.",
        file=sys.stderr,
    )

    sys.exit(1)


try:
    config = load_config()
    api_cfg = config.api
except Exception as e:
    print(
        f"ERROR: Failed to load configuration for API: {e}. Using defaults.",
        file=sys.stderr,
    )
    api_cfg = type(
        "obj",
        (object,),
        {"title": "Churn Prediction API (Default)", "version": "0.0.1"},
    )()


setup_logging()
logger.info("Logging configured for API.")


app = FastAPI(
    title=api_cfg.title,
    version=api_cfg.version,
    description="API to predict customer churn based on input features.",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Adds a header 'X-Process-Time' to the response indicating the processing time.

    This middleware measures the time taken to process each request and adds
    the 'X-Process-Time' header to the response with the duration in seconds.
    It also logs the request method, URL path, processing time, and status code.

    Args:
        request (Request): The incoming request object.
        call_next: The next function in the middleware chain.

    Returns:
        Response: The response object with the added 'X-Process-Time' header.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(
        f"{request.method} {request.url.path} - Completed in {process_time:.4f}s - Status: {response.status_code}"
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        f"Request validation error: {exc.errors()} for request URL: {request.url}"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(
        f"HTTP Exception caught: {exc.status_code} - {exc.detail} for request URL: {request.url}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    logger.exception(f"Unhandled exception during request to {request.url}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred."},
    )


@app.get("/", tags=["Health"], summary="Basic API Health Check")
async def health_check():
    """Returns the operational status of the API and the prediction model."""
    logger.debug("Health check endpoint called.")

    if prediction_service._model_load_error():
        model_status = "error"
        logger.warning(
            f"Health check reports model error: {prediction_service._model_load_error()}"
        )
    elif prediction_service._config_cache():
        model_status = "loaded"
    else:
        model_status = "not initialized"

    return {"status": "ok", "model_status": model_status}


api_router_v1 = APIRouter(prefix="/api/v1")
api_router_v1.include_router(predict_endpoint.router, tags=["Prediction"])

app.include_router(api_router_v1)


@app.on_event("startup")
async def startup_event():
    """Log API startup status."""
    logger.info("--- API Startup Event Triggered ---")
    if prediction_service._model_load_error():
        logger.critical(
            f"API Startup: Prediction service failed to initialize: {prediction_service._model_load_error()}"
        )
    elif prediction_service._config_cache():
        logger.info("API Startup: Prediction service appears initialized.")
    else:
        logger.warning(
            "API Startup: Prediction service initialization status unknown or pending."
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Log API shutdown event."""
    logger.info("--- API Shutdown Event Triggered ---")
