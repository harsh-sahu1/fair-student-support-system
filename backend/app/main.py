"""
Main FastAPI application entry point.
Implements lifespan model loading, explicit CORS configuration, structured logging,
and robust JSON error handling.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import (
    CORS_ORIGINS,
    MODEL_PATH,
    METRICS_PATH,
    VALIDATION_PREDICTIONS_PATH,
)
from backend.app.ml.model import StudentSupportModel
from backend.app.ml.explainability import FeatureExplainer
from backend.app.api.routes import router as api_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("student_support_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Loads trained model artifacts and validation metrics once at startup.
    Fails fast or trains automatically if artifacts do not exist.
    """
    logger.info("Initializing Student Support AI backend...")
    
    # Check if artifacts exist; if not, train automatically
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        logger.warning(
            f"Artifact '{MODEL_PATH}' not found. Triggering automated offline training..."
        )
        from backend.scripts.train import run_training
        run_training()
        logger.info("Automated training complete. Artifacts created.")
        
    try:
        # Load model pipeline into app state
        logger.info(f"Loading trained model from {MODEL_PATH}")
        model = StudentSupportModel.load(str(MODEL_PATH))
        explainer = FeatureExplainer(model)
        app.state.model = model
        app.state.explainer = explainer
        
        # Load precomputed validation predictions and metrics summary
        logger.info(f"Loading validation metrics from {METRICS_PATH}")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            app.state.metrics_summary = json.load(f)
            
        with open(VALIDATION_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            app.state.validation_students = json.load(f)
            
        logger.info("Model and validation artifacts successfully loaded into memory.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to load model artifacts: {str(e)}", exc_info=True)
        raise RuntimeError(f"Startup failure: {str(e)}") from e
        
    yield
    
    logger.info("Shutting down Student Support AI backend...")


app = FastAPI(
    title="Fair Student-Support Prioritization API",
    version="1.0.0",
    description=(
        "Decision-support system for allocating limited academic intervention capacity (20%) "
        "while actively monitoring and mitigating group recall disparities under educational fairness benchmarks."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Explicit CORS configuration (not wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Converts Pydantic/FastAPI validation errors into clean, field-specific JSON errors.
    """
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        errors.append({
            "field": field_loc,
            "message": msg,
            "type": err.get("type", "value_error")
        })
        
    logger.warning(f"Request validation failure on {request.method} {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "message": "Input validation failed. Please check field requirements and ensure no leakage attributes (G3) are included.",
            "details": errors
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions to prevent stack trace leaks to the client.
    """
    logger.error(
        f"Unhandled server exception on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the request. Details have been logged server-side."
        }
    )


# Mount API routes
app.include_router(api_router, tags=["Prioritization & Fairness"])


@app.get("/", summary="API Root", tags=["System"])
async def root():
    """
    Root endpoint directing clients and users to API documentation and key resources.
    """
    return {
        "name": "Fair Student-Support Prioritization API",
        "version": "1.0.0",
        "status": "online",
        "message": "Welcome to the Fair Student-Support Prioritization API. Navigate to /docs for interactive documentation.",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
        "frontend_url": "http://127.0.0.1:5173",
        "endpoints": {
            "health": "/health",
            "overview": "/overview",
            "students": "/students",
            "fairness": "/fairness",
            "performance": "/performance",
            "predict": "/predict"
        }
    }

