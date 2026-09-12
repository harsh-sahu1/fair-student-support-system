"""
API route definitions for the Student Support AI backend.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, status

from backend.app.schemas import (
    HealthResponse,
    OverviewResponse,
    StudentListItem,
    StudentDetailResponse,
    FairnessResponse,
    ModelPerformanceResponse,
    StudentInput,
    PredictBatchRequest,
    PredictResponse,
    StudentPrediction,
    FactorDetail,
)
from backend.app.ml.model import StudentSupportModel
from backend.app.ml.ranking import rank_students
from backend.app.ml.explainability import FeatureExplainer, EXPLANATION_DISCLAIMER

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health and Model Status",
    description="Returns API status and verifies that the ML model artifact is loaded in memory."
)
async def health(request: Request) -> HealthResponse:
    model: Optional[StudentSupportModel] = getattr(request.app.state, "model", None)
    is_loaded = model is not None and model.is_fitted
    return HealthResponse(
        status="ok",
        model_loaded=is_loaded,
        version="1.0.0"
    )


@router.get(
    "/overview",
    response_model=OverviewResponse,
    summary="Dashboard Summary Metrics",
    description="Returns high-level cohort allocation metrics, capacity rules, worst-group recall, and fairness gap on the validation benchmark."
)
async def get_overview(request: Request) -> OverviewResponse:
    metrics = getattr(request.app.state, "metrics_summary", None)
    if not metrics or "overview" not in metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Overview metrics not loaded. Artifacts may be missing."
        )
    return OverviewResponse(**metrics["overview"])


@router.get(
    "/students",
    response_model=List[StudentListItem],
    summary="Prioritized Student Cohort",
    description="Returns the full ranked validation cohort with ranks, 0-100 need scores, selection status, demographic metadata, and top contributing factors."
)
async def get_students(
    request: Request,
    selected_only: bool = Query(False, description="Filter to only students selected for support"),
    group: Optional[str] = Query(None, description="Filter by sex (F, M) or school (GP, MS)"),
    limit: Optional[int] = Query(None, ge=1, description="Max students to return")
) -> List[StudentListItem]:
    students = getattr(request.app.state, "validation_students", None)
    if students is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Validation student records not loaded."
        )
        
    filtered = students
    if selected_only:
        filtered = [s for s in filtered if s.get("selected") is True]
        
    if group:
        grp_upper = group.strip().upper()
        filtered = [
            s for s in filtered
            if s.get("sex", "").upper() == grp_upper or s.get("school", "").upper() == grp_upper
        ]
        
    if limit is not None:
        filtered = filtered[:limit]
        
    return [StudentListItem(**item) for item in filtered]


@router.get(
    "/students/{student_id}",
    response_model=StudentDetailResponse,
    summary="Student Profile and Factor Attribution",
    description="Retrieves comprehensive record, raw input features, need score, selection status, and detailed factor contributions for a specific student."
)
async def get_student_detail(student_id: str, request: Request) -> StudentDetailResponse:
    students = getattr(request.app.state, "validation_students", None)
    if students is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Student records not loaded."
        )
        
    target = next((s for s in students if s["student_id"] == student_id), None)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID '{student_id}' not found in current cohort."
        )
        
    return StudentDetailResponse(
        student_id=target["student_id"],
        rank=target["rank"],
        need_score=target["need_score"],
        probability=target["probability"],
        selected=target["selected"],
        sex=target.get("sex", "Unknown"),
        school=target.get("school", "Unknown"),
        ground_truth_support_needed=target.get("ground_truth_support_needed"),
        top_factors=target["top_factors"],
        factor_details=target.get("factor_details", []),
        raw_features=target.get("raw_features", {}),
        disclaimer=EXPLANATION_DISCLAIMER
    )


@router.get(
    "/fairness",
    response_model=FairnessResponse,
    summary="Group-Wise Fairness and Observed Recall Gap",
    description="Returns demographic group breakdown, sample eligibility flags, group-specific recall, and the intra-attribute Observed Recall Gap."
)
async def get_fairness(request: Request) -> FairnessResponse:
    metrics = getattr(request.app.state, "metrics_summary", None)
    if not metrics or "fairness" not in metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fairness audit data not loaded."
        )
    return FairnessResponse(**metrics["fairness"])


@router.get(
    "/model-performance",
    response_model=ModelPerformanceResponse,
    summary="Model Calibration and Robustness Benchmark",
    description="Returns Brier score, recall at capacity, 10-seed cross-validation statistics, and educational interpretation context."
)
async def get_model_performance(request: Request) -> ModelPerformanceResponse:
    metrics = getattr(request.app.state, "metrics_summary", None)
    if not metrics or "robustness" not in metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model performance metrics not loaded."
        )
    return ModelPerformanceResponse(
        overview=OverviewResponse(**metrics["overview"]),
        robustness=metrics["robustness"],
        meta=metrics.get("meta", {})
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Score and Explain Student Need (Submission-Shaped)",
    description="Accepts one or more raw student records without G3/support_needed. Returns probability, 0-100 need score, and non-causal contributing factors."
)
async def predict_students(
    payload: Union[StudentInput, PredictBatchRequest, List[StudentInput]],
    request: Request
) -> PredictResponse:
    model: Optional[StudentSupportModel] = getattr(request.app.state, "model", None)
    explainer: Optional[FeatureExplainer] = getattr(request.app.state, "explainer", None)
    
    if model is None or not model.is_fitted or explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not initialized."
        )
        
    # Unify input format
    if isinstance(payload, StudentInput):
        items = [payload]
    elif isinstance(payload, PredictBatchRequest):
        items = payload.students
    else:
        items = payload
        
    if not items:
        return PredictResponse(predictions=[], count=0)
        
    records = [item.model_dump() for item in items]
    df_in = pd.DataFrame(records)
    
    try:
        probs = model.predict_probabilities(df_in)
        scores = model.predict_need_scores(df_in)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error processing student features: {str(e)}"
        )
        
    predictions: List[StudentPrediction] = []
    for idx, row in df_in.iterrows():
        explanation = explainer.explain_student(row, top_k=3)
        sid = str(row.get("student_id") or f"NEW{idx:04d}")
        
        details = [
            FactorDetail(**fd) for fd in explanation.get("factor_details", [])
        ]
        
        pred = StudentPrediction(
            student_id=sid,
            probability=float(np.round(probs[idx], 4)),
            need_score=float(scores[idx]),
            top_factors=explanation["top_factors"],
            factor_details=details,
            disclaimer=explanation["disclaimer"]
        )
        predictions.append(pred)
        
    return PredictResponse(predictions=predictions, count=len(predictions))
