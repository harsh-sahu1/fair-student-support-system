"""
Pydantic v2 schemas for API validation and serialization.
Strictly prohibits target and leakage fields (G3, support_needed) and validates feature ranges.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator, field_validator


class StudentInput(BaseModel):
    """
    Validated raw student features for inference.
    Features follow the fixed 30-feature schema from the UCI Student Performance benchmark.
    G3 and support_needed are strictly forbidden.
    """
    student_id: Optional[str] = Field(default=None, description="Optional student identifier for tracking")
    
    # Optional sensitive attributes (used solely for downstream auditing, NEVER for scoring)
    sex: Optional[str] = Field(default=None, description="Demographic group F/M (audit only)")
    school: Optional[str] = Field(default=None, description="School GP/MS (audit only)")

    # 15 Numeric features
    age: int = Field(..., ge=15, le=22, description="Student age (15 to 22)")
    Medu: int = Field(..., ge=0, le=4, description="Mother's education (0=none to 4=higher)")
    Fedu: int = Field(..., ge=0, le=4, description="Father's education (0=none to 4=higher)")
    traveltime: int = Field(..., ge=1, le=4, description="Home to school travel time (1=<15min to 4=>1hour)")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time (1=<2hrs to 4=>10hrs)")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures (0 to 4)")
    famrel: int = Field(..., ge=1, le=5, description="Quality of family relationships (1=very bad to 5=excellent)")
    freetime: int = Field(..., ge=1, le=5, description="Free time after school (1=very low to 5=very high)")
    goout: int = Field(..., ge=1, le=5, description="Going out with friends (1=very low to 5=very high)")
    Dalc: int = Field(..., ge=1, le=5, description="Workday alcohol consumption (1=very low to 5=very high)")
    Walc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption (1=very low to 5=very high)")
    health: int = Field(..., ge=1, le=5, description="Current health status (1=very bad to 5=very good)")
    absences: int = Field(..., ge=0, le=93, description="Number of school absences (0 to 93)")
    G1: int = Field(..., ge=0, le=20, description="First period grade (0 to 20)")
    G2: int = Field(..., ge=0, le=20, description="Second period grade (0 to 20)")

    # 15 Categorical features
    address: str = Field(..., description="Home address type: 'U' (urban) or 'R' (rural)")
    famsize: str = Field(..., description="Family size: 'LE3' (<=3) or 'GT3' (>3)")
    Pstatus: str = Field(..., description="Parent's cohabitation status: 'T' (together) or 'A' (apart)")
    Mjob: str = Field(..., description="Mother's job: 'teacher', 'health', 'services', 'at_home', 'other'")
    Fjob: str = Field(..., description="Father's job: 'teacher', 'health', 'services', 'at_home', 'other'")
    reason: str = Field(..., description="Reason to choose school: 'home', 'reputation', 'course', 'other'")
    guardian: str = Field(..., description="Student guardian: 'mother', 'father', 'other'")
    schoolsup: str = Field(..., description="Extra educational support: 'yes' or 'no'")
    famsup: str = Field(..., description="Family educational support: 'yes' or 'no'")
    paid: str = Field(..., description="Extra paid classes within the course: 'yes' or 'no'")
    activities: str = Field(..., description="Extra-curricular activities: 'yes' or 'no'")
    nursery: str = Field(..., description="Attended nursery school: 'yes' or 'no'")
    higher: str = Field(..., description="Wants to take higher education: 'yes' or 'no'")
    internet: str = Field(..., description="Internet access at home: 'yes' or 'no'")
    romantic: str = Field(..., description="In a romantic relationship: 'yes' or 'no'")

    model_config = {
        "extra": "allow"  # Allows inspect for forbidden fields in model_validator
    }

    @model_validator(mode="before")
    @classmethod
    def check_leakage_fields(cls, values: Any) -> Any:
        """
        Rejects any payload that contains G3 or support_needed.
        Prevents label leakage into the inference vector.
        """
        if isinstance(values, dict):
            for forbidden in ["G3", "support_needed", "g3", "target"]:
                if forbidden in values:
                    raise ValueError(
                        f"Data leakage violation: '{forbidden}' is strictly prohibited from prediction inputs. "
                        "G3 is the future evaluation label and must never be provided for model scoring."
                    )
        return values

    @field_validator("address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
                     "guardian", "schoolsup", "famsup", "paid", "activities",
                     "nursery", "higher", "internet", "romantic")
    @classmethod
    def normalize_and_validate_categoricals(cls, v: str) -> str:
        return v.strip()


class FactorDetail(BaseModel):
    feature: str
    value: Any
    contribution: float
    description: str


class StudentPrediction(BaseModel):
    student_id: str
    rank: Optional[int] = None
    need_score: float = Field(..., ge=0.0, le=100.0, description="0-100 Support Need Score (higher = greater need)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated model probability of support need")
    selected: Optional[bool] = None
    top_factors: List[str] = Field(..., description="Top contributing factors pushing risk up")
    factor_details: Optional[List[FactorDetail]] = None
    disclaimer: str


class PredictBatchRequest(BaseModel):
    students: List[StudentInput]


class PredictResponse(BaseModel):
    predictions: List[StudentPrediction]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class OverviewResponse(BaseModel):
    total_students: int
    capacity_fraction: float
    num_selected: int
    average_need_score: float
    worst_group_recall: float
    fairness_gap: float
    overall_recall_at_capacity: float
    brier_score: float


class StudentListItem(BaseModel):
    student_id: str
    rank: int
    need_score: float
    probability: float
    selected: bool
    sex: str
    school: str
    ground_truth_support_needed: Optional[int] = None
    top_factors: List[str]


class StudentDetailResponse(BaseModel):
    student_id: str
    rank: int
    need_score: float
    probability: float
    selected: bool
    sex: str
    school: str
    ground_truth_support_needed: Optional[int] = None
    top_factors: List[str]
    factor_details: List[Dict[str, Any]]
    raw_features: Dict[str, Any]
    disclaimer: str


class GroupFairnessItem(BaseModel):
    attribute: str
    group: str
    n_rows: int
    n_positives: int
    n_selected_positives: int
    eligible: bool
    recall: float
    status_label: str


class FairnessResponse(BaseModel):
    groups: List[GroupFairnessItem]
    worst_group_recall: float
    fairness_gap: float
    metric_name: str
    pooling_rule: str


class ModelPerformanceResponse(BaseModel):
    overview: OverviewResponse
    robustness: Dict[str, Any]
    meta: Dict[str, Any]
