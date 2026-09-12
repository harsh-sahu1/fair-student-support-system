/**
 * Typed API Client for Student Support AI Backend.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  version: string;
}

export interface OverviewResponse {
  total_students: number;
  capacity_fraction: number;
  num_selected: number;
  average_need_score: number;
  worst_group_recall: number;
  fairness_gap: number;
  overall_recall_at_capacity: number;
  brier_score: number;
}

export interface StudentListItem {
  student_id: string;
  rank: number;
  need_score: number;
  probability: number;
  selected: boolean;
  sex: string;
  school: string;
  ground_truth_support_needed?: number;
  top_factors: string[];
}

export interface FactorDetail {
  feature: string;
  value: any;
  contribution: number;
  description: string;
}

export interface StudentDetailResponse {
  student_id: string;
  rank: number;
  need_score: number;
  probability: number;
  selected: boolean;
  sex: string;
  school: string;
  ground_truth_support_needed?: number;
  top_factors: string[];
  factor_details: FactorDetail[];
  raw_features: Record<string, any>;
  disclaimer: string;
}

export interface GroupFairnessItem {
  attribute: string;
  group: string;
  n_rows: number;
  n_positives: number;
  n_selected_positives: number;
  eligible: boolean;
  recall: number;
  status_label: string;
}

export interface FairnessResponse {
  groups: GroupFairnessItem[];
  worst_group_recall: number;
  fairness_gap: number;
  metric_name: string;
  pooling_rule: string;
}

export interface RobustnessMetric {
  mean: number;
  std: number;
}

export interface ModelPerformanceResponse {
  overview: OverviewResponse;
  robustness: {
    seeds_evaluated: number;
    recall_at_capacity: RobustnessMetric;
    worst_group_recall: RobustnessMetric;
    fairness_gap: RobustnessMetric;
    brier_score: RobustnessMetric;
    interpretability_note: string;
  };
  meta: Record<string, any>;
}

export interface StudentPrediction {
  student_id: string;
  rank?: number;
  need_score: number;
  probability: number;
  selected?: boolean;
  top_factors: string[];
  factor_details?: FactorDetail[];
  disclaimer: string;
}

export interface PredictResponse {
  predictions: StudentPrediction[];
  count: number;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.message) {
        errorDetail = errJson.message;
      } else if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function fetchOverview(): Promise<OverviewResponse> {
  const res = await fetch(`${BASE_URL}/overview`);
  return handleResponse<OverviewResponse>(res);
}

export async function fetchStudents(params?: {
  selectedOnly?: boolean;
  group?: string;
  limit?: number;
}): Promise<StudentListItem[]> {
  const query = new URLSearchParams();
  if (params?.selectedOnly) query.append('selected_only', 'true');
  if (params?.group) query.append('group', params.group);
  if (params?.limit) query.append('limit', params.limit.toString());
  
  const res = await fetch(`${BASE_URL}/students?${query.toString()}`);
  return handleResponse<StudentListItem[]>(res);
}

export async function fetchStudentDetail(studentId: string): Promise<StudentDetailResponse> {
  const res = await fetch(`${BASE_URL}/students/${encodeURIComponent(studentId)}`);
  return handleResponse<StudentDetailResponse>(res);
}

export async function fetchFairness(): Promise<FairnessResponse> {
  const res = await fetch(`${BASE_URL}/fairness`);
  return handleResponse<FairnessResponse>(res);
}

export async function fetchModelPerformance(): Promise<ModelPerformanceResponse> {
  const res = await fetch(`${BASE_URL}/model-performance`);
  return handleResponse<ModelPerformanceResponse>(res);
}

export async function postPredict(payload: any): Promise<PredictResponse> {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<PredictResponse>(res);
}
