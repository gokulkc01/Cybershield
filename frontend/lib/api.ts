/**
 * API Client for CyberShield v2 Backend
 * 
 * Provides typed HTTP client with automatic error handling,
 * request/response logging, and type safety.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

/**
 * API Client class for CyberShield backend
 */
export class CyberShieldAPI {
    private client: AxiosInstance;

    constructor(baseURL: string = `${API_BASE_URL}/api/v1`) {
        this.client = axios.create({
            baseURL,
            timeout: 60000, // 60 second timeout
        });

        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            response => response,
            error => {
                console.error('API Error:', error);
                return Promise.reject(error);
            }
        );
    }

    /**
     * DETECTION ENDPOINTS
     */

    async analyzeSession(sessionData: any) {
        const response = await this.client.post('/detection/session', sessionData);
        return response.data;
    }

    async analyzeBatch(batch: any) {
        const response = await this.client.post('/detection/batch', batch);
        return response.data;
    }

    async evaluateDataset(request: any) {
        const response = await this.client.post('/detection/dataset', request);
        return response.data;
    }

    /**
     * MUTATION ENDPOINTS
     */

    async listMutations() {
        const response = await this.client.get('/mutation/available');
        return response.data;
    }

    async applyMutations(request: any) {
        const response = await this.client.post('/mutation/apply', request);
        return response.data;
    }

    async evaluateRobustness(request: any) {
        const response = await this.client.post('/mutation/evaluate', request);
        return response.data;
    }

    /**
     * ROBUSTNESS ENDPOINTS
     */

    async getRobustnessReport(evaluationId: string) {
        const response = await this.client.get(`/robustness/report/${evaluationId}`);
        return response.data;
    }

    async getRobustnessMetrics(evaluationId: string) {
        const response = await this.client.get(`/robustness/metrics/${evaluationId}`);
        return response.data;
    }

    async analyzeFailures(request: any) {
        const response = await this.client.post('/robustness/failures', request);
        return response.data;
    }

    async compareRobustness(request: any) {
        const response = await this.client.post('/robustness/compare', request);
        return response.data;
    }

    async evaluateRedAgentMutations(request: any) {
        const response = await this.client.post('/robustness/red-agent/evaluate', request);
        return response.data;
    }

    async getRedAgentDemoReport() {
        const response = await this.client.get('/robustness/red-agent/demo-report');
        return response.data;
    }

    /**
     * ARTIFACTS ENDPOINTS
     */

    async uploadFile(formData: FormData) {
        const response = await this.client.post('/artifacts/upload', formData);
        return response.data;
    }

    async downloadFile(fileId: string) {
        const response = await this.client.get(`/artifacts/file/${fileId}`, {
            responseType: 'blob',
        });
        return response.data;
    }

    async getFileMetadata(fileId: string) {
        const response = await this.client.get(`/artifacts/metadata/${fileId}`);
        return response.data;
    }

    async listArtifacts(params?: { file_type?: string; limit?: number; offset?: number }) {
        const response = await this.client.get('/artifacts/list', { params });
        return response.data;
    }

    async deleteArtifact(fileId: string) {
        const response = await this.client.delete(`/artifacts/file/${fileId}`);
        return response.data;
    }

    async saveReport(request: any) {
        const response = await this.client.post('/artifacts/report', request);
        return response.data;
    }

    /**
     * HEALTH CHECK
     */

    async getHealth() {
        const response = await axios.get(`${API_BASE_URL}/health`);
        return response.data;
    }
}

// Export singleton instance
export const api = new CyberShieldAPI();

// Export types for API responses
export type DetectionResult = {
    session_id: string;
    risk_score: number;
    is_suspicious: boolean;
    confidence: number;
    behavioral_features: Record<string, number>;
    timestamp: number;
};

export type MutationResult = {
    mutation_id: string;
    source_file_id: string;
    mutations_applied: any[];
    samples_mutated: number;
    output_npz_file_id: string;
    metadata_file_id: string;
    created_at: string;
};

export type RobustnessMetrics = {
    baseline_recall: number;
    mutated_recall: number;
    recall_degradation: number;
    behavioral_invariance: number;
    fpr_shift: number;
};

export type SessionDetectionSummary = {
    session_id: string;
    host_key: string;
    risk_score: number;
    is_suspicious: boolean;
    confidence: number;
    duration: number;
    bytes_in: number;
    bytes_out: number;
    src_port: number;
    dst_port: number;
};

export type HostBehaviorSummary = {
    host_key: string;
    session_count: number;
    suspicious_count: number;
    suspicious_rate: number;
    avg_risk_score: number;
    max_risk_score: number;
    avg_duration: number;
    dominant_src_port: number;
    dominant_dst_port: number;
    dominant_protocol: string;
};

export type HostCentricSummary = {
    total_hosts: number;
    suspicious_hosts: number;
    avg_sessions_per_host: number;
    top_hosts: HostBehaviorSummary[];
    high_risk_hosts: HostBehaviorSummary[];
};

export type DatasetEvaluationResult = {
    evaluation_id: string;
    total_sessions: number;
    suspicious_count: number;
    detection_rate: number;
    avg_risk_score: number;
    std_risk_score: number;
    risk_distribution: {
        bins: number[];
        counts: number[];
    };
    session_results: SessionDetectionSummary[];
    session_tensors?: number[][][];
    session_masks?: boolean[][];
    host_summary: HostCentricSummary;
    created_at: string;
};
