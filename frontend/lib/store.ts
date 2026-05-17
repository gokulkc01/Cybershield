/**
 * Global State Store using Zustand
 * 
 * Manages:
 * - Current dataset state
 * - Mutation configuration
 * - Robustness evaluation results
 * - UI state
 */

import { create } from 'zustand';
import type { DatasetEvaluationResult } from '@/lib/api';

export type MutationConfig = {
    mutation_type: string;
    severity: number;
    params: Record<string, any>;
};

export type RobustnessResult = {
    baseline_recall: number;
    mutated_recall: number;
    recall_degradation: number;
    behavioral_invariance: number;
    fpr_shift: number;
    fragile_mutations: Record<string, number>;
    per_mutation_metrics: Record<string, any>;
};

export type AppState = {
    // Dataset state
    baselineFileId: string | null;
    mutatedFileId: string | null;
    setBaselineFileId: (id: string | null) => void;
    setMutatedFileId: (id: string | null) => void;

    // Uploaded files (persistent)
    uploadedFiles: any[];
    setUploadedFiles: (files: any[]) => void;
    addUploadedFile: (file: any) => void;

    // Mutation configuration
    mutations: MutationConfig[];
    addMutation: (mutation: MutationConfig) => void;
    removeMutation: (index: number) => void;
    updateMutation: (index: number, mutation: MutationConfig) => void;
    clearMutations: () => void;

    // Robustness results
    robustnessResult: RobustnessResult | null;
    setRobustnessResult: (result: RobustnessResult | null) => void;

    // Dataset evaluation results
    datasetEvaluationResult: DatasetEvaluationResult | null;
    setDatasetEvaluationResult: (result: DatasetEvaluationResult | null) => void;

    // Analysis progress
    isAnalyzing: boolean;
    setIsAnalyzing: (analyzing: boolean) => void;
    analysisProgress: string;
    setAnalysisProgress: (progress: string) => void;

    // UI state
    isLoading: boolean;
    setIsLoading: (loading: boolean) => void;
    error: string | null;
    setError: (error: string | null) => void;
    selectedPage: 'dashboard' | 'analysis' | 'mutation-lab' | 'robustness' | 'model-demo' | 'upload' | 'host-insights';
    setSelectedPage: (page: AppState['selectedPage']) => void;

    // Reset all state
    reset: () => void;
};

export const useAppStore = create<AppState>((set) => ({
    // Dataset state
    baselineFileId: null,
    mutatedFileId: null,
    setBaselineFileId: (id) => set({ baselineFileId: id }),
    setMutatedFileId: (id) => set({ mutatedFileId: id }),

    // Uploaded files
    uploadedFiles: [],
    setUploadedFiles: (files) => set({ uploadedFiles: files }),
    addUploadedFile: (file) =>
        set((state) => ({
            uploadedFiles: [...state.uploadedFiles, file],
        })),

    // Mutation configuration
    mutations: [],
    addMutation: (mutation) =>
        set((state) => ({
            mutations: [...state.mutations, mutation],
        })),
    removeMutation: (index) =>
        set((state) => ({
            mutations: state.mutations.filter((_, i) => i !== index),
        })),
    updateMutation: (index, mutation) =>
        set((state) => ({
            mutations: state.mutations.map((m, i) => (i === index ? mutation : m)),
        })),
    clearMutations: () => set({ mutations: [] }),

    // Robustness results
    robustnessResult: null,
    setRobustnessResult: (result) => set({ robustnessResult: result }),

    // Dataset evaluation results
    datasetEvaluationResult: null,
    setDatasetEvaluationResult: (result) => set({ datasetEvaluationResult: result }),

    // Analysis progress
    isAnalyzing: false,
    setIsAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
    analysisProgress: '',
    setAnalysisProgress: (progress) => set({ analysisProgress: progress }),

    // UI state
    isLoading: false,
    setIsLoading: (loading) => set({ isLoading: loading }),
    error: null,
    setError: (error) => set({ error }),
    selectedPage: 'dashboard',
    setSelectedPage: (page) => set({ selectedPage: page }),

    // Reset
    reset: () =>
        set({
            baselineFileId: null,
            mutatedFileId: null,
            uploadedFiles: [],
            mutations: [],
            robustnessResult: null,
            datasetEvaluationResult: null,
            isAnalyzing: false,
            analysisProgress: '',
            isLoading: false,
            error: null,
            selectedPage: 'dashboard',
        }),
}));
