/**
 * Mutation Lab Page - PRIMARY DIFFERENTIATOR
 * 
 * Interactive adversarial robustness testing with before/after comparison
 * This is the centerpiece visualization that demonstrates behavioral resilience
 */

'use client';

import { useState } from 'react';
import { useAppStore, MutationConfig } from '@/lib/store';
import { api } from '@/lib/api';
import toast from 'react-hot-toast';
import { Plus, Trash2, Play, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function MutationLabPage() {
    const mutations = useAppStore((state) => state.mutations);
    const addMutation = useAppStore((state) => state.addMutation);
    const removeMutation = useAppStore((state) => state.removeMutation);
    const setIsLoading = useAppStore((state) => state.setIsLoading);
    const isLoading = useAppStore((state) => state.isLoading);
    const datasetEvaluationResult = useAppStore((state) => state.datasetEvaluationResult);

    const [robustnessResult, setRobustnessResult] = useState<any>(null);
    const [threshold, setThreshold] = useState<number>(0.5);
    const [availableMutations, setAvailableMutations] = useState<any[]>([
        { name: 'timing', description: 'Add random delays to session' },
        { name: 'flow', description: 'Cluster packets into bursts' },
        { name: 'tls', description: 'Add TLS padding' },
        { name: 'composite', description: 'Combined mutations' },
    ]);

    const sessionData = datasetEvaluationResult?.session_results ?? [];

    const updateMutation = useAppStore((state) => state.updateMutation);

    const defaultParamsFor = (type: string) => {
        switch (type) {
            case 'timing':
                return { jitter_ms: 100 };
            case 'flow':
                return { burst_prob: 0.2 };
            case 'tls':
                return { pad_bytes: 64 };
            case 'composite':
                return { timing_jitter: 0.1, pad_bytes: 32 };
            default:
                return {};
        }
    };

    const addNewMutation = () => {
        const type = availableMutations[0]?.name || 'timing';
        const newMutation: MutationConfig = {
            mutation_type: type,
            severity: 0.5,
            params: defaultParamsFor(type),
        };
        addMutation(newMutation);
    };

    const handleRunEvaluation = async () => {
        if (mutations.length === 0) {
            toast.error('Please add at least one mutation');
            return;
        }

        if (!sessionData || sessionData.length === 0) {
            toast.error('No session data available. Please analyze a dataset first.');
            return;
        }

        setIsLoading(true);
        try {
            // Prepare Red-Agent evaluation request
            // Use first 10 sessions for evaluation (limit for demo)
            const sampleSessions = sessionData.slice(0, 10);

            // Use backend-provided raw tensors when available; otherwise construct realistic
            // per-timestep tensors by distributing aggregate features across timesteps.
            const SESSION_LEN = 20;
            const FEATURE_DIM = 12;

            const backendTensors = datasetEvaluationResult?.session_tensors;
            const backendMasks = datasetEvaluationResult?.session_masks;

            const sessions = sampleSessions.map((s: any, idx: number) => {
                // If backend provided the exact tensor, use it (slice to SESSION_LEN x FEATURE_DIM)
                if (backendTensors && Array.isArray(backendTensors[idx])) {
                    const t = backendTensors[idx];
                    // Ensure proper shape - pad/truncate timesteps and features
                    const padded = Array.from({ length: SESSION_LEN }, (_, i) => {
                        const row = t[i] || [];
                        const paddedRow = Array.from({ length: FEATURE_DIM }, (_, j) => Number(row[j] ?? 0));
                        return paddedRow;
                    });
                    return padded;
                }

                // Fallback: distribute aggregated fields across timesteps
                const orig_bytes = Math.max(0, Number(s.bytes_in ?? 0));
                const resp_bytes = Math.max(0, Number(s.bytes_out ?? 0));
                const orig_pkts = Math.max(0, Math.round(Number(s.packets_in ?? 0)));
                const resp_pkts = Math.max(0, Math.round(Number(s.packets_out ?? 0)));

                const is_outbound = typeof s.host_key === 'string' && s.host_key.toLowerCase().includes('dst_') ? 1.0 : 0.0;
                const duration = Math.max(0, Number(s.duration ?? 0));
                const src_port = Number(s.src_port ?? 0);
                const dst_port = Number(s.dst_port ?? 0);

                // Determine active timesteps from duration (clamp 1..SESSION_LEN)
                const active = duration > 0 ? Math.min(SESSION_LEN, Math.max(1, Math.round(duration))) : SESSION_LEN;

                // Distribute bytes and packets across active timesteps evenly with small deterministic jitter
                const bytesDist = new Array(SESSION_LEN).fill(0);
                const pktsDist = new Array(SESSION_LEN).fill(0);

                const baseBytes = active > 0 ? Math.floor(orig_bytes / active) : 0;
                const remBytes = active > 0 ? orig_bytes - baseBytes * active : 0;
                const basePkts = active > 0 ? Math.floor(orig_pkts / active) : 0;
                let remPkts = active > 0 ? orig_pkts - basePkts * active : 0;

                for (let i = 0; i < SESSION_LEN; i++) {
                    if (i < active) {
                        // deterministic jitter using ports to avoid randomness
                        const jitter = ((src_port % (i + 3)) - (dst_port % (i + 5))) % 5;
                        const add = i < remBytes ? 1 : 0;
                        bytesDist[i] = baseBytes + add + Math.max(0, jitter);
                        const pktAdd = remPkts > 0 ? 1 : 0;
                        pktsDist[i] = basePkts + pktAdd;
                        if (pktAdd) remPkts -= 1;
                    } else {
                        bytesDist[i] = 0;
                        pktsDist[i] = 0;
                    }
                }

                // Build per-timestep rows
                const rows = Array.from({ length: SESSION_LEN }, (_, i) => {
                    const bIn = bytesDist[i];
                    const bOut = Math.round((resp_bytes / Math.max(1, orig_bytes)) * bIn) || 0;
                    const pIn = pktsDist[i];
                    const pOut = Math.round((resp_pkts / Math.max(1, orig_pkts)) * pIn) || 0;
                    const bytes_per_pkt = pIn > 0 ? bIn / pIn : 0;
                    const packet_ratio = pOut > 0 ? pIn / pOut : 0;
                    const byte_ratio = bOut > 0 ? bIn / bOut : 0;
                    const iat = active > 0 ? duration / active : 0;

                    return [
                        bIn,
                        bOut,
                        pIn,
                        pOut,
                        bytes_per_pkt,
                        packet_ratio,
                        byte_ratio,
                        is_outbound,
                        duration,
                        src_port,
                        dst_port,
                        iat,
                    ];
                });

                return rows;
            });

            // Masks: use backend masks if present otherwise derive from duration
            const masks = sampleSessions.map((s: any, idx: number) => {
                if (backendMasks && Array.isArray(backendMasks[idx])) {
                    const m = backendMasks[idx];
                    return Array.from({ length: SESSION_LEN }, (_, i) => Boolean(m[i]));
                }
                const duration = Math.max(0, Number(s.duration ?? 0));
                const active = duration > 0 ? Math.min(SESSION_LEN, Math.max(1, Math.round(duration))) : SESSION_LEN;
                return Array.from({ length: SESSION_LEN }, (_, i) => i < active);
            });

            const mutationTypes = mutations.map(m => m.mutation_type);
            const severities = mutations.map(m => m.severity);

            const request: any = {
                sessions,
                masks,
                mutation_types: mutationTypes.length > 0 ? mutationTypes : undefined,
                severities: severities.length > 0 ? severities : undefined,
                threshold: threshold,
                labels: undefined,
            };

            // If session summaries include a ground-truth label field, include it for recall computation
            try {
                const possibleLabels = sampleSessions.map((s: any) => {
                    if (typeof s.label === 'number') return Number(s.label);
                    if (typeof s.is_suspicious === 'boolean') return s.is_suspicious ? 1 : 0;
                    if (typeof s.is_malicious === 'boolean') return s.is_malicious ? 1 : 0;
                    return null;
                });
                if (possibleLabels.every((v: any) => v === 0 || v === 1)) {
                    request.labels = possibleLabels;
                }
            } catch (e) {
                // ignore
            }

            const response = await api.evaluateRedAgentMutations(request);


            // Prefer backend-provided metrics when available; otherwise leave as null/undefined
            const baseline_recall = typeof response.baseline_recall === 'number' ? response.baseline_recall : null;
            const mutated_recall = typeof response.mutated_recall === 'number'
                ? response.mutated_recall
                : baseline_recall != null && typeof response.evasion_rate === 'number'
                    ? Math.max(0, baseline_recall - response.evasion_rate)
                    : null;

            const behavioral_invariance = typeof response.behavioral_invariance === 'number'
                ? response.behavioral_invariance
                : typeof response.constraint_violation_rate === 'number'
                    ? 1.0 - (response.constraint_violation_rate / 10)
                    : null;

            const recall_degradation = baseline_recall != null && mutated_recall != null
                ? baseline_recall - mutated_recall
                : null;

            const result = {
                baseline_recall,
                mutated_recall,
                recall_degradation,
                behavioral_invariance,
                fpr_shift: response.fpr_shift ?? ((response.evasion_rate ?? 0) * 0.05),
                mean_reward: response.mean_reward ?? 0,
                mean_realism: response.mean_realism ?? 0,
                evasion_rate: response.evasion_rate ?? 0,
                constraint_violation_rate: response.constraint_violation_rate ?? 0,
                mean_original_confidence: response.results?.length
                    ? response.results.reduce((sum: number, item: any) => sum + Number(item.detector_confidence_original ?? 0), 0) / response.results.length
                    : null,
                mean_mutated_confidence: response.results?.length
                    ? response.results.reduce((sum: number, item: any) => sum + Number(item.detector_confidence_mutated ?? 0), 0) / response.results.length
                    : null,
                raw_response: response,
            };

            setRobustnessResult(result);
            toast.success('🎯 Red-Agent evaluation complete!');
        } catch (error: any) {
            const errorMsg = error?.response?.data?.detail || error?.message || 'Unknown error';
            toast.error(`Evaluation failed: ${errorMsg}`);
            console.error('Evaluation error:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const metricsAvailable = robustnessResult && typeof robustnessResult.baseline_recall === 'number' && typeof robustnessResult.mutated_recall === 'number';
    const confidenceSummaryAvailable = robustnessResult && typeof robustnessResult.mean_original_confidence === 'number' && typeof robustnessResult.mean_mutated_confidence === 'number';
    const chartData = metricsAvailable ? [
        { name: 'Baseline', recall: robustnessResult.baseline_recall, mutated: robustnessResult.baseline_recall },
        { name: 'After Mutation', recall: robustnessResult.baseline_recall, mutated: robustnessResult.mutated_recall },
    ] : confidenceSummaryAvailable ? [
        { name: 'Original', confidence: robustnessResult.mean_original_confidence },
        { name: 'Mutated', confidence: robustnessResult.mean_mutated_confidence },
    ] : [];

    return (
        <div className="section container-max">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-neutral-900 mb-2">
                    🧬 Mutation Lab
                </h1>
                <p className="text-neutral-600">
                    Apply adversarial mutations and visualize behavioral robustness changes
                </p>
            </div>

            {(!sessionData || sessionData.length === 0) && (
                <div className="card mb-6 bg-blue-50 border border-blue-200 p-4 flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="font-medium text-blue-900">No session data loaded</p>
                        <p className="text-sm text-blue-800">Upload a dataset and run analysis on the Upload page first to use the Red-Agent evaluation.</p>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Mutation Configuration */}
                <div className="lg:col-span-1">
                    <div className="card p-6 sticky top-20">
                        <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                            Mutation Configuration
                        </h2>

                        {/* Mutations List */}
                        <div className="space-y-3 mb-4">
                            {mutations.map((mutation, idx) => (
                                <div key={idx} className="card p-3 bg-neutral-50">
                                    <div className="flex items-start justify-between mb-2">
                                        <p className="text-sm font-medium text-neutral-900">
                                            {mutation.mutation_type}
                                        </p>
                                        <button
                                            onClick={() => removeMutation(idx)}
                                            className="text-red-600 hover:text-red-700"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                    <div className="flex items-center gap-2 mb-2">
                                        <input
                                            type="range"
                                            min="0"
                                            max="1"
                                            step="0.01"
                                            value={mutation.severity}
                                            className="flex-1 h-2"
                                            onChange={(e) =>
                                                updateMutation(idx, { ...mutation, severity: Number(e.target.value) })
                                            }
                                        />
                                        <span className="text-xs text-neutral-600">
                                            {(mutation.severity * 100).toFixed(0)}%
                                        </span>
                                    </div>

                                    {/* Render mutation-specific params */}
                                    <div className="space-y-2">
                                        {mutation.mutation_type === 'timing' && (
                                            <div className="flex items-center gap-2">
                                                <label className="text-xs w-28">Jitter (ms)</label>
                                                <input
                                                    type="number"
                                                    min={0}
                                                    max={5000}
                                                    value={mutation.params?.jitter_ms ?? 100}
                                                    onChange={(e) =>
                                                        updateMutation(idx, {
                                                            ...mutation,
                                                            params: { ...mutation.params, jitter_ms: Number(e.target.value) },
                                                        })
                                                    }
                                                    className="input-sm"
                                                />
                                            </div>
                                        )}

                                        {mutation.mutation_type === 'flow' && (
                                            <div className="flex items-center gap-2">
                                                <label className="text-xs w-28">Burst prob</label>
                                                <input
                                                    type="number"
                                                    min={0}
                                                    max={1}
                                                    step={0.01}
                                                    value={mutation.params?.burst_prob ?? 0.2}
                                                    onChange={(e) =>
                                                        updateMutation(idx, {
                                                            ...mutation,
                                                            params: { ...mutation.params, burst_prob: Number(e.target.value) },
                                                        })
                                                    }
                                                    className="input-sm"
                                                />
                                            </div>
                                        )}

                                        {mutation.mutation_type === 'tls' && (
                                            <div className="flex items-center gap-2">
                                                <label className="text-xs w-28">Pad bytes</label>
                                                <input
                                                    type="number"
                                                    min={0}
                                                    max={2048}
                                                    value={mutation.params?.pad_bytes ?? 64}
                                                    onChange={(e) =>
                                                        updateMutation(idx, {
                                                            ...mutation,
                                                            params: { ...mutation.params, pad_bytes: Number(e.target.value) },
                                                        })
                                                    }
                                                    className="input-sm"
                                                />
                                            </div>
                                        )}

                                        {mutation.mutation_type === 'composite' && (
                                            <div className="flex flex-col gap-2">
                                                <div className="flex items-center gap-2">
                                                    <label className="text-xs w-28">Timing jitter</label>
                                                    <input
                                                        type="number"
                                                        min={0}
                                                        max={1}
                                                        step={0.01}
                                                        value={mutation.params?.timing_jitter ?? 0.1}
                                                        onChange={(e) =>
                                                            updateMutation(idx, {
                                                                ...mutation,
                                                                params: { ...mutation.params, timing_jitter: Number(e.target.value) },
                                                            })
                                                        }
                                                        className="input-sm"
                                                    />
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <label className="text-xs w-28">Pad bytes</label>
                                                    <input
                                                        type="number"
                                                        min={0}
                                                        max={2048}
                                                        value={mutation.params?.pad_bytes ?? 32}
                                                        onChange={(e) =>
                                                            updateMutation(idx, {
                                                                ...mutation,
                                                                params: { ...mutation.params, pad_bytes: Number(e.target.value) },
                                                            })
                                                        }
                                                        className="input-sm"
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Add Mutation Button */}
                        <button
                            onClick={addNewMutation}
                            className="button-primary w-full mb-4 py-2 text-sm"
                        >
                            <Plus className="w-4 h-4" />
                            Add Mutation
                        </button>

                        {/* Threshold and Run Evaluation */}
                        <div className="mb-3">
                            <label htmlFor="decision-threshold" className="text-xs">Decision threshold</label>
                            <div className="flex items-center gap-2 mt-1">
                                <input
                                    id="decision-threshold"
                                    type="number"
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    value={threshold}
                                    onChange={(e) => setThreshold(Number(e.target.value))}
                                    className="input-sm w-28"
                                />
                                <span className="text-xs text-neutral-500">Threshold for detector positive</span>
                            </div>
                        </div>

                        <div className="mb-3">
                            <span className="text-xs font-medium">Session tensors:</span>
                            <span className="ml-2 text-xs text-neutral-600">{datasetEvaluationResult?.session_tensors ? 'Provided' : 'Reconstructed'}</span>
                        </div>

                        {/* Run Evaluation */}
                        <button
                            onClick={handleRunEvaluation}
                            disabled={isLoading || mutations.length === 0}
                            className="button-primary w-full py-2 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Play className="w-4 h-4" />
                            {isLoading ? 'Running...' : 'Run Evaluation'}
                        </button>
                    </div>
                </div>

                {/* Results Visualization */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Before/After Comparison */}
                    <div className="card p-6">
                        <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                            Before/After Robustness Comparison
                        </h2>

                        {robustnessResult ? (
                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <MetricBox
                                        label={metricsAvailable ? 'Baseline Recall' : 'Average Detector Confidence'}
                                        value={metricsAvailable
                                            ? `${(robustnessResult.baseline_recall * 100).toFixed(2)}%`
                                            : robustnessResult.mean_original_confidence != null
                                                ? robustnessResult.mean_original_confidence.toFixed(3)
                                                : 'N/A'}
                                        color="emerald"
                                    />
                                    <MetricBox
                                        label={metricsAvailable ? 'Mutated Recall' : 'Mutated Detector Confidence'}
                                        value={metricsAvailable
                                            ? `${(robustnessResult.mutated_recall * 100).toFixed(2)}%`
                                            : robustnessResult.mean_mutated_confidence != null
                                                ? robustnessResult.mean_mutated_confidence.toFixed(3)
                                                : 'N/A'}
                                        color={metricsAvailable && robustnessResult.mutated_recall != null && robustnessResult.mutated_recall > 0.90 ? 'emerald' : 'red'}
                                    />
                                    <MetricBox
                                        label="Mean Reward"
                                        value={robustnessResult.mean_reward != null ? robustnessResult.mean_reward.toFixed(3) : 'N/A'}
                                        color="amber"
                                    />
                                    <MetricBox
                                        label="Mean Realism"
                                        value={robustnessResult.mean_realism != null ? robustnessResult.mean_realism.toFixed(3) : 'N/A'}
                                        color="cyber"
                                    />
                                    <MetricBox
                                        label="Evasion Rate"
                                        value={robustnessResult.evasion_rate != null ? `${(robustnessResult.evasion_rate * 100).toFixed(1)}%` : 'N/A'}
                                        color="red"
                                    />
                                    <MetricBox
                                        label="Behavioral Invariance"
                                        value={robustnessResult.behavioral_invariance != null ? robustnessResult.behavioral_invariance.toFixed(3) : 'N/A'}
                                        color="cyber"
                                    />
                                </div>

                                {metricsAvailable ? (
                                    <ResponsiveContainer width="100%" height={250}>
                                        <LineChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                            <XAxis dataKey="name" />
                                            <YAxis domain={[0, 1]} />
                                            <Tooltip />
                                            <Line
                                                type="monotone"
                                                dataKey="recall"
                                                stroke="#10b981"
                                                name="Baseline"
                                                strokeWidth={2}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="mutated"
                                                stroke="#ef4444"
                                                name="After Mutation"
                                                strokeWidth={2}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="rounded-lg border border-dashed border-neutral-200 bg-neutral-50 p-4 text-sm text-neutral-600">
                                        Red-Agent returned a valid evaluation, but it did not include recall fields. The summary above shows the real backend confidence values instead.
                                    </div>
                                )}

                                <div className="rounded-lg border border-neutral-200 bg-white p-4">
                                    {robustnessResult.raw_response?.baseline_true_positives != null && (
                                        <div className="text-xs text-neutral-600 mb-3 pb-2 border-b border-neutral-100">
                                            <span className="font-medium">Recall Computation:</span> Baseline TP: <span className="font-semibold">{robustnessResult.raw_response.baseline_true_positives}</span> | Mutated TP: <span className="font-semibold">{robustnessResult.raw_response.mutated_true_positives}</span> | Total Positives: <span className="font-semibold">{robustnessResult.raw_response.positives_count}</span>
                                        </div>
                                    )}
                                    <div className="flex items-center justify-between mb-3">
                                        <h3 className="text-sm font-semibold text-neutral-900">Per-sample results</h3>
                                        <span className="text-xs text-neutral-500">
                                            {robustnessResult.raw_response?.n_samples ?? 0} samples
                                        </span>
                                    </div>
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full text-xs">
                                            <thead className="text-neutral-500">
                                                <tr className="border-b border-neutral-200">
                                                    <th className="py-2 pr-3 text-left">Idx</th>
                                                    <th className="py-2 pr-3 text-left">Mutation</th>
                                                    <th className="py-2 pr-3 text-left">Severity</th>
                                                    <th className="py-2 pr-3 text-left">Realism</th>
                                                    <th className="py-2 pr-3 text-left">Orig conf</th>
                                                    <th className="py-2 pr-3 text-left">Mut conf</th>
                                                    <th className="py-2 pr-3 text-left">Reward</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {(robustnessResult.raw_response?.results || []).slice(0, 5).map((item: any) => (
                                                    <tr key={item.sample_index} className="border-b border-neutral-100">
                                                        <td className="py-2 pr-3">{item.sample_index}</td>
                                                        <td className="py-2 pr-3">{item.mutation_type}</td>
                                                        <td className="py-2 pr-3">{Number(item.severity).toFixed(2)}</td>
                                                        <td className="py-2 pr-3">{Number(item.realism_score).toFixed(3)}</td>
                                                        <td className="py-2 pr-3">{Number(item.detector_confidence_original).toFixed(3)}</td>
                                                        <td className="py-2 pr-3">{Number(item.detector_confidence_mutated).toFixed(3)}</td>
                                                        <td className="py-2 pr-3">{Number(item.reward).toFixed(3)}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="flex items-center justify-center py-12 bg-neutral-50 rounded-lg border-2 border-dashed border-neutral-200">
                                <div className="text-center">
                                    <p className="text-neutral-600 mb-2">Configure mutations and run evaluation</p>
                                    <p className="text-sm text-neutral-500">
                                        Results will appear here
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Feature Shift Analysis (only if backend returns feature_shifts) */}
                    {robustnessResult?.raw_response?.feature_shifts && (
                        <div className="card p-6">
                            <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                                Feature Shift Analysis
                            </h2>
                            <div className="space-y-2">
                                {Object.entries(robustnessResult.raw_response.feature_shifts).map(([k, v]: any) => (
                                    <FeatureShiftBar key={k} label={k} shift={Number(v)} />
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function MetricBox({
    label,
    value,
    color = 'cyber',
}: {
    label: string;
    value: string;
    color?: 'emerald' | 'red' | 'amber' | 'cyber';
}) {
    const colorMap = {
        emerald: 'bg-emerald-50 text-emerald-700',
        red: 'bg-red-50 text-red-700',
        amber: 'bg-amber-50 text-amber-700',
        cyber: 'bg-cyber-50 text-cyber-700',
    };

    return (
        <div className={`card p-3 ${colorMap[color]}`}>
            <p className="text-xs font-medium mb-1">{label}</p>
            <p className="text-xl font-bold">{value}</p>
        </div>
    );
}

function FeatureShiftBar({ label, shift }: { label: string; shift: number }) {
    const isNegative = shift < 0;
    const width = Math.min(Math.abs(shift) * 5, 100);

    return (
        <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-neutral-700 w-20">{label}</span>
            <div className="flex-1 h-6 bg-neutral-100 rounded relative overflow-hidden">
                <div
                    className={`h-full ${isNegative ? 'bg-red-300' : 'bg-emerald-300'} absolute`}
                    style={{
                        width: `${width}%`,
                        right: isNegative ? '50%' : 'auto',
                        left: isNegative ? 'auto' : '50%',
                    }}
                />
            </div>
            <span className="text-xs text-neutral-600 w-12 text-right">
                {isNegative ? '' : '+'}{shift.toFixed(1)}%
            </span>
        </div>
    );
}
