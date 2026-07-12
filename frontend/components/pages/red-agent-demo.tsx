"use client";

import { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { api } from '@/lib/api';

interface RedAgentReport {
    summary_path: string;
    training_log_path: string;
    detector_checkpoint: string | null;
    host_npz: string | null;
    policy_checkpoint: string | null;
    detector_threshold: number;
    samples: number;
    log_rows: number;
    epochs: number;
    mean_original_probability: number;
    mean_mutated_probability: number;
    mean_probability_delta: number;
    median_probability_delta: number;
    best_probability_drop: number;
    worst_probability_increase: number;
    probability_drop_rate: number;
    threshold_evasion_rate: number;
    threshold_evasions: number;
    mean_reward: number;
    mean_realism: number;
    realistic_rate: number;
    constraint_violation_rate: number;
    mutation_counts: Record<string, number>;
    mutation_summaries: Record<
        string,
        {
            count: number;
            mean_probability_delta: number;
            mean_reward: number;
            confidence_drop_rate: number;
        }
    >;
    interpretation: string;
    caveats: string[];
}

export default function RedAgentDemoPage() {
    const [loading, setLoading] = useState(false);
    const [markdown, setMarkdown] = useState<string | null>(null);
    const [report, setReport] = useState<RedAgentReport | null>(null);

    const fetchReport = async () => {
        setLoading(true);
        try {
            const resp = await api.getRedAgentDemoReport();
            if (resp && resp.status === 'success') {
                setReport(resp.report ?? null);
                setMarkdown(resp.markdown ?? null);
                toast.success('Red-Agent demo report loaded');
            } else {
                toast.error('No demo report available');
            }
        } catch (err: any) {
            const msg = err?.response?.data?.detail || err?.message || 'Failed to load demo report';
            toast.error(msg);
            console.error('Red-Agent demo fetch error', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchReport();
    }, []);

    const formatPct = (value: number) => `${(value * 100).toFixed(1)}%`;
    const formatNumber = (value: number | null | undefined, decimals = 4) =>
        value === null || value === undefined ? '-' : value.toFixed(decimals);

    return (
        <div className="section container-max">
            <div className="mb-8">
                <p className="text-sm uppercase tracking-[0.24em] text-cyber-700">Red-Agent demo</p>
                <h1 className="mt-3 text-4xl font-semibold tracking-tight text-slate-950">Host-aware policy summary</h1>
                <p className="mt-4 max-w-3xl text-base leading-8 text-slate-600">
                    Explore the existing Red-Agent smoke workflow through a clean demo presentation. This report is generated from a saved host-aware run and visualizes the policy's effect on the detector score, mutation realism, and robustness metrics.
                </p>
            </div>

            <div className="grid gap-4 xl:grid-cols-[1.45fr_1fr] mb-8">
                <div className="rounded-[1.75rem] border border-slate-200 bg-white p-6 shadow-[0_24px_80px_-40px_rgba(15,23,42,0.12)]">
                    <div className="flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <h2 className="text-xl font-semibold text-slate-950">Run snapshot</h2>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                Uses the cached smoke summary from <span className="font-medium text-slate-900">experiments/red_agent_host_aware_smoke</span> and the demo report API.
                            </p>
                        </div>
                        <button
                            onClick={fetchReport}
                            disabled={loading}
                            className="inline-flex h-11 items-center justify-center rounded-full bg-cyber-600 px-5 text-sm font-semibold text-white transition hover:bg-cyber-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            {loading ? 'Refreshing…' : 'Refresh report'}
                        </button>
                    </div>

                    <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                        <SummaryBadge label="Backend endpoint" value="/api/v1/robustness/red-agent/demo-report" />
                        <SummaryBadge label="Saved summary" value="host_aware_phase1_summary.json" />
                        <SummaryBadge label="Policy status" value="cached smoke run" accent />
                    </div>
                </div>

                <div className="rounded-[1.75rem] border border-slate-200 bg-slate-950 p-6 text-white shadow-[0_24px_80px_-40px_rgba(15,23,42,0.18)]">
                    <h2 className="text-xl font-semibold">Demo guidance</h2>
                    <p className="mt-4 text-sm leading-7 text-slate-300">
                        This page is intended for demonstration and exploratory validation. It does not execute live training; it surfaces the existing result artifacts in a premium, presentation-ready layout.
                    </p>
                    <ul className="mt-6 space-y-3 text-sm text-slate-300">
                        <li>• Leverages existing smoke output, no retraining required for the demo.</li>
                        <li>• Highlights key robustness metrics for the Red-Agent policy.</li>
                        <li>• Includes mutation-level detail and human-readable interpretation.</li>
                    </ul>
                </div>
            </div>

            {report ? (
                <>
                    <div className="grid gap-4 xl:grid-cols-3 mb-8">
                        <MetricCard label="Original C2 probability" value={formatNumber(report.mean_original_probability, 4)} />
                        <MetricCard label="Mutated C2 probability" value={formatNumber(report.mean_mutated_probability, 4)} />
                        <MetricCard label="Probability delta" value={formatNumber(report.mean_probability_delta, 4)} suffix="drop" />
                        <MetricCard label="Evasion rate" value={formatPct(report.threshold_evasion_rate)} />
                        <MetricCard label="Mean reward" value={formatNumber(report.mean_reward, 4)} />
                        <MetricCard label="Realism score" value={formatNumber(report.mean_realism, 4)} />
                    </div>

                    <div className="grid gap-4 lg:grid-cols-2 mb-8">
                        <InfoPanel title="Training & report details">
                            <DetailRow label="Detector checkpoint" value={report.detector_checkpoint ?? 'N/A'} />
                            <DetailRow label="Policy checkpoint" value={report.policy_checkpoint ?? 'N/A'} />
                            <DetailRow label="Training log" value={report.training_log_path} />
                            <DetailRow label="Host NPZ" value={report.host_npz ?? 'N/A'} />
                            <DetailRow label="Detector threshold" value={formatNumber(report.detector_threshold, 6)} />
                            <DetailRow label="Epochs" value={String(report.epochs)} />
                            <DetailRow label="Samples" value={String(report.samples)} />
                            <DetailRow label="Log rows" value={String(report.log_rows)} />
                        </InfoPanel>
                        <InfoPanel title="Robustness overview">
                            <DetailRow label="Realistic mutation rate" value={formatPct(report.realistic_rate)} />
                            <DetailRow label="Constraint violation rate" value={formatPct(report.constraint_violation_rate)} />
                            <DetailRow label="Best confidence drop" value={formatNumber(report.best_probability_drop, 4)} />
                            <DetailRow label="Worst confidence rise" value={formatNumber(report.worst_probability_increase, 4)} />
                            <DetailRow label="Median probability delta" value={formatNumber(report.median_probability_delta, 4)} />
                        </InfoPanel>
                    </div>

                    <div className="rounded-[1.75rem] border border-slate-200 bg-white p-6 shadow-[0_24px_80px_-40px_rgba(15,23,42,0.08)] mb-8">
                        <div className="flex flex-col gap-4">
                            <div>
                                <h2 className="text-xl font-semibold text-slate-950">Mutation breakdown</h2>
                                <p className="mt-2 text-sm leading-6 text-slate-600">Review the most common transformations the Red-Agent used to evade detection.</p>
                            </div>
                            <div className="overflow-hidden rounded-[1.5rem] border border-slate-100">
                                <table className="min-w-full text-left text-sm">
                                    <thead className="bg-slate-50 text-xs uppercase tracking-[0.2em] text-slate-500">
                                        <tr>
                                            <th className="px-4 py-4">Mutation</th>
                                            <th className="px-4 py-4">Count</th>
                                            <th className="px-4 py-4">Mean Δ</th>
                                            <th className="px-4 py-4">Mean reward</th>
                                            <th className="px-4 py-4">Drop rate</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-100 bg-white">
                                        {Object.entries(report.mutation_summaries).map(([mutation, summary]) => (
                                            <tr key={mutation} className="hover:bg-slate-50 transition-colors">
                                                <td className="px-4 py-4 font-medium text-slate-900">{mutation}</td>
                                                <td className="px-4 py-4 text-slate-700">{String(summary.count)}</td>
                                                <td className="px-4 py-4 text-slate-700">{formatNumber(summary.mean_probability_delta, 4)}</td>
                                                <td className="px-4 py-4 text-slate-700">{formatNumber(summary.mean_reward, 4)}</td>
                                                <td className="px-4 py-4 text-slate-700">{formatPct(summary.confidence_drop_rate)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    <div className="grid gap-4 lg:grid-cols-2 mb-8">
                        <InfoPanel title="Interpretation" subtle>
                            <p className="text-sm leading-7 text-slate-700">{report.interpretation}</p>
                        </InfoPanel>
                        <InfoPanel title="Caveats" subtle>
                            {report.caveats.length > 0 ? (
                                <ul className="list-disc space-y-2 pl-5 text-sm text-slate-700">
                                    {report.caveats.map((caveat, index) => (
                                        <li key={index}>{caveat}</li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="text-sm text-slate-700">No caveats were generated for this report.</p>
                            )}
                        </InfoPanel>
                    </div>

                    {markdown ? (
                        <div className="rounded-[1.75rem] border border-slate-200 bg-slate-50 p-6 shadow-[0_24px_80px_-40px_rgba(15,23,42,0.08)]">
                            <h2 className="text-xl font-semibold text-slate-950 mb-4">Raw report markdown</h2>
                            <pre className="max-h-[420px] overflow-y-auto rounded-[1.25rem] border border-slate-200 bg-white p-4 text-sm leading-6 text-slate-700">{markdown}</pre>
                        </div>
                    ) : null}
                </>
            ) : (
                <div className="rounded-[1.75rem] border border-slate-200 bg-white p-6 shadow-[0_24px_80px_-40px_rgba(15,23,42,0.08)] text-sm text-slate-700">
                    No real Red-Agent report is available yet. Generate <code className="rounded bg-slate-100 px-1 py-0.5 text-slate-700">experiments/red_agent_host_aware_smoke/host_aware_phase1_summary.json</code> using the host-aware smoke training pipeline, then refresh.
                </div>
            )}
        </div>
    );
}

function MetricCard({
    label,
    value,
    suffix,
}: {
    label: string;
    value: string;
    suffix?: string;
}) {
    return (
        <div className="rounded-[1.5rem] border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-sm uppercase tracking-[0.18em] text-slate-500 mb-3">{label}</p>
            <p className="text-3xl font-semibold text-slate-950">{value}{suffix ? ` ${suffix}` : ''}</p>
        </div>
    );
}

function InfoPanel({
    title,
    children,
    subtle,
}: {
    title: string;
    children: React.ReactNode;
    subtle?: boolean;
}) {
    return (
        <div className={`rounded-[1.5rem] border ${subtle ? 'border-slate-200 bg-slate-50' : 'border-slate-200 bg-white'} p-6 shadow-sm`}>
            <h2 className="text-xl font-semibold text-slate-950 mb-4">{title}</h2>
            <div className="space-y-3 text-sm text-slate-700">{children}</div>
        </div>
    );
}

// FIXED: Added minWidth: 0 to parent container, and inline truncation styles to the value text
function SummaryBadge({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
    return (
        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-4" style={{ minWidth: 0 }}>
            <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">{label}</p>
            <p 
                className={`mt-2 text-sm font-medium ${accent ? 'text-cyber-700' : 'text-slate-900'}`}
                title={value}
                style={{
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                }}
            >
                {value}
            </p>
        </div>
    );
}

// FIXED: Removed generic break-words and added strict inline word-break
function DetailRow({ label, value }: { label: string; value: string }) {
    return (
        <div className="grid gap-2 rounded-3xl bg-slate-50 px-4 py-3">
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
            <p 
                className="text-sm font-medium text-slate-900" 
                style={{ wordBreak: 'break-all' }}
            >
                {value}
            </p>
        </div>
    );
}