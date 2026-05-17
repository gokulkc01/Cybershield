/**
 * Robustness Analytics Page
 * 
 * Research-grade robustness metrics and analysis
 */

'use client';

import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function RobustnessAnalyticsPage() {
    const mutationImpactData = [
        { mutation: 'timing_jitter', recall_loss: 0.6 },
        { mutation: 'burst_callback', recall_loss: 1.2 },
        { mutation: 'tls_padding', recall_loss: 0.4 },
        { mutation: 'packet_variation', recall_loss: 0.9 },
        { mutation: 'byte_distribution', recall_loss: 0.3 },
    ];

    const robustnessMetrics = [
        { mutation: 'Baseline', baseline_recall: 96.36, fpr: 1.08 },
        { mutation: 'Timing', baseline_recall: 95.80, fpr: 1.12 },
        { mutation: 'TLS', baseline_recall: 96.10, fpr: 1.10 },
        { mutation: 'Flow', baseline_recall: 95.50, fpr: 1.15 },
    ];

    return (
        <div className="section container-max">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-neutral-900 mb-2">
                    🛡️ Robustness Analytics
                </h1>
                <p className="text-neutral-600">
                    Research-grade behavioral robustness evaluation and comparative analysis
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                {/* Mutation Impact Ranking */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                        Mutation Impact Ranking
                    </h2>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={mutationImpactData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="mutation" />
                            <YAxis label={{ value: 'Recall Loss (%)', angle: -90, position: 'insideLeft' }} />
                            <Tooltip />
                            <Bar dataKey="recall_loss" fill="#ef4444" radius={[8, 8, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Robustness Comparison */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                        Recall vs FPR by Mutation
                    </h2>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={robustnessMetrics}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="mutation" />
                            <YAxis yAxisId="left" />
                            <YAxis yAxisId="right" orientation="right" />
                            <Tooltip />
                            <Legend />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="baseline_recall"
                                stroke="#10b981"
                                name="Recall (%)"
                            />
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="fpr"
                                stroke="#f59e0b"
                                name="FPR (%)"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Summary Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <SummaryCard
                    title="Mean Behavioral Invariance"
                    value="0.975"
                    description="Average invariance across mutations"
                />
                <SummaryCard
                    title="Maximum Recall Degradation"
                    value="1.86%"
                    description="Worst-case mutation impact"
                />
                <SummaryCard
                    title="Fragile Samples"
                    value="124"
                    description="Samples detected in baseline but not mutated"
                />
            </div>
        </div>
    );
}

function SummaryCard({ title, value, description }: { title: string; value: string; description: string }) {
    return (
        <div className="card p-6">
            <p className="text-sm text-neutral-600 mb-2">{title}</p>
            <p className="text-3xl font-bold text-neutral-900 mb-2">{value}</p>
            <p className="text-xs text-neutral-500">{description}</p>
        </div>
    );
}
