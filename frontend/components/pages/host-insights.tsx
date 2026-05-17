/**
 * Host-Centric Insights Page
 *
 * Live dataset analysis results and host/cohort summaries.
 */

'use client';

import { useMemo } from 'react';
import { useAppStore } from '@/lib/store';
import { api } from '@/lib/api';
import toast from 'react-hot-toast';
import { ArrowRight, Database, ShieldAlert, Activity, Server } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function HostInsightsPage() {
    const datasetEvaluationResult = useAppStore((state) => state.datasetEvaluationResult);
    const setSelectedPage = useAppStore((state) => state.setSelectedPage);

    const hostChartData = useMemo(() => {
        if (!datasetEvaluationResult) return [];
        return datasetEvaluationResult.host_summary.top_hosts.slice(0, 8).map((host) => ({
            host: host.host_key,
            suspicious_rate: Number((host.suspicious_rate * 100).toFixed(2)),
            avg_risk_score: Number((host.avg_risk_score * 100).toFixed(2)),
            session_count: host.session_count,
        }));
    }, [datasetEvaluationResult]);

    return (
        <div className="section container-max">
            <div className="mb-8 flex items-center justify-between gap-4 flex-wrap">
                <div>
                    <h1 className="text-3xl font-bold text-neutral-900 mb-2">
                        Behavioral Cohort Insights
                    </h1>
                    <p className="text-neutral-600">
                        Live C2 detection results grouped by flow-behavior cohorts
                    </p>
                </div>
                <button
                    onClick={() => setSelectedPage('upload')}
                    className="button-primary text-sm py-2"
                >
                    Upload another dataset
                    <ArrowRight className="w-4 h-4" />
                </button>
            </div>

            {!datasetEvaluationResult ? (
                <div className="card p-10 text-center border-2 border-dashed border-neutral-200">
                    <Database className="w-10 h-10 text-cyber-500 mx-auto mb-3" />
                    <h2 className="text-xl font-semibold text-neutral-900 mb-2">No dataset analyzed yet</h2>
                    <p className="text-neutral-600 mb-4">
                        Upload a .npz session dataset to run live C2 detection and generate host-centric insights.
                    </p>
                    <button onClick={() => setSelectedPage('upload')} className="button-primary">
                        Go to Upload
                    </button>
                </div>
            ) : (
                <>
                    <div className="card mb-8 border border-cyber-200 bg-cyber-50">
                        <div className="p-4 text-sm text-neutral-700">
                            This NPZ dataset does not include IP addresses, so the labels below are derived
                            from behavioral flow features such as port ranges and protocol. They represent
                            cohorts, not literal hosts.
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-8">
                        <MetricCard title="Sessions Analyzed" value={datasetEvaluationResult.total_sessions.toLocaleString()} icon={<Activity className="w-5 h-5" />} />
                        <MetricCard title="Suspicious Sessions" value={datasetEvaluationResult.suspicious_count.toLocaleString()} icon={<ShieldAlert className="w-5 h-5" />} />
                        <MetricCard title="Detection Rate" value={`${(datasetEvaluationResult.detection_rate * 100).toFixed(2)}%`} icon={<Server className="w-5 h-5" />} />
                        <MetricCard title="Cohorts" value={datasetEvaluationResult.host_summary.total_hosts.toLocaleString()} icon={<Database className="w-5 h-5" />} />
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                        <div className="card p-6">
                            <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                                Risk distribution
                            </h2>
                            <ResponsiveContainer width="100%" height={280}>
                                <BarChart data={datasetEvaluationResult.risk_distribution.counts.map((count, index) => ({
                                    bucket: `${datasetEvaluationResult.risk_distribution.bins[index].toFixed(1)}-${datasetEvaluationResult.risk_distribution.bins[index + 1].toFixed(1)}`,
                                    count,
                                }))}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                    <XAxis dataKey="bucket" />
                                    <YAxis />
                                    <Tooltip />
                                    <Bar dataKey="count" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="card p-6">
                            <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                                Cohort risk overview
                            </h2>
                            <ResponsiveContainer width="100%" height={280}>
                                <BarChart data={hostChartData} layout="vertical" margin={{ left: 16 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                    <XAxis type="number" />
                                    <YAxis type="category" dataKey="host" width={180} />
                                    <Tooltip />
                                    <Bar dataKey="suspicious_rate" fill="#ef4444" name="Suspicious %" radius={[0, 8, 8, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                        <div className="card overflow-hidden">
                            <div className="px-6 py-4 border-b border-neutral-200">
                                <h2 className="text-lg font-semibold text-neutral-900">Top cohort summaries</h2>
                            </div>
                            <table className="w-full">
                                <thead className="bg-neutral-50 border-b border-neutral-200">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">Cohort</th>
                                        <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">Sessions</th>
                                        <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">Suspicious</th>
                                        <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">Avg Risk</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {datasetEvaluationResult.host_summary.top_hosts.map((host) => (
                                        <tr key={host.host_key} className="border-b border-neutral-100 hover:bg-neutral-50">
                                            <td className="px-6 py-4 text-sm font-mono text-cyber-700">{host.host_key}</td>
                                            <td className="px-6 py-4 text-sm text-neutral-700">{host.session_count}</td>
                                            <td className="px-6 py-4 text-sm text-neutral-700">{host.suspicious_count}</td>
                                            <td className="px-6 py-4 text-sm text-neutral-700">{(host.avg_risk_score * 100).toFixed(2)}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        <div className="card overflow-hidden">
                            <div className="px-6 py-4 border-b border-neutral-200">
                                <h2 className="text-lg font-semibold text-neutral-900">Highest-risk sessions</h2>
                            </div>
                            <div className="max-h-[420px] overflow-auto">
                                <table className="w-full">
                                    <thead className="bg-neutral-50 border-b border-neutral-200 sticky top-0">
                                        <tr>
                                            <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">Session</th>
                                            <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">Cohort key</th>
                                            <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">Risk</th>
                                            <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {datasetEvaluationResult.session_results
                                            .slice()
                                            .sort((a, b) => b.risk_score - a.risk_score)
                                            .slice(0, 12)
                                            .map((session) => (
                                                <tr key={session.session_id} className="border-b border-neutral-100 hover:bg-neutral-50">
                                                    <td className="px-6 py-4 text-sm font-mono text-neutral-800">{session.session_id}</td>
                                                    <td className="px-6 py-4 text-sm font-mono text-cyber-700">{session.host_key}</td>
                                                    <td className="px-6 py-4 text-sm text-neutral-700">{(session.risk_score * 100).toFixed(2)}%</td>
                                                    <td className="px-6 py-4 text-sm">
                                                        <span className={`px-2 py-1 rounded text-xs font-medium ${session.is_suspicious ? 'bg-red-100 text-red-800' : 'bg-emerald-100 text-emerald-800'}`}>
                                                            {session.is_suspicious ? 'Suspicious' : 'Benign'}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

function MetricCard({ title, value, icon }: { title: string; value: string; icon: React.ReactNode }) {
    return (
        <div className="card p-6">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <p className="text-sm text-neutral-600 mb-1">{title}</p>
                    <p className="text-2xl font-bold text-neutral-900">{value}</p>
                </div>
                <div className="p-2 rounded-lg bg-cyber-100 text-cyber-700">{icon}</div>
            </div>
        </div>
    );
}