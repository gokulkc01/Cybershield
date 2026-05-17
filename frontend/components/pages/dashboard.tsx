/**
 * Dashboard Page - Homepage for CyberShield v2
 * 
 * Displays high-level metrics, recent analyses, and quick actions
 */

'use client';

import { useState, useEffect } from 'react';
import { useAppStore } from '@/lib/store';
import { api } from '@/lib/api';
import toast from 'react-hot-toast';
import { Activity, TrendingUp, Shield, Zap } from 'lucide-react';
import {
    LineChart,
    Line,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';

export default function DashboardPage() {
    const setSelectedPage = useAppStore((state) => state.setSelectedPage);
    const [metrics, setMetrics] = useState({
        totalSessions: 374187,
        suspiciousSessions: 3608,
        detectionRate: 0.9636,
        avgRiskScore: 0.247,
    });

    const [chartData, setChartData] = useState([
        { name: 'Safe', value: 370579 },
        { name: 'Low Risk', value: 2100 },
        { name: 'Medium Risk', value: 1200 },
        { name: 'High Risk', value: 308 },
    ]);

    const [robustnessData, setRobustnessData] = useState([
        { mutation: 'Baseline', recall: 0.9636 },
        { mutation: 'Timing Jitter', recall: 0.9642 },
        { mutation: 'TLS Padding', recall: 0.9598 },
        { mutation: 'Packet Variation', recall: 0.9544 },
    ]);

    useEffect(() => {
        // Initial health check
        const checkHealth = async () => {
            try {
                const health = await api.getHealth();
                toast.success('Backend connected');
            } catch (err) {
                toast.error('Backend connection failed');
            }
        };

        checkHealth();
    }, []);

    return (
        <div className="section container-max">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-neutral-900 mb-2">
                    Behavioral Adversarial Robustness Intelligence
                </h1>
                <p className="text-neutral-600">
                    Analyze behavioral C2 patterns and evaluate robustness under adversarial mutation
                </p>
            </div>

            {/* Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <MetricCard
                    title="Total Sessions Analyzed"
                    value="374,187"
                    icon={<Activity className="w-5 h-5" />}
                    color="cyber"
                />
                <MetricCard
                    title="C2 Behaviors Detected"
                    value="3,608"
                    percentage="0.96%"
                    icon={<Shield className="w-5 h-5" />}
                    color="risk"
                />
                <MetricCard
                    title="Baseline Detection Rate"
                    value="96.36%"
                    change="+0.5%"
                    icon={<TrendingUp className="w-5 h-5" />}
                    color="success"
                />
                <MetricCard
                    title="Behavioral Invariance"
                    value="0.975"
                    change="+0.02"
                    icon={<Zap className="w-5 h-5" />}
                    color="info"
                />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                {/* Risk Distribution */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                        Risk Distribution
                    </h2>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="value" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Robustness Trend */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                        Mutation Robustness
                    </h2>
                    <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={robustnessData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="mutation" />
                            <YAxis domain={[0.95, 0.97]} />
                            <Tooltip />
                            <Line
                                type="monotone"
                                dataKey="recall"
                                stroke="#0ea5e9"
                                strokeWidth={2}
                                dot={{ fill: '#0ea5e9', r: 4 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="card p-6 hover:shadow-md transition-shadow cursor-pointer" onClick={() => setSelectedPage('upload')}>
                    <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                        📤 Upload Dataset
                    </h3>
                    <p className="text-neutral-600 text-sm mb-4">
                        Upload NPZ or Zeek log files for behavioral analysis and robustness testing
                    </p>
                    <button className="button-primary text-sm py-2">
                        Upload Files
                    </button>
                </div>

                <div className="card p-6 hover:shadow-md transition-shadow cursor-pointer" onClick={() => setSelectedPage('mutation-lab')}>
                    <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                        🧬 Mutation Lab
                    </h3>
                    <p className="text-neutral-600 text-sm mb-4">
                        Apply adversarial mutations and visualize behavioral robustness changes
                    </p>
                    <button className="button-primary text-sm py-2">
                        Open Lab
                    </button>
                </div>
            </div>
        </div>
    );
}

function MetricCard({
    title,
    value,
    percentage,
    change,
    icon,
    color = 'cyber',
}: {
    title: string;
    value: string;
    percentage?: string;
    change?: string;
    icon: React.ReactNode;
    color?: 'cyber' | 'risk' | 'success' | 'info';
}) {
    const colorMap = {
        cyber: 'bg-cyber-100 text-cyber-600',
        risk: 'bg-red-100 text-red-600',
        success: 'bg-emerald-100 text-emerald-600',
        info: 'bg-blue-100 text-blue-600',
    };

    return (
        <div className="card p-6">
            <div className="flex items-start justify-between mb-4">
                <div>
                    <p className="text-sm text-neutral-600 mb-1">{title}</p>
                    <p className="text-2xl font-bold text-neutral-900">{value}</p>
                    {percentage && <p className="text-xs text-neutral-500 mt-1">{percentage}</p>}
                    {change && (
                        <p className={`text-xs mt-1 ${change.startsWith('+') ? 'text-emerald-600' : 'text-red-600'}`}>
                            {change}
                        </p>
                    )}
                </div>
                <div className={`p-2 rounded-lg ${colorMap[color]}`}>{icon}</div>
            </div>
        </div>
    );
}
