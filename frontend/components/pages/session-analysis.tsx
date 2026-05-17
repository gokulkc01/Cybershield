/**
 * Session Analysis Page
 * 
 * Per-session behavioral inspection and risk assessment
 */

'use client';

import { useState } from 'react';
import { Search, AlertCircle } from 'lucide-react';

export default function SessionAnalysisPage() {
    const [searchId, setSearchId] = useState('');
    const [selectedSession, setSelectedSession] = useState<any>(null);

    const sampleSessions = [
        {
            id: 'sess_001',
            duration: 45.2,
            bytes_in: 50000,
            bytes_out: 100000,
            packets_in: 150,
            packets_out: 200,
            risk_score: 0.92,
            confidence: 0.98,
            is_suspicious: true,
        },
        {
            id: 'sess_002',
            duration: 12.5,
            bytes_in: 5000,
            bytes_out: 8000,
            packets_in: 20,
            packets_out: 25,
            risk_score: 0.15,
            confidence: 0.95,
            is_suspicious: false,
        },
    ];

    return (
        <div className="section container-max">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-neutral-900 mb-4">Session Analysis</h1>
                <p className="text-neutral-600">
                    Inspect individual network sessions and analyze behavioral patterns
                </p>
            </div>

            {/* Search */}
            <div className="mb-6">
                <div className="relative">
                    <Search className="absolute left-3 top-3 w-5 h-5 text-neutral-400" />
                    <input
                        type="text"
                        placeholder="Search by session ID..."
                        value={searchId}
                        onChange={(e) => setSearchId(e.target.value)}
                        className="input-focus pl-10"
                    />
                </div>
            </div>

            {/* Sessions Table */}
            <div className="card overflow-hidden">
                <table className="w-full">
                    <thead className="bg-neutral-50 border-b border-neutral-200">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                Session ID
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                Duration
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                Risk Score
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                Status
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {sampleSessions.map((session) => (
                            <tr
                                key={session.id}
                                onClick={() => setSelectedSession(session)}
                                className="border-b border-neutral-200 hover:bg-neutral-50 cursor-pointer transition-colors"
                            >
                                <td className="px-6 py-4 text-sm font-mono text-cyber-600">
                                    {session.id}
                                </td>
                                <td className="px-6 py-4 text-sm text-neutral-900">
                                    {session.duration.toFixed(1)}s
                                </td>
                                <td className="px-6 py-4 text-sm font-semibold">
                                    <span className={`px-2 py-1 rounded text-xs font-medium ${session.risk_score > 0.5
                                            ? 'bg-red-100 text-red-800'
                                            : 'bg-emerald-100 text-emerald-800'
                                        }`}>
                                        {(session.risk_score * 100).toFixed(1)}%
                                    </span>
                                </td>
                                <td className="px-6 py-4">
                                    {session.is_suspicious && (
                                        <div className="flex items-center gap-1 text-red-600">
                                            <AlertCircle className="w-4 h-4" />
                                            <span className="text-xs font-medium">Suspicious</span>
                                        </div>
                                    )}
                                    {!session.is_suspicious && (
                                        <span className="text-xs text-emerald-600 font-medium">Benign</span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Session Details */}
            {selectedSession && (
                <div className="card p-6 mt-6">
                    <h2 className="text-lg font-semibold text-neutral-900 mb-4">
                        Session Details: {selectedSession.id}
                    </h2>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <DetailCard label="Duration" value={`${selectedSession.duration.toFixed(1)}s`} />
                        <DetailCard label="Bytes In" value={`${(selectedSession.bytes_in / 1000).toFixed(1)}KB`} />
                        <DetailCard label="Bytes Out" value={`${(selectedSession.bytes_out / 1000).toFixed(1)}KB`} />
                        <DetailCard label="Risk Score" value={`${(selectedSession.risk_score * 100).toFixed(1)}%`} />
                    </div>
                </div>
            )}
        </div>
    );
}

function DetailCard({ label, value }: { label: string; value: string }) {
    return (
        <div className="card p-4">
            <p className="text-xs text-neutral-600 mb-1">{label}</p>
            <p className="text-lg font-semibold text-neutral-900">{value}</p>
        </div>
    );
}
