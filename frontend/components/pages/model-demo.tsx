/**
 * Model Demo Page
 * 
 * Real-time C2 detection using domain-adaptive transformer
 * Upload NPZ files and get instant predictions
 */

'use client';

import { useState } from 'react';
import toast from 'react-hot-toast';
import { Upload, CheckCircle, AlertCircle, Loader } from 'lucide-react';

interface Prediction {
    index: number;
    domain: string;
    probability: number;
    label: string;
    label_int: number;
}

export default function ModelDemoPage() {
    const [apiUrl, setApiUrl] = useState('http://localhost:8000');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [summary, setSummary] = useState<any>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].name.endsWith('.npz')) {
            setSelectedFile(files[0]);
        } else {
            toast.error('Please select a valid NPZ file');
        }
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            setSelectedFile(files[0]);
        }
    };

    const uploadFile = async () => {
        if (!selectedFile) {
            toast.error('No file selected');
            return;
        }

        if (!apiUrl.trim()) {
            toast.error('Please enter API endpoint URL');
            return;
        }

        setUploading(true);
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(`${apiUrl}/api/v1/predict/predict-file`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();

            if (data.predictions && data.predictions.length > 0) {
                const preds = data.predictions as Prediction[];
                setPredictions(preds);

                // Calculate summary
                const c2Count = preds.filter(p => p.label_int === 1).length;
                const benignCount = preds.length - c2Count;
                const avgProb = (preds.reduce((sum, p) => sum + p.probability, 0) / preds.length * 100).toFixed(1);

                setSummary({
                    totalSessions: preds.length,
                    c2Count,
                    benignCount,
                    detectionRate: ((c2Count / preds.length) * 100).toFixed(1),
                    avgConfidence: avgProb,
                });

                toast.success(`✅ Analysis complete! ${preds.length} sessions analyzed.`);
            } else {
                toast.error('No predictions returned');
            }
        } catch (error: any) {
            toast.error('Upload failed: ' + (error.message || 'Unknown error'));
            console.error('Error:', error);
        } finally {
            setUploading(false);
        }
    };

    const clearFile = () => {
        setSelectedFile(null);
        setPredictions([]);
        setSummary(null);
    };

    return (
        <div className="section container-max">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-neutral-900 mb-2">
                    🛡️ Model Demo - Real-time C2 Detection
                </h1>
                <p className="text-neutral-600">
                    Upload NPZ files to get instant predictions using the domain-adaptive transformer
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                {/* Upload Card */}
                <div className="card">
                    <h2 className="text-xl font-bold text-neutral-900 mb-4">📤 Upload NPZ File</h2>

                    <div className="mb-4">
                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                            API Endpoint
                        </label>
                        <input
                            type="text"
                            value={apiUrl}
                            onChange={(e) => setApiUrl(e.target.value)}
                            placeholder="http://localhost:8000"
                            className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyber-500"
                        />
                        <p className="text-xs text-neutral-500 mt-1">
                            Default: http://localhost:8000
                        </p>
                    </div>

                    <div
                        className="border-2 border-dashed border-neutral-300 rounded-lg p-8 text-center hover:border-cyber-400 transition-colors cursor-pointer"
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                        onClick={() => document.getElementById('file-input')?.click()}
                    >
                        <div className="text-4xl mb-3">📦</div>
                        <p className="font-semibold text-neutral-900 mb-1">
                            Drag and drop your NPZ file
                        </p>
                        <p className="text-sm text-neutral-600">
                            or click to browse
                        </p>
                        <input
                            id="file-input"
                            type="file"
                            accept=".npz"
                            onChange={handleFileSelect}
                            className="hidden"
                        />
                    </div>

                    {selectedFile && (
                        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                            <p className="text-sm">
                                <strong>Selected:</strong> {selectedFile.name}
                            </p>
                            <p className="text-sm text-neutral-600">
                                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                            <div className="flex gap-2 mt-3">
                                <button
                                    onClick={uploadFile}
                                    disabled={uploading}
                                    className="flex-1 px-4 py-2 bg-cyber-600 text-white rounded-lg hover:bg-cyber-700 disabled:bg-gray-400 transition-colors flex items-center justify-center gap-2"
                                >
                                    {uploading ? (
                                        <>
                                            <Loader className="w-4 h-4 animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <Upload className="w-4 h-4" />
                                            Analyze
                                        </>
                                    )}
                                </button>
                                <button
                                    onClick={clearFile}
                                    className="px-4 py-2 bg-neutral-200 text-neutral-800 rounded-lg hover:bg-neutral-300 transition-colors"
                                >
                                    Clear
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Results Card */}
                <div className="card">
                    <h2 className="text-xl font-bold text-neutral-900 mb-4">📊 Summary</h2>

                    {summary ? (
                        <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-blue-50 p-4 rounded-lg">
                                    <p className="text-xs text-neutral-600">Total Sessions</p>
                                    <p className="text-2xl font-bold text-cyber-600">
                                        {summary.totalSessions}
                                    </p>
                                </div>
                                <div className="bg-red-50 p-4 rounded-lg">
                                    <p className="text-xs text-neutral-600">C2 Detected</p>
                                    <p className="text-2xl font-bold text-red-600">
                                        {summary.c2Count}
                                    </p>
                                </div>
                                <div className="bg-green-50 p-4 rounded-lg">
                                    <p className="text-xs text-neutral-600">Benign</p>
                                    <p className="text-2xl font-bold text-green-600">
                                        {summary.benignCount}
                                    </p>
                                </div>
                                <div className="bg-purple-50 p-4 rounded-lg">
                                    <p className="text-xs text-neutral-600">Detection Rate</p>
                                    <p className="text-2xl font-bold text-purple-600">
                                        {summary.detectionRate}%
                                    </p>
                                </div>
                            </div>
                            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                                <p className="text-xs text-neutral-600 mb-1">Avg Confidence</p>
                                <p className="text-2xl font-bold text-yellow-600">
                                    {summary.avgConfidence}%
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-60 text-center">
                            <CheckCircle className="w-12 h-12 text-neutral-300 mb-4" />
                            <p className="text-neutral-500">
                                Upload an NPZ file to see results
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Predictions Table */}
            {predictions.length > 0 && (
                <div className="card">
                    <h2 className="text-xl font-bold text-neutral-900 mb-4">
                        📋 Predictions (showing first 100 of {predictions.length})
                    </h2>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead className="bg-neutral-100 border-b border-neutral-200">
                                <tr>
                                    <th className="px-4 py-3 text-left font-semibold">Index</th>
                                    <th className="px-4 py-3 text-left font-semibold">Domain</th>
                                    <th className="px-4 py-3 text-left font-semibold">Label</th>
                                    <th className="px-4 py-3 text-left font-semibold">Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {predictions.slice(0, 100).map((pred, idx) => (
                                    <tr key={idx} className="border-b border-neutral-200 hover:bg-neutral-50">
                                        <td className="px-4 py-3 text-neutral-600">
                                            {pred.index}
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-1 rounded text-xs font-medium ${pred.domain === 'uwf'
                                                    ? 'bg-orange-100 text-orange-800'
                                                    : 'bg-blue-100 text-blue-800'
                                                }`}>
                                                {pred.domain}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-1 rounded text-xs font-bold ${pred.label_int === 1
                                                    ? 'bg-red-100 text-red-800'
                                                    : 'bg-green-100 text-green-800'
                                                }`}>
                                                {pred.label}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                <div className="flex-1 bg-neutral-200 rounded-full h-2">
                                                    <div
                                                        className="bg-cyber-600 h-2 rounded-full"
                                                        style={{
                                                            width: `${Math.round(pred.probability * 100)}%`,
                                                        }}
                                                    />
                                                </div>
                                                <span className="text-xs font-medium w-12">
                                                    {Math.round(pred.probability * 100)}%
                                                </span>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
