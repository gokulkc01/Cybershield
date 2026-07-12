/**
 * Model Demo Page
 * 
 * Real-time C2 detection using domain-adaptive transformer
 * Upload NPZ files and get instant predictions
 */

'use client';

import { useState } from 'react';
import toast from 'react-hot-toast';
import { Upload, CheckCircle, Loader } from 'lucide-react';

interface Prediction {
    index: number;
    domain: string;
    probability: number;
    threshold?: number;
    label: string;
    label_int: number;
    ground_truth_label?: string;
    ground_truth_label_int?: number;
    correct?: boolean;
    host_id?: string;
    dest_id?: string;
}

type ModelType = 'session' | 'domain_adaptive' | 'host_aware';

const modelOptions: Record<ModelType, { label: string; input: string; demoFile: string; description: string }> = {
    session: {
        label: 'Session Model',
        input: 'Session NPZ',
        demoFile: 'D:\\CyberShield\\data\\processed\\uwf_adaptation_split\\test_sessions.npz',
        description: 'Plain Transformer over individual sessions.',
    },
    domain_adaptive: {
        label: 'Domain-Adaptive Model',
        input: 'Session NPZ',
        demoFile: 'D:\\CyberShield\\data\\processed\\uwf_adaptation_split\\test_sessions.npz',
        description: 'Current default session model with domain-specific thresholds.',
    },
    host_aware: {
        label: 'Host-Aware Model',
        input: 'Host-window NPZ',
        demoFile: 'D:\\CyberShield\\data\\processed\\host_aware_uwf_attack_mvp\\test_host_windows.npz',
        description: 'Uses current session plus prior sessions from the same host. Best recall demo path.',
    },
};

interface EvaluationMetrics {
    has_labels: boolean;
    total: number;
    correct: number;
    incorrect: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    specificity: number;
    false_positive_rate: number;
    false_negative_rate: number;
    ground_truth: {
        c2: number;
        benign: number;
    };
    predicted: {
        c2: number;
        benign: number;
    };
    confusion_matrix: {
        true_positive: number;
        true_negative: number;
        false_positive: number;
        false_negative: number;
    };
}

export default function ModelDemoPage() {
    const [apiUrl, setApiUrl] = useState('http://localhost:8000');
    const [modelType, setModelType] = useState<ModelType>('domain_adaptive');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [summary, setSummary] = useState<any>(null);
    const [evaluation, setEvaluation] = useState<EvaluationMetrics | null>(null);

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

            const response = await fetch(`${apiUrl}/api/v1/predict/predict-file?model_type=${modelType}`, {
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
                const metrics = data.evaluation?.has_labels ? data.evaluation as EvaluationMetrics : null;
                setPredictions(preds);
                setEvaluation(metrics);

                // Calculate summary
                const c2Count = preds.filter(p => p.label_int === 1).length;
                const benignCount = preds.length - c2Count;
                const avgProb = (preds.reduce((sum, p) => sum + p.probability, 0) / preds.length * 100).toFixed(1);

                setSummary({
                    modelType: data.model_type || modelType,
                    modelLabel: data.model_label || modelOptions[modelType].label,
                    inputFormat: data.input_format || modelOptions[modelType].input,
                    threshold: typeof data.threshold === 'number' ? data.threshold : undefined,
                    totalSessions: preds.length,
                    c2Count,
                    benignCount,
                    detectionRate: ((c2Count / preds.length) * 100).toFixed(1),
                    avgConfidence: avgProb,
                    labeled: Boolean(metrics),
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
        setEvaluation(null);
    };

    const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

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

                    <div className="mb-4">
                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                            Model
                        </label>
                        <div className="grid grid-cols-1 gap-2">
                            {(Object.keys(modelOptions) as ModelType[]).map((key) => (
                                <button
                                    key={key}
                                    type="button"
                                    onClick={() => {
                                        setModelType(key);
                                        setPredictions([]);
                                        setSummary(null);
                                        setEvaluation(null);
                                    }}
                                    className={`text-left p-3 border rounded-lg transition-colors ${modelType === key
                                            ? 'border-cyber-500 bg-blue-50'
                                            : 'border-neutral-200 hover:border-neutral-300'
                                        }`}
                                >
                                    <div className="flex items-center justify-between gap-3">
                                        <span className="font-semibold text-neutral-900">
                                            {modelOptions[key].label}
                                        </span>
                                        <span className="text-xs px-2 py-1 rounded bg-neutral-100 text-neutral-700">
                                            {modelOptions[key].input}
                                        </span>
                                    </div>
                                    <p className="text-xs text-neutral-600 mt-1">
                                        {modelOptions[key].description}
                                    </p>
                                </button>
                            ))}
                        </div>
                        <div className="mt-3 p-3 bg-neutral-50 border border-neutral-200 rounded-lg">
                            <p className="text-xs text-neutral-600 mb-1">Recommended demo file</p>
                            <p className="text-xs font-mono text-neutral-800 break-all">
                                {modelOptions[modelType].demoFile}
                            </p>
                        </div>
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
                            <div className="bg-neutral-50 p-4 rounded-lg border border-neutral-200">
                                <p className="text-xs text-neutral-600">Selected Model</p>
                                <p className="text-lg font-bold text-neutral-900">
                                    {summary.modelLabel}
                                </p>
                                <p className="text-xs text-neutral-600 mt-1">
                                    Input: {summary.inputFormat}
                                    {typeof summary.threshold === 'number' && ` | Threshold: ${summary.threshold.toFixed(4)}`}
                                </p>
                            </div>
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
                            {summary.labeled ? (
                                <div className="bg-emerald-50 p-4 rounded-lg border border-emerald-200">
                                    <p className="text-xs text-neutral-600 mb-1">Ground Truth Labels</p>
                                    <p className="text-sm font-semibold text-emerald-800">
                                        Labels found. Correctness metrics are shown below.
                                    </p>
                                </div>
                            ) : (
                                <div className="bg-neutral-50 p-4 rounded-lg border border-neutral-200">
                                    <p className="text-xs text-neutral-600 mb-1">Ground Truth Labels</p>
                                    <p className="text-sm font-semibold text-neutral-700">
                                        No labels found. This result shows predictions only.
                                    </p>
                                </div>
                            )}
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

            {/* Evaluation Metrics */}
            {evaluation && (
                <div className="card mb-8">
                    <h2 className="text-xl font-bold text-neutral-900 mb-4">
                        Correctness Evaluation
                    </h2>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div className="bg-emerald-50 p-4 rounded-lg">
                            <p className="text-xs text-neutral-600">Accuracy</p>
                            <p className="text-2xl font-bold text-emerald-700">
                                {formatPercent(evaluation.accuracy)}
                            </p>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg">
                            <p className="text-xs text-neutral-600">Precision</p>
                            <p className="text-2xl font-bold text-blue-700">
                                {formatPercent(evaluation.precision)}
                            </p>
                        </div>
                        <div className="bg-cyan-50 p-4 rounded-lg">
                            <p className="text-xs text-neutral-600">Recall</p>
                            <p className="text-2xl font-bold text-cyan-700">
                                {formatPercent(evaluation.recall)}
                            </p>
                        </div>
                        <div className="bg-violet-50 p-4 rounded-lg">
                            <p className="text-xs text-neutral-600">F1 Score</p>
                            <p className="text-2xl font-bold text-violet-700">
                                {formatPercent(evaluation.f1)}
                            </p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="border border-neutral-200 rounded-lg p-4">
                            <h3 className="font-semibold text-neutral-900 mb-3">Dataset Labels</h3>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-neutral-600">Ground-truth C2</span>
                                    <span className="font-semibold">{evaluation.ground_truth.c2}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-neutral-600">Ground-truth benign</span>
                                    <span className="font-semibold">{evaluation.ground_truth.benign}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-neutral-600">Correct predictions</span>
                                    <span className="font-semibold text-emerald-700">{evaluation.correct}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-neutral-600">Incorrect predictions</span>
                                    <span className="font-semibold text-red-700">{evaluation.incorrect}</span>
                                </div>
                            </div>
                        </div>

                        <div className="border border-neutral-200 rounded-lg p-4">
                            <h3 className="font-semibold text-neutral-900 mb-3">Predicted Classes</h3>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-neutral-600">Predicted C2</span>
                                    <span className="font-semibold">{evaluation.predicted.c2}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-neutral-600">Predicted benign</span>
                                    <span className="font-semibold">{evaluation.predicted.benign}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-neutral-600">False positive rate</span>
                                    <span className="font-semibold">{formatPercent(evaluation.false_positive_rate)}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-neutral-600">False negative rate</span>
                                    <span className="font-semibold">{formatPercent(evaluation.false_negative_rate)}</span>
                                </div>
                            </div>
                        </div>

                        <div className="border border-neutral-200 rounded-lg p-4">
                            <h3 className="font-semibold text-neutral-900 mb-3">Confusion Matrix</h3>
                            <div className="grid grid-cols-2 gap-2 text-center text-sm">
                                <div className="bg-emerald-50 rounded p-3">
                                    <p className="text-xs text-neutral-600">True Positive</p>
                                    <p className="text-xl font-bold text-emerald-700">
                                        {evaluation.confusion_matrix.true_positive}
                                    </p>
                                </div>
                                <div className="bg-red-50 rounded p-3">
                                    <p className="text-xs text-neutral-600">False Positive</p>
                                    <p className="text-xl font-bold text-red-700">
                                        {evaluation.confusion_matrix.false_positive}
                                    </p>
                                </div>
                                <div className="bg-red-50 rounded p-3">
                                    <p className="text-xs text-neutral-600">False Negative</p>
                                    <p className="text-xl font-bold text-red-700">
                                        {evaluation.confusion_matrix.false_negative}
                                    </p>
                                </div>
                                <div className="bg-emerald-50 rounded p-3">
                                    <p className="text-xs text-neutral-600">True Negative</p>
                                    <p className="text-xl font-bold text-emerald-700">
                                        {evaluation.confusion_matrix.true_negative}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

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
                                    {predictions.some(pred => pred.host_id) && (
                                        <th className="px-4 py-3 text-left font-semibold">Host</th>
                                    )}
                                    <th className="px-4 py-3 text-left font-semibold">Label</th>
                                    {evaluation && (
                                        <>
                                            <th className="px-4 py-3 text-left font-semibold">Ground Truth</th>
                                            <th className="px-4 py-3 text-left font-semibold">Correct?</th>
                                        </>
                                    )}
                                    <th className="px-4 py-3 text-left font-semibold">Confidence</th>
                                    {predictions.some(pred => typeof pred.threshold === 'number') && (
                                        <th className="px-4 py-3 text-left font-semibold">Threshold</th>
                                    )}
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
                                        {predictions.some(item => item.host_id) && (
                                            <td className="px-4 py-3 text-xs text-neutral-600">
                                                {pred.host_id || '-'}
                                            </td>
                                        )}
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-1 rounded text-xs font-bold ${pred.label_int === 1
                                                    ? 'bg-red-100 text-red-800'
                                                    : 'bg-green-100 text-green-800'
                                                }`}>
                                                {pred.label}
                                            </span>
                                        </td>
                                        {evaluation && (
                                            <>
                                                <td className="px-4 py-3">
                                                    <span className={`px-2 py-1 rounded text-xs font-bold ${pred.ground_truth_label_int === 1
                                                            ? 'bg-red-100 text-red-800'
                                                            : 'bg-green-100 text-green-800'
                                                        }`}>
                                                        {pred.ground_truth_label}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-3">
                                                    <span className={`px-2 py-1 rounded text-xs font-bold ${pred.correct
                                                            ? 'bg-emerald-100 text-emerald-800'
                                                            : 'bg-red-100 text-red-800'
                                                        }`}>
                                                        {pred.correct ? 'Yes' : 'No'}
                                                    </span>
                                                </td>
                                            </>
                                        )}
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
                                        {predictions.some(item => typeof item.threshold === 'number') && (
                                            <td className="px-4 py-3 text-xs text-neutral-600">
                                                {typeof pred.threshold === 'number' ? pred.threshold.toFixed(4) : '-'}
                                            </td>
                                        )}
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
