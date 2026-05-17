/**
 * Upload & Evaluation Page
 * 
 * Dataset upload and evaluation execution
 */

'use client';

import { useState } from 'react';
import { useAppStore } from '@/lib/store';
import { api } from '@/lib/api';
import toast from 'react-hot-toast';
import { Upload, FileCheck, Play, Loader } from 'lucide-react';

export default function UploadPage() {
    const uploadedFiles = useAppStore((state) => state.uploadedFiles);
    const addUploadedFile = useAppStore((state) => state.addUploadedFile);
    const setBaselineFileId = useAppStore((state) => state.setBaselineFileId);
    const setDatasetEvaluationResult = useAppStore((state) => state.setDatasetEvaluationResult);
    const setSelectedPage = useAppStore((state) => state.setSelectedPage);
    const isAnalyzing = useAppStore((state) => state.isAnalyzing);
    const setIsAnalyzing = useAppStore((state) => state.setIsAnalyzing);
    const analysisProgress = useAppStore((state) => state.analysisProgress);
    const setAnalysisProgress = useAppStore((state) => state.setAnalysisProgress);

    const [uploading, setUploading] = useState(false);

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        setUploading(true);
        try {
            for (const file of files) {
                const formData = new FormData();
                formData.append('file', file);

                // Determine file type
                let fileType = 'json';
                if (file.name.endsWith('.npz')) fileType = 'npz';
                if (file.name.endsWith('.jsonl')) fileType = 'jsonl';
                if (file.name.endsWith('.csv')) fileType = 'csv';

                formData.append('file_type', fileType);

                let response;
                try {
                    response = await api.uploadFile(formData);
                } catch (uploadError) {
                    toast.error(`Upload failed: ${file.name}`);
                    continue;
                }

                addUploadedFile(response);
                setBaselineFileId(response.file_id);
                toast.success(`✅ Uploaded: ${file.name}`);

                // Auto-run analysis for NPZ files
                if (fileType === 'npz') {
                    await runAnalysis(response);
                }
            }
        } catch (error) {
            toast.error('Upload failed');
        } finally {
            setUploading(false);
        }
    };

    const runAnalysis = async (fileMetadata: any) => {
        setIsAnalyzing(true);
        setAnalysisProgress('Loading model and preparing data...');

        try {
            const analysisRequest: { npz_file_id: string; sample_limit?: number; batch_size: number } = {
                npz_file_id: fileMetadata.file_id,
                batch_size: 256,
            };

            if (typeof fileMetadata.sample_count === 'number' && fileMetadata.sample_count > 10000) {
                analysisRequest.sample_limit = 10000;
            }

            setAnalysisProgress('Running C2 detection model...');
            const analysisResult = await api.evaluateDataset(analysisRequest);

            setAnalysisProgress('Computing host-centric insights...');
            setDatasetEvaluationResult(analysisResult);

            setAnalysisProgress('');
            setSelectedPage('host-insights');
            toast.success('🎯 Live C2 detection complete!');
        } catch (error: any) {
            setAnalysisProgress('');
            const errorMsg = error?.response?.data?.detail || error?.message || 'Unknown error';
            toast.error(`Analysis failed: ${errorMsg}`);
            console.error('Analysis error:', error);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleAnalyzeClick = async () => {
        if (uploadedFiles.length === 0) {
            toast.error('No files uploaded');
            return;
        }

        const npzFile = uploadedFiles.find((f) => f.file_type === 'npz');
        if (!npzFile) {
            toast.error('No NPZ file found. Please upload an NPZ dataset.');
            return;
        }

        await runAnalysis(npzFile);
    };

    return (
        <div className="section container-max">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-neutral-900 mb-2">
                    📤 Upload & Evaluation
                </h1>
                <p className="text-neutral-600">
                    Upload datasets for behavioral analysis and robustness evaluation
                </p>
            </div>

            {/* Workflow Guide */}
            <div className="card bg-cyber-50 border border-cyber-200 mb-8">
                <div className="p-4">
                    <h3 className="font-semibold text-neutral-900 mb-2">📋 Workflow:</h3>
                    <ol className="text-sm text-neutral-700 space-y-1 list-decimal list-inside">
                        <li><strong>Upload</strong> an NPZ dataset file</li>
                        <li><strong>Automatic analysis</strong> runs immediately to detect C2 traffic</li>
                        <li><strong>Navigate</strong> to the Host Insights page to see detection results</li>
                        <li>Use the <strong>Analyze</strong> button to re-run analysis or analyze uploaded files</li>
                    </ol>
                </div>
            </div>

            {/* Upload Area */}
            <div className="card p-12 mb-8 border-2 border-dashed border-neutral-200 text-center hover:border-cyber-400 transition-colors cursor-pointer">
                <input
                    type="file"
                    multiple
                    accept=".npz,.jsonl,.json,.csv,.binetflow"
                    onChange={handleFileUpload}
                    disabled={uploading || isAnalyzing}
                    className="hidden"
                    id="file-input"
                />

                <label htmlFor="file-input" className="cursor-pointer">
                    <div className="flex flex-col items-center gap-3">
                        <div className="p-3 bg-cyber-100 rounded-lg">
                            <Upload className="w-6 h-6 text-cyber-600" />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-neutral-900">
                                {uploading ? '⏳ Uploading...' : isAnalyzing ? '🔄 Analyzing...' : 'Drag files here'}
                            </h3>
                            <p className="text-sm text-neutral-600">
                                or click to select files (NPZ, JSONL, CSV, Zeek logs)
                            </p>
                            {analysisProgress && (
                                <p className="text-sm text-cyber-600 mt-2 font-medium">{analysisProgress}</p>
                            )}
                        </div>
                    </div>
                </label>
            </div>

            {/* Uploaded Files */}
            {uploadedFiles.length > 0 && (
                <div className="card overflow-hidden">
                    <div className="p-4 border-b border-neutral-200 flex items-center justify-between">
                        <h3 className="font-semibold text-neutral-900">
                            Uploaded Files ({uploadedFiles.length})
                        </h3>
                        <button
                            onClick={handleAnalyzeClick}
                            disabled={isAnalyzing}
                            className="flex items-center gap-2 px-4 py-2 bg-cyber-600 text-white rounded-lg hover:bg-cyber-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            {isAnalyzing ? (
                                <>
                                    <Loader className="w-4 h-4 animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <Play className="w-4 h-4" />
                                    Analyze
                                </>
                            )}
                        </button>
                    </div>

                    <table className="w-full">
                        <thead className="bg-neutral-50 border-b border-neutral-200">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                    File
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                    Type
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                    Size
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                    Samples
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-neutral-700">
                                    Status
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {uploadedFiles.map((file) => (
                                <tr key={file.file_id} className="border-b border-neutral-200 hover:bg-neutral-50">
                                    <td className="px-6 py-4 text-sm font-medium text-neutral-900">
                                        {file.filename}
                                    </td>
                                    <td className="px-6 py-4 text-sm text-neutral-600">
                                        {file.file_type.toUpperCase()}
                                    </td>
                                    <td className="px-6 py-4 text-sm text-neutral-600">
                                        {(file.size_bytes / 1024 / 1024).toFixed(2)} MB
                                    </td>
                                    <td className="px-6 py-4 text-sm text-neutral-600">
                                        {file.sample_count ? `${file.sample_count.toLocaleString()}` : '-'}
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-1 text-emerald-600">
                                            <FileCheck className="w-4 h-4" />
                                            <span className="text-xs font-medium">Ready</span>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {uploadedFiles.length === 0 && !uploading && (
                <div className="text-center py-12">
                    <p className="text-neutral-600">📂 No files uploaded yet</p>
                </div>
            )}
        </div>
    );
}
