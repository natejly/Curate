"use client";
import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

interface ModelInfo {
  session_id: string;
  filename: string;
  s3_key: string;
  size: number;
  last_modified: string;
  url: string;
}

interface TrainingStats {
  final_accuracy: number | null;
  final_loss: number | null;
  final_val_accuracy: number | null;
  final_val_loss: number | null;
  total_epochs: number;
  best_epoch: number | null;
  training_time: string | null;
  dataset_name: string | null;
  img_size: any;
  num_classes: number | null;
  base_model_name: string | null;
  model_info: any;
}

interface ModelInfoWithStats extends ModelInfo {
  stats?: TrainingStats;
}

interface ModelsResponse {
  models: ModelInfo[];
  error?: string;
}

export default function ModelsPage() {
  const router = useRouter();
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [modelsWithStats, setModelsWithStats] = useState<ModelInfoWithStats[]>([]);

  useEffect(() => {
    fetchModels();
  }, []);

  useEffect(() => {
    if (modelsData?.models) {
      fetchStatsForAllModels();
    }
  }, [modelsData]);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `${process.env.BACKEND_URL || 'http://localhost:8000'}/list-models`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setModelsData(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models');
      console.error('Error fetching models:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  const fetchStatsForAllModels = async () => {
    if (!modelsData?.models) return;

    const modelsWithStatsPromises = modelsData.models.map(async (model) => {
      try {
        const response = await fetch(
          `${process.env.BACKEND_URL || 'http://localhost:8000'}/model-stats/${model.session_id}`
        );

        if (response.ok) {
          const statsData = await response.json();
          return { ...model, stats: statsData.stats };
        } else {
          console.warn(`Failed to fetch stats for ${model.session_id}`);
          return model;
        }
      } catch (error) {
        console.warn(`Error fetching stats for ${model.session_id}:`, error);
        return model;
      }
    });

    const modelsWithStats = await Promise.all(modelsWithStatsPromises);
    setModelsWithStats(modelsWithStats);
  };

  const downloadModel = async (model: ModelInfo) => {
    setDownloading(model.session_id);
    try {
      // Generate presigned URL for download using the actual filename
      const response = await fetch(
        `${process.env.BACKEND_URL || 'http://localhost:8000'}/download-model/${model.session_id}?filename=${encodeURIComponent(model.filename)}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        alert(`Download failed: ${data.error}`);
        return;
      }

      // Trigger download using the presigned URL
      const link = document.createElement('a');
      link.href = data.download_url;
      link.download = data.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Download error:', error);
      alert('Failed to download model. Please try again.');
    } finally {
      setDownloading(null);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen bg-black">
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="text-white text-center">
            <div className="text-xl mb-4">Loading Models...</div>
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen bg-black">
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="text-white text-center max-w-md">
            <div className="text-xl mb-4 text-red-400">Error Loading Models</div>
            <div className="text-gray-300 mb-6">{error}</div>
            <button
              onClick={fetchModels}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-black">
      <div className="flex-1 p-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Trained Models</h1>
          </div>

          {/* Models Grid */}
          {modelsWithStats.length > 0 ? (
            <div className="grid grid-cols-1 gap-8">
              {modelsWithStats.map((model) => (
                <div key={model.session_id} className="bg-gray-900 border border-gray-700 rounded-lg p-10 hover:border-gray-600 transition-colors duration-200">
                  <div className="flex items-start justify-between mb-6">
                    <div className="flex-1">
                      <h3 className="text-2xl font-semibold text-white mb-3">
                        {model.filename.replace('.onnx', '')}
                      </h3>
                      <div className="text-sm text-gray-400 space-y-2">
                        <div>Session: {model.session_id}</div>
                        <div>Size: {formatFileSize(model.size)}</div>
                        <div>Trained: {formatDate(model.last_modified)}</div>

                        {model.stats ? (
                          <>
                            {model.stats.final_accuracy !== null && (
                              <div>Train Accuracy: {(model.stats.final_accuracy * 100).toFixed(1)}%</div>
                            )}
                            {model.stats.final_loss !== null && (
                              <div>Train Loss: {model.stats.final_loss.toFixed(4)}</div>
                            )}
                            {model.stats.final_val_accuracy !== null && (
                              <div>Val Accuracy: {(model.stats.final_val_accuracy * 100).toFixed(1)}%</div>
                            )}
                            {model.stats.final_val_loss !== null && (
                              <div>Val Loss: {model.stats.final_val_loss.toFixed(4)}</div>
                            )}
                            {model.stats.dataset_name && (
                              <div>Dataset: {model.stats.dataset_name}</div>
                            )}
                            {model.stats.img_size && (
                              <div>Image Size: {Array.isArray(model.stats.img_size) ? model.stats.img_size.join('×') : model.stats.img_size}</div>
                            )}
                            {model.stats.num_classes && (
                              <div>Classes: {model.stats.num_classes}</div>
                            )}
                            {model.stats.base_model_name && (
                              <div>Base Model: {model.stats.base_model_name}</div>
                            )}
                            {model.stats.total_epochs > 0 && (
                              <div>Epochs: {model.stats.total_epochs}</div>
                            )}
                            {model.stats.best_epoch && (
                              <div>Best Epoch: {model.stats.best_epoch}</div>
                            )}
                          </>
                        ) : (
                          <div className="text-xs text-gray-500 italic">Loading stats...</div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex space-x-3">
                    <button
                      onClick={() => downloadModel(model)}
                      disabled={downloading === model.session_id}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center justify-center gap-2"
                    >
                      {downloading === model.session_id ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Downloading...
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          Download
                        </>
                      )}
                    </button>

                    <button
                      onClick={() => router.push(`/training-console?sessionId=${model.session_id}`)}
                      className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-medium transition-colors duration-200"
                    >
                      View Training
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-gray-900 border border-gray-700 rounded-lg p-12">
              <div className="text-center">
                <h3 className="text-lg font-medium text-white mb-2">No Models Found</h3>
                <p className="text-gray-400 mb-6">
                  No trained models have been uploaded to cloud storage yet.
                  Start a training session to create your first model.
                </p>
                <button
                  onClick={() => router.push('/')}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200"
                >
                  Start Training
                </button>
              </div>
            </div>
          )}

          {/* Refresh Button */}
          <div className="mt-8 text-center">
            <button
              onClick={fetchModels}
              disabled={loading}
              className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200 flex items-center gap-2 mx-auto"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh Models
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
