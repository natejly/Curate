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
  // Training metrics (from epochs)
  final_training_accuracy: number | null;
  final_training_loss: number | null;
  final_val_accuracy: number | null;
  final_val_loss: number | null;
  total_epochs: number;
  best_epoch: number | null;
  
  // Test metrics (from model.evaluate())
  test_accuracy: number | null;
  test_loss: number | null;
  
  // General info
  training_time: string | null;
  dataset_name: string | null;
  dataset_path: string | null;
  img_size: any;
  num_classes: number | null;
  base_model_name: string | null;
  
  // Model details
  model_total_parameters: number | null;
  model_trainable_parameters: number | null;
  model_non_trainable_parameters: number | null;
  base_model_layers: number | null;
  base_model_trainable: boolean | null;
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
  const [statsLoading, setStatsLoading] = useState(false);
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

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
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/list-models`);

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

    setStatsLoading(true);
    console.log(`Fetching stats for ${modelsData.models.length} models...`);
    
    const modelsWithStatsPromises = modelsData.models.map(async (model) => {
      try {
        const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
        const url = `${backendUrl}/model-stats/${model.session_id}`;
        
        console.log(`Fetching stats from: ${url}`);
        
        const response = await fetch(url, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const statsData = await response.json();
          console.log(`Stats for ${model.session_id}:`, statsData);
          return { ...model, stats: statsData.stats };
        } else {
          const errorText = await response.text();
          console.warn(`Failed to fetch stats for ${model.session_id} (${response.status}):`, errorText);
          return model;
        }
      } catch (error) {
        console.error(`Error fetching stats for ${model.session_id}:`, error);
        return model;
      }
    });

    const modelsWithStats = await Promise.all(modelsWithStatsPromises);
    setModelsWithStats(modelsWithStats);
    setStatsLoading(false);
  };

  const downloadModel = async (model: ModelInfo) => {
    setDownloading(model.session_id);
    try {
      // Generate presigned URL for download using the actual filename
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(
        `${backendUrl}/download-model/${model.session_id}?filename=${encodeURIComponent(model.filename)}`
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

  if (loading || statsLoading) {
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
              className="bg-white text-black px-6 py-2 rounded-lg font-semibold hover:bg-white/90 transition-colors duration-200 shadow-lg"
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
                <div key={model.session_id} className="bg-black/60 backdrop-blur-lg border border-white/30 rounded-2xl p-8 hover:bg-black/70 hover:border-white/40 transition-all duration-300 shadow-lg">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="text-2xl font-semibold text-white mb-3">
                        {model.filename.replace('.onnx', '')}
                      </h3>
                    </div>
                  </div>

                  <div className="flex justify-end space-x-3 mt-6">
                    <button
                      onClick={() => downloadModel(model)}
                      disabled={downloading === model.session_id}
                      className="bg-white text-black hover:bg-white/90 disabled:bg-white/60 disabled:cursor-not-allowed px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center justify-center gap-1 shadow-lg"
                    >
                      {downloading === model.session_id ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-3 w-3 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Downloading...
                        </>
                      ) : (
                        <>
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          Download
                        </>
                      )}
                    </button>

                    <button
                      onClick={() => setExpandedModel(expandedModel === model.session_id ? null : model.session_id)}
                      className="px-4 py-2 bg-black/40 backdrop-blur-sm border border-white/20 hover:bg-black/60 hover:border-white/30 text-white rounded-lg text-sm font-medium transition-all duration-200 shadow-lg"
                    >
                      {expandedModel === model.session_id ? 'Hide Information' : 'View Information'}
                    </button>
                  </div>

                  {/* Expanded Information */}
                  {expandedModel === model.session_id && (
                    <div className="mt-6 space-y-4 animate-in slide-in-from-top-2 duration-300">
                      {/* Training Stats */}
                      {model.stats ? (
                        <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-4">
                          <h3 className="text-lg font-semibold text-white mb-3">Training Results</h3>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            {/* Test Metrics (Most Important) */}
                            {model.stats.test_accuracy !== null && (
                              <div className="col-span-2 bg-green-900/20 border border-green-500/30 rounded-lg p-3 mb-2">
                                <span className="text-green-300 font-semibold">🎯 Test Accuracy:</span>
                                <div className="text-green-400 font-bold text-lg">{(model.stats.test_accuracy * 100).toFixed(1)}%</div>
                                <div className="text-xs text-green-300/70">Final performance on unseen test data</div>
                              </div>
                            )}
                            {model.stats.test_loss !== null && (
                              <div className="col-span-2 bg-red-900/20 border border-red-500/30 rounded-lg p-3 mb-4">
                                <span className="text-red-300 font-semibold">📊 Test Loss:</span>
                                <div className="text-red-400 font-bold text-lg">{model.stats.test_loss.toFixed(4)}</div>
                                <div className="text-xs text-red-300/70">Final loss on unseen test data</div>
                              </div>
                            )}
                            
                            {/* Training Metrics */}
                            {model.stats.final_training_accuracy !== null && (
                              <div>
                                <span className="text-gray-400">Final Train Accuracy:</span>
                                <div className="text-green-400 font-semibold">{(model.stats.final_training_accuracy * 100).toFixed(1)}%</div>
                              </div>
                            )}
                            {model.stats.final_training_loss !== null && (
                              <div>
                                <span className="text-gray-400">Final Train Loss:</span>
                                <div className="text-yellow-400">{model.stats.final_training_loss.toFixed(4)}</div>
                              </div>
                            )}
                            {model.stats.final_val_accuracy !== null && (
                              <div>
                                <span className="text-gray-400">Final Val Accuracy:</span>
                                <div className="text-blue-400 font-semibold">{(model.stats.final_val_accuracy * 100).toFixed(1)}%</div>
                              </div>
                            )}
                            {model.stats.final_val_loss !== null && (
                              <div>
                                <span className="text-gray-400">Final Val Loss:</span>
                                <div className="text-orange-400">{model.stats.final_val_loss.toFixed(4)}</div>
                              </div>
                            )}
                            {model.stats.total_epochs > 0 && (
                              <div>
                                <span className="text-gray-400">Total Epochs:</span>
                                <div className="text-white">{model.stats.total_epochs}</div>
                              </div>
                            )}
                            {model.stats.best_epoch && (
                              <div>
                                <span className="text-gray-400">Best Epoch:</span>
                                <div className="text-purple-400 font-semibold">{model.stats.best_epoch}</div>
                              </div>
                            )}
                          </div>
                        </div>
                      ) : (
                        <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-4 text-center">
                          <div className="text-gray-400">
                            {statsLoading ? 'Loading training statistics...' : 'No training statistics available'}
                          </div>
                          {!statsLoading && (
                            <div className="text-xs text-gray-500 mt-2">
                              Training log may not be available for this model
                            </div>
                          )}
                        </div>
                      )}

                      {/* Model Architecture Info */}
                      {model.stats && (
                        <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-4">
                          <h3 className="text-lg font-semibold text-white mb-3">Model Architecture</h3>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            {model.stats.base_model_name && (
                              <div>
                                <span className="text-gray-400">Base Model:</span>
                                <div className="text-white font-semibold">{model.stats.base_model_name}</div>
                              </div>
                            )}
                            {model.stats.model_total_parameters && (
                              <div>
                                <span className="text-gray-400">Total Parameters:</span>
                                <div className="text-white">{model.stats.model_total_parameters.toLocaleString()}</div>
                              </div>
                            )}
                            {model.stats.model_trainable_parameters && (
                              <div>
                                <span className="text-gray-400">Trainable Parameters:</span>
                                <div className="text-green-400 font-semibold">{model.stats.model_trainable_parameters.toLocaleString()}</div>
                              </div>
                            )}
                            {model.stats.model_non_trainable_parameters && (
                              <div>
                                <span className="text-gray-400">Frozen Parameters:</span>
                                <div className="text-blue-400">{model.stats.model_non_trainable_parameters.toLocaleString()}</div>
                              </div>
                            )}
                            {model.stats.base_model_layers && (
                              <div>
                                <span className="text-gray-400">Base Model Layers:</span>
                                <div className="text-white">{model.stats.base_model_layers}</div>
                              </div>
                            )}
                            {model.stats.base_model_trainable !== null && (
                              <div>
                                <span className="text-gray-400">Base Model Status:</span>
                                <div className={model.stats.base_model_trainable ? "text-green-400" : "text-blue-400"}>
                                  {model.stats.base_model_trainable ? "Trainable" : "Frozen"}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Dataset Info */}
                      {model.stats && (
                        <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-4">
                          <h3 className="text-lg font-semibold text-white mb-3">Dataset Information</h3>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            {model.stats.dataset_name && (
                              <div>
                                <span className="text-gray-400">Dataset:</span>
                                <div className="text-white">{model.stats.dataset_name}</div>
                              </div>
                            )}
                            {model.stats.img_size && (
                              <div>
                                <span className="text-gray-400">Image Size:</span>
                                <div className="text-white">
                                  {Array.isArray(model.stats.img_size)
                                    ? model.stats.img_size.join('×')
                                    : model.stats.img_size
                                  }
                                </div>
                              </div>
                            )}
                            {model.stats.num_classes && (
                              <div>
                                <span className="text-gray-400">Classes:</span>
                                <div className="text-white">{model.stats.num_classes}</div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-black/60 backdrop-blur-lg border border-white/30 rounded-2xl p-12 text-center">
              <h3 className="text-xl font-semibold text-white mb-4">No Models Found</h3>
              <p className="text-gray-300 mb-8 leading-relaxed">
                No trained models have been uploaded to cloud storage yet.
                Start a training session to create your first model.
              </p>
              <button
                onClick={() => router.push('/')}
                className="bg-white text-black px-8 py-3 rounded-lg font-semibold hover:bg-white/90 transition-colors duration-200 shadow-lg"
              >
                Start Training
              </button>
            </div>
          )}

          {/* Refresh Button */}
          <div className="mt-8 text-center">
            <button
              onClick={fetchModels}
              disabled={loading || statsLoading}
              className="bg-black/60 backdrop-blur-lg border border-white/30 hover:bg-black/70 hover:border-white/40 disabled:bg-black/40 disabled:border-white/20 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 flex items-center gap-2 mx-auto shadow-lg"
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
