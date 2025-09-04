"use client";

import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface MetricsData {
  session_id: string;
  stage1_metrics: Array<{
    epoch: number;
    loss: number;
    accuracy: number;
    val_loss: number;
    val_accuracy: number;
    timestamp: string;
  }>;
  stage2_metrics: Array<{
    epoch: number;
    loss: number;
    accuracy: number;
    val_loss: number;
    val_accuracy: number;
    timestamp: string;
  }>;
  current_stage: number;
  training_status: string;
  stage_info?: {
    stage: number | string;
    type: string;
    message: string;
  };
  final_test_results?: Record<string, unknown> | null;
}

interface LiveMetricsProps {
  sessionId?: string;
  metricsData?: MetricsData | null;
}

export default function LiveMetrics({ sessionId, metricsData }: LiveMetricsProps) {
  // Determine connection status based on metrics data
  const getConnectionStatus = () => {
    if (!metricsData) return 'connecting';
    if (metricsData.training_status === 'completed') return 'completed';
    if (metricsData.stage1_metrics.length > 0 || metricsData.stage2_metrics.length > 0) return 'connected';
    return 'connecting';
  };

  const connectionStatus = getConnectionStatus();

  // Use metricsData directly instead of component state
  const currentMetricsData = metricsData;

  // Handle model download
  const handleDownloadModel = async (format: 'keras' | 'onnx') => {
    if (!sessionId) return;

    try {
      const response = await fetch(`${process.env.BACKEND_URL || 'http://localhost:8000'}/download-model/${sessionId}?format=${format}`);
      if (!response.ok) {
        throw new Error('Failed to get download URL');
      }

      const data = await response.json();

      if (data.download_url) {
        // Create a temporary link and trigger download
        const link = document.createElement('a');
        link.href = data.download_url;
        link.download = data.filename || `trained_model.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else if (data.available_formats) {
        alert(`Available formats: ${data.available_formats.join(', ')}. Please try a different format.`);
      } else {
        alert(`Model download not available for ${format} format. Please check that training has completed successfully.`);
      }
    } catch (error) {
      console.error('Download error:', error);
      alert(`Failed to download ${format} model. Please try again later.`);
    }
  };

  const createChartData = (metrics: Array<{ epoch: number; loss: number; accuracy: number; val_loss: number; val_accuracy: number; timestamp: string }>, label: string, color: string, valueKey: keyof typeof metrics[0]) => {
    return {
      labels: metrics.map(m => `Epoch ${m.epoch}`),
      datasets: [{
        label,
        data: metrics.map(m => m[valueKey] as number),
        borderColor: color,
        backgroundColor: color + '20',
        tension: 0.1,
        fill: false,
        pointRadius: 3,
        pointHoverRadius: 5,
      }]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Training Metrics',
        font: {
          size: 16,
          weight: 'bold' as const,
        },
        color: '#ffffff',
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        beginAtZero: true,
        ticks: {
          color: '#ffffff',
        },
        grid: {
          color: '#374151',
        },
      }
    },
    animation: {
      duration: 300,
    },
  };

  if (!currentMetricsData) {
    return (
      <div className="bg-black border border-white/20 rounded-lg p-6">
        <div className="text-white text-center">
          <div className="mb-4">
            <div className={`inline-block px-3 py-1 rounded text-sm ${
              connectionStatus === 'connecting' ? 'bg-yellow-600' :
              'bg-blue-600'
            }`}>
                          {connectionStatus === 'connecting' ? 'Connecting to metrics...' :
             'Waiting for data...'}
            </div>
          </div>
          <p className="text-gray-400">Waiting for training metrics...</p>
        </div>
      </div>
    );
  }

  const allMetrics = [...currentMetricsData.stage1_metrics, ...currentMetricsData.stage2_metrics];
  const hasData = allMetrics.length > 0;

  return (
    <div className="bg-black border border-white/20 rounded-lg p-6">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-xl font-bold text-white">Live Training Metrics</h3>
        <div className="flex items-center gap-4">
          <div className={`px-3 py-1 rounded text-sm ${
            connectionStatus === 'connected' ? 'bg-green-600' :
            connectionStatus === 'completed' ? 'bg-blue-600' :
            'bg-yellow-600'
          }`}>
            {connectionStatus === 'connected' ? 'Live' :
             connectionStatus === 'completed' ? 'Completed' :
             'Connecting...'}
          </div>
          {currentMetricsData.stage_info && (
            <div className="text-sm text-gray-400">
              {currentMetricsData.stage_info.message}
            </div>
          )}
        </div>
      </div>

      {!hasData ? (
        <div className="text-center text-gray-400 py-8">
          <p>Training metrics will appear here once training begins...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Loss Chart */}
          <div className="h-64">
            <Line
              data={createChartData(allMetrics, 'Training Loss', '#ef4444', 'loss')}
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: 'Loss',
                    font: {
                      size: 14,
                      weight: 'bold' as const,
                    },
                    color: '#ffffff',
                  }
                }
              }}
            />
          </div>

          {/* Accuracy Chart */}
          <div className="h-64">
            <Line
              data={createChartData(allMetrics, 'Training Accuracy', '#10b981', 'accuracy')}
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: 'Accuracy',
                    font: {
                      size: 14,
                      weight: 'bold' as const,
                    },
                    color: '#ffffff',
                  }
                },
                scales: {
                  ...chartOptions.scales,
                  y: { ...chartOptions.scales.y, max: 1 }
                }
              }}
            />
          </div>

          {/* Validation Loss Chart */}
          <div className="h-64">
            <Line
              data={createChartData(allMetrics, 'Validation Loss', '#f59e0b', 'val_loss')}
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: 'Validation Loss',
                    font: {
                      size: 14,
                      weight: 'bold' as const,
                    },
                    color: '#ffffff',
                  }
                }
              }}
            />
          </div>

          {/* Validation Accuracy Chart */}
          <div className="h-64">
            <Line
              data={createChartData(allMetrics, 'Validation Accuracy', '#3b82f6', 'val_accuracy')}
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: 'Validation Accuracy',
                    font: {
                      size: 14,
                      weight: 'bold' as const,
                    },
                    color: '#ffffff',
                  }
                },
                scales: {
                  ...chartOptions.scales,
                  y: { ...chartOptions.scales.y, max: 1 }
                }
              }}
            />
          </div>
        </div>
      )}

      {/* Metrics Summary */}
      {hasData && (
        <div className="mt-6 p-4 bg-gray-900 rounded-lg">
          <h4 className="text-white font-semibold mb-2">Latest Training Metrics</h4>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
            <div className="text-gray-300">
              <span className="text-gray-500">Epoch:</span> {allMetrics[allMetrics.length - 1]?.epoch}
            </div>
            <div className="text-gray-300">
              <span className="text-gray-500">Loss:</span> {allMetrics[allMetrics.length - 1]?.loss.toFixed(4)}
            </div>
            <div className="text-gray-300">
              <span className="text-gray-500">Accuracy:</span> {(allMetrics[allMetrics.length - 1]?.accuracy * 100).toFixed(2)}%
            </div>
            <div className="text-gray-300">
              <span className="text-gray-500">Val Acc:</span> {(allMetrics[allMetrics.length - 1]?.val_accuracy * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      )}

      {/* Final Test Results */}
      {currentMetricsData.final_test_results && (
        <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
          <h4 className="text-green-400 font-semibold mb-3 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Final Test Results
          </h4>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
            {Object.entries(currentMetricsData.final_test_results).map(([key, value]) => (
              <div key={key} className="text-green-300">
                <span className="text-green-500 capitalize">{key.replace('_', ' ')}:</span>
                {typeof value === 'number' && key.toLowerCase().includes('acc')
                  ? ` ${(value * 100).toFixed(2)}%`
                  : typeof value === 'number'
                    ? ` ${value.toFixed(4)}`
                    : ` ${value}`
                }
              </div>
            ))}
          </div>
                      <div className="mt-3 flex items-center justify-between">
            <div className="text-xs text-green-500">
              These results show the model&apos;s performance on the held-out test dataset
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => handleDownloadModel('keras')}
                className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center gap-1"
                title="Download TensorFlow model (.keras)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                .keras
              </button>
              <button
                onClick={() => handleDownloadModel('onnx')}
                className="bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center gap-1"
                title="Download ONNX model (.onnx)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                .onnx
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
