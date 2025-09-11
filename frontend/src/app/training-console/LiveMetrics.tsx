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
  optimization_iterations?: Array<{
    iteration: number;
    timestamp: string;
    test_results: Record<string, unknown>;
    ai_recommendations?: {
      recommendation_summary?: {
        recommendations_applied?: number;
        total_recommendations?: number;
      };
      applied_changes?: Record<string, any>;
    };
    training_type: string;
    is_optimization: boolean;
  }>;
}

interface LiveMetricsProps {
  sessionId?: string;
  metricsData?: MetricsData | null;
}

export default function LiveMetrics({ sessionId, metricsData }: LiveMetricsProps) {
  // State to preserve iteration history and prevent resets
  const [iterationHistory, setIterationHistory] = React.useState<{
    originalResults: any;
    iterations: Array<{
      iteration: number;
      timestamp: string;
      test_results: Record<string, unknown>;
      ai_recommendations?: any;
      training_type: string;
      is_optimization: boolean;
    }>;
  }>({
    originalResults: null,
    iterations: []
  });

  // Update iteration history when new data arrives
  React.useEffect(() => {
    if (!metricsData) return;

  // Debug log removed for clean UI

    setIterationHistory(prev => {
      const newHistory = { ...prev };
      let hasChanges = false;

      // Update original results if we have them and haven't stored them yet
      if (metricsData.final_test_results && !newHistory.originalResults) {
        newHistory.originalResults = metricsData.final_test_results;
        hasChanges = true;
      }

      // Add new optimization iterations (avoid duplicates)
      if (metricsData.optimization_iterations) {
        const existingIterationIds = new Set(newHistory.iterations.map(iter => iter.iteration));
        metricsData.optimization_iterations.forEach(iter => {
          if (!existingIterationIds.has(iter.iteration)) {
            newHistory.iterations.push(iter);
            newHistory.iterations.sort((a, b) => a.iteration - b.iteration);
            hasChanges = true;
          }
        });
      }

  // No UI log for updated iteration history

      return newHistory;
    });
  }, [metricsData]);

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

  // Get current iteration metrics (only the latest training data, not cumulative)
  const getCurrentIterationMetrics = () => {
    // If we have optimization iterations, we want to show only the current iteration's epoch data
    // For this, we need to determine which metrics belong to the current training
    // Since epoch graphs should reset after each iteration, we'll show all current metrics
    // This works because the training log creates separate entries for each iteration
    return allMetrics;
  };

  const currentIterationMetrics = getCurrentIterationMetrics();

  // Create iteration summary data for the bottom chart (live updates)
  const createIterationSummaryData = () => {
    const labels = [];
    const testAccuracies = [];
    const testLosses = [];
    
    // Use preserved iteration history to maintain state across updates
    // Add original training run results (the baseline before any optimization)
    if (iterationHistory.originalResults) {
      labels.push('Original Run');
      const originalAcc = iterationHistory.originalResults.test_accuracy || 
                          iterationHistory.originalResults.accuracy || 0;
      const originalLoss = iterationHistory.originalResults.test_loss || 
                           iterationHistory.originalResults.loss || 0;
      testAccuracies.push(typeof originalAcc === 'number' ? originalAcc : 0);
      testLosses.push(typeof originalLoss === 'number' ? originalLoss : 0);
    }
    
    // Add optimization iteration results from preserved history
    if (iterationHistory.iterations.length > 0) {
      iterationHistory.iterations.forEach(iter => {
        labels.push(`Iteration ${iter.iteration}`);
        const acc = iter.test_results.test_accuracy || iter.test_results.accuracy || 0;
        const loss = iter.test_results.test_loss || iter.test_results.loss || 0;
        testAccuracies.push(typeof acc === 'number' ? acc : 0);
        testLosses.push(typeof loss === 'number' ? loss : 0);
      });
    }

    // If no data yet, show placeholder
    if (labels.length === 0) {
      labels.push('Waiting for results...');
      testAccuracies.push(0);
      testLosses.push(0);
    }

    return {
      labels,
      accuracyData: testAccuracies,
      lossData: testLosses,
      hasIterations: iterationHistory.iterations.length > 0
    };
  };

  const iterationSummaryData = createIterationSummaryData();
  
  // Debug logging to help troubleshoot the chart data
  console.log('📊 Chart Data Debug:', {
    labels: iterationSummaryData.labels,
    accuracyData: iterationSummaryData.accuracyData,
    lossData: iterationSummaryData.lossData,
    hasIterations: iterationSummaryData.hasIterations,
    preservedHistory: {
      originalResults: !!iterationHistory.originalResults,
      originalAccuracy: iterationHistory.originalResults?.test_accuracy || iterationHistory.originalResults?.accuracy,
      originalLoss: iterationHistory.originalResults?.test_loss || iterationHistory.originalResults?.loss,
      iterationsCount: iterationHistory.iterations.length,
      iterationNumbers: iterationHistory.iterations.map(i => i.iteration),
      iterationData: iterationHistory.iterations.map(iter => ({
        iteration: iter.iteration,
        accuracy: iter.test_results?.test_accuracy || iter.test_results?.accuracy,
        loss: iter.test_results?.test_loss || iter.test_results?.loss
      }))
    },
    currentOptimizationIterations: currentMetricsData?.optimization_iterations?.length || 0
  });

    return (
      <div className="bg-black border border-white/20 rounded-lg p-4">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-bold text-white">Live Training Metrics</h3>
          <div className="flex items-center gap-3">
            <div className={`px-2 py-1 rounded text-xs ${
              connectionStatus === 'connected' ? 'bg-green-600' :
              connectionStatus === 'completed' ? 'bg-blue-600' :
              'bg-yellow-600'
            }`}>
              {connectionStatus === 'connected' ? 'Live' :
               connectionStatus === 'completed' ? 'Completed' :
               'Connecting...'}
            </div>
            {currentMetricsData.stage_info && (
              <div className="text-xs text-gray-400">
                {currentMetricsData.stage_info.message}
              </div>
            )}
          </div>
        </div>

        {/* Final Test Results Section - Show prominently when available */}
        {currentMetricsData.final_test_results && Object.keys(currentMetricsData.final_test_results).length > 0 && (
          <div className="mb-4 bg-blue-900/30 border border-blue-500/50 rounded-lg p-4">
            <h4 className="text-blue-400 font-semibold mb-3 flex items-center gap-2 text-sm">
              Final Test Results
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(currentMetricsData.final_test_results)
                .filter(([key]) => !['test_header'].includes(key))
                .map(([key, value]) => (
                <div key={key} className="text-center">
                  <div className="text-blue-400 text-xs uppercase font-medium mb-1">
                    {key.replace('_', ' ')}
                  </div>
                  <div className="text-white text-lg font-bold">
                    {typeof value === 'number' 
                      ? (key.toLowerCase().includes('acc') || key.toLowerCase().includes('f1') || key.toLowerCase().includes('auc') || key.toLowerCase().includes('precision') || key.toLowerCase().includes('recall'))
                        ? `${(value * 100).toFixed(1)}%`
                        : value.toFixed(4)
                      : String(value)
                    }
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      {!hasData ? (
        <div className="text-center text-gray-400 py-8">
          <p>Training metrics will appear here once training begins...</p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Current Iteration Epoch Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Loss Chart */}
            <div className="h-48">
              <Line
                data={createChartData(currentIterationMetrics, 'Training Loss', '#ef4444', 'loss')}
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'Current Iteration - Loss',
                      font: {
                        size: 12,
                        weight: 'bold' as const,
                      },
                      color: '#ffffff',
                    }
                  }
                }}
              />
            </div>

            {/* Accuracy Chart */}
            <div className="h-48">
              <Line
                data={createChartData(currentIterationMetrics, 'Training Accuracy', '#10b981', 'accuracy')}
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'Current Iteration - Accuracy',
                      font: {
                        size: 12,
                        weight: 'bold' as const,
                      },
                      color: '#ffffff',
                    }
                  },
                  scales: {
                    ...chartOptions.scales,
                    y: { 
                      ...chartOptions.scales.y, 
                      beginAtZero: false,
                      max: 1,
                      // Adaptive scaling with accuracy cap at 1
                    }
                  }
                }}
              />
            </div>

            {/* Validation Loss Chart */}
            <div className="h-48">
              <Line
                data={createChartData(currentIterationMetrics, 'Validation Loss', '#f59e0b', 'val_loss')}
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'Current Iteration - Validation Loss',
                      font: {
                        size: 12,
                        weight: 'bold' as const,
                      },
                      color: '#ffffff',
                    }
                  }
                }}
              />
            </div>

            {/* Validation Accuracy Chart */}
            <div className="h-48">
              <Line
                data={createChartData(currentIterationMetrics, 'Validation Accuracy', '#3b82f6', 'val_accuracy')}
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'Current Iteration - Validation Accuracy',
                      font: {
                        size: 12,
                        weight: 'bold' as const,
                      },
                      color: '#ffffff',
                    }
                  },
                  scales: {
                    ...chartOptions.scales,
                    y: { 
                      ...chartOptions.scales.y, 
                      beginAtZero: false,
                      max: 1,
                      // Adaptive scaling with accuracy cap at 1
                    }
                  }
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* AI Optimization Iterations */}
      {iterationHistory.iterations.length > 0 && (
        <div className="mt-4">
          <h4 className="text-blue-400 font-semibold mb-3 flex items-center gap-2 text-sm">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            AI Optimization Iterations
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            {iterationHistory.iterations.map((iteration, index) => (
              <div key={index} className="p-3 bg-blue-900/20 border border-blue-500/30 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-blue-300 font-medium text-sm">
                    Optimization {iteration.iteration}
                  </h5>
                  <span className="text-xs text-blue-400 bg-blue-900/30 px-1 py-0.5 rounded">
                    {iteration.training_type}
                  </span>
                </div>
                
                {/* Test Results */}
                <div className="mb-2">
                  <div className="text-xs text-blue-400 mb-1">Performance:</div>
                  <div className="grid grid-cols-1 gap-1 text-xs">
                    {Object.entries(iteration.test_results).slice(0, 2).map(([key, value]) => (
                      <div key={key} className="text-blue-200">
                        <span className="text-blue-400 capitalize">{key.replace('_', ' ')}:</span>
                        {typeof value === 'number' && key.toLowerCase().includes('acc')
                          ? ` ${value.toFixed(3)}`
                          : typeof value === 'number'
                            ? ` ${value.toFixed(3)}`
                            : ` ${value}`
                        }
                      </div>
                    ))}
                  </div>
                </div>

                {/* AI Recommendations Applied */}
                {iteration.ai_recommendations && (
                  <div className="mb-2">
                    <div className="text-xs text-blue-400 mb-1">AI Changes:</div>
                    <div className="text-xs text-blue-300">
                      {iteration.ai_recommendations.recommendation_summary?.recommendations_applied || 0} of{' '}
                      {iteration.ai_recommendations.recommendation_summary?.total_recommendations || 0} applied
                    </div>
                  </div>
                )}

                {/* Timestamp */}
                <div className="text-xs text-blue-500 border-t border-blue-500/20 pt-1">
                  {new Date(iteration.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Live Iteration Summary Chart - Always visible at bottom */}
      <div className="mt-6">
        <h4 className="text-md font-semibold text-white mb-3 flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Hyperparameter Tuning Iterations
        </h4>
        
        {!iterationSummaryData.hasIterations && (
          <div className="text-center text-gray-400 py-4 mb-3">
            <p className="text-sm">Optimization results will appear here alongside the original training run...</p>
          </div>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Live Test Accuracy Across Iterations */}
          <div className="h-64 bg-gray-900/50 border border-gray-700 rounded-lg p-3">
            <Line
              data={{
                labels: iterationSummaryData.labels,
                datasets: [{
                  label: 'Test Accuracy',
                  data: iterationSummaryData.accuracyData,
                  borderColor: '#10b981',
                  backgroundColor: '#10b98120',
                  tension: 0.1,
                  fill: false,
                  pointRadius: 5,
                  pointHoverRadius: 7,
                  pointBackgroundColor: '#10b981',
                  pointBorderColor: '#ffffff',
                  pointBorderWidth: 2,
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    display: false,
                  },
                  title: {
                    display: true,
                    text: iterationSummaryData.hasIterations 
                      ? `Test Accuracy Progress` 
                      : 'Test Accuracy (Waiting for data...)',
                    font: {
                      size: 14,
                      weight: 'bold' as const,
                    },
                    color: '#ffffff',
                  },
                },
                scales: {
                  x: {
                    ticks: {
                      color: '#ffffff',
                      font: {
                        size: 11,
                      },
                    },
                    grid: {
                      color: '#374151',
                    },
                  },
                  y: {
                    beginAtZero: false,
                    max: 1,
                    ticks: {
                      color: '#ffffff',
                      font: {
                        size: 11,
                      },
                    },
                    grid: {
                      color: '#374151',
                    },
                  }
                },
                animation: {
                  duration: 600,
                },
              }}
            />
          </div>

          {/* Live Test Loss Across Iterations */}
          <div className="h-64 bg-gray-900/50 border border-gray-700 rounded-lg p-3">
            <Line
              data={{
                labels: iterationSummaryData.labels,
                datasets: [{
                  label: 'Test Loss',
                  data: iterationSummaryData.lossData,
                  borderColor: '#ef4444',
                  backgroundColor: '#ef444420',
                  tension: 0.1,
                  fill: false,
                  pointRadius: 5,
                  pointHoverRadius: 7,
                  pointBackgroundColor: '#ef4444',
                  pointBorderColor: '#ffffff',
                  pointBorderWidth: 2,
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    display: false,
                  },
                  title: {
                    display: true,
                    text: iterationSummaryData.hasIterations 
                      ? `Test Loss Progress` 
                      : 'Test Loss (Waiting for data...)',
                    font: {
                      size: 14,
                      weight: 'bold' as const,
                    },
                    color: '#ffffff',
                  },
                },
                scales: {
                  x: {
                    ticks: {
                      color: '#ffffff',
                      font: {
                        size: 11,
                      },
                    },
                    grid: {
                      color: '#374151',
                    },
                  },
                  y: {
                    beginAtZero: true,
                    ticks: {
                      color: '#ffffff',
                      font: {
                        size: 11,
                      },
                    },
                    grid: {
                      color: '#374151',
                    },
                  }
                },
                animation: {
                  duration: 600,
                },
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
