'use client'

import { useState } from 'react'

export default function MLTrainingTab() {
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [selectedModel, setSelectedModel] = useState('classification')
  const [epochs, setEpochs] = useState(100)

  const startTraining = () => {
    setIsTraining(true)
    setProgress(0)

    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsTraining(false)
          return 100
        }
        return prev + Math.random() * 5
      })
    }, 500)
  }

  const models = [
    { id: 'classification', name: 'Image Classification', icon: 'CLASS' },
    { id: 'detection', name: 'Object Detection', icon: 'DETECT' },
    { id: 'segmentation', name: 'Image Segmentation', icon: 'SEGMENT' },
    { id: 'nlp', name: 'Natural Language Processing', icon: 'NLP' },
    { id: 'audio', name: 'Audio Processing', icon: 'AUDIO' }
  ]

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
          Machine Learning Training
        </h2>
        <p className="text-gray-300 text-lg">
          Train your AI models with uploaded data
        </p>
      </div>

      {/* Model Selection */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-white mb-4">Select Model Type</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {models.map((model) => (
            <button
              key={model.id}
              onClick={() => setSelectedModel(model.id)}
              className={`p-4 rounded-lg border transition-all duration-300 ${
                selectedModel === model.id
                  ? 'border-blue-400 bg-blue-500/20'
                  : 'border-gray-600 bg-gray-800/50 hover:border-gray-400'
              }`}
            >
              <div className="text-sm font-bold mb-3 bg-gray-700 px-3 py-1 rounded text-gray-300">{model.icon}</div>
              <div className="text-white font-medium">{model.name}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Training Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div>
          <label className="block text-white font-medium mb-2">
            Training Epochs
          </label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white focus:border-blue-400 focus:outline-none"
            placeholder="100"
          />
        </div>

        <div>
          <label className="block text-white font-medium mb-2">
            Learning Rate
          </label>
          <select className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white focus:border-blue-400 focus:outline-none">
            <option value="0.001">0.001 (Default)</option>
            <option value="0.01">0.01</option>
            <option value="0.0001">0.0001</option>
          </select>
        </div>
      </div>

      {/* Training Metrics Placeholder */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-gray-800/50 rounded-lg p-6 text-center">
          <div className="text-sm font-bold mb-2 bg-gray-700 px-2 py-1 rounded text-gray-300">ACC</div>
          <div className="text-gray-400 text-sm">Accuracy</div>
          <div className="text-white text-xl font-bold">--</div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-6 text-center">
          <div className="text-sm font-bold mb-2 bg-gray-700 px-2 py-1 rounded text-gray-300">LOSS</div>
          <div className="text-gray-400 text-sm">Loss</div>
          <div className="text-white text-xl font-bold">--</div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-6 text-center">
          <div className="text-sm font-bold mb-2 bg-gray-700 px-2 py-1 rounded text-gray-300">ETA</div>
          <div className="text-gray-400 text-sm">ETA</div>
          <div className="text-white text-xl font-bold">--</div>
        </div>
      </div>

      {/* Training Progress */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-semibold text-white">Training Progress</h3>
          <span className="text-gray-400">{Math.round(progress)}%</span>
        </div>
        <div className="bg-gray-700 rounded-full h-3">
          <div
            className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Training Button */}
      <div className="text-center">
        <button
          onClick={startTraining}
          disabled={isTraining}
          className={`px-8 py-4 rounded-lg font-semibold text-lg transition-all duration-300 ${
            isTraining
              ? 'bg-gray-600 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 hover:scale-105 shadow-lg'
          }`}
        >
          {isTraining ? (
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              Training in Progress...
            </div>
          ) : (
            'Start Training'
          )}
        </button>
      </div>

      {/* Skeleton/Placeholder Content */}
      <div className="mt-12 opacity-50">
        <h3 className="text-xl font-semibold text-gray-400 mb-4">
          Advanced Settings (Coming Soon)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-800/30 rounded-lg p-6 animate-pulse">
            <div className="h-4 bg-gray-600 rounded mb-4"></div>
            <div className="h-8 bg-gray-600 rounded mb-4"></div>
            <div className="h-4 bg-gray-600 rounded"></div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-6 animate-pulse">
            <div className="h-4 bg-gray-600 rounded mb-4"></div>
            <div className="h-8 bg-gray-600 rounded mb-4"></div>
            <div className="h-4 bg-gray-600 rounded"></div>
          </div>
        </div>
      </div>
    </div>
  )
}
