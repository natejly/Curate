'use client'

import { useState, useRef } from 'react'

export default function TestingTab() {
  const [testInput, setTestInput] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [results, setResults] = useState<any>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null
    setSelectedFile(file)
  }

  const handleTest = async () => {
    if (!testInput && !selectedFile) return

    setIsProcessing(true)

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000))

    // Mock results based on input type
    const mockResults = {
      confidence: Math.random() * 100,
      prediction: selectedFile
        ? `Detected: ${selectedFile.name.split('.')[0]}`
        : `Predicted: ${testInput}`,
      processingTime: '1.2s',
      modelUsed: 'Trained Model v1.0'
    }

    setResults(mockResults)
    setIsProcessing(false)
  }

  const testTypes = [
    {
      id: 'text',
      name: 'Text Analysis',
      icon: 'TEXT',
      placeholder: 'Enter text to analyze...',
      description: 'Analyze sentiment, classify content, or extract insights'
    },
    {
      id: 'image',
      name: 'Image Analysis',
      icon: 'IMAGE',
      placeholder: 'Upload an image to analyze',
      description: 'Object detection, image classification, or feature extraction'
    },
    {
      id: 'audio',
      name: 'Audio Analysis',
      icon: 'AUDIO',
      placeholder: 'Upload audio file to process',
      description: 'Speech recognition, audio classification, or transcription'
    }
  ]

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
          Testing Inference
        </h2>
        <p className="text-gray-300 text-lg">
          Test your trained model with new data
        </p>
      </div>

      {/* Test Type Selection */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-white mb-4">Select Test Type</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {testTypes.map((type) => (
            <div
              key={type.id}
              className="bg-gray-800/50 rounded-lg p-6 border border-gray-600 hover:border-gray-400 transition-colors"
            >
              <div className="text-sm font-bold mb-3 bg-gray-700 px-3 py-1 rounded text-gray-300">{type.icon}</div>
              <h4 className="text-white font-semibold mb-2">{type.name}</h4>
              <p className="text-gray-400 text-sm">{type.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Input Section */}
      <div className="mb-8">
        <div className="space-y-4">
          {/* Text Input */}
          <div>
            <label className="block text-white font-medium mb-2">
              Text Input
            </label>
            <textarea
              value={testInput}
              onChange={(e) => setTestInput(e.target.value)}
              placeholder="Enter text to test your model..."
              className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-3 text-white focus:border-blue-400 focus:outline-none resize-none"
              rows={4}
            />
          </div>

          {/* File Input */}
          <div>
            <label className="block text-white font-medium mb-2">
              Or Upload File
            </label>
            <div className="flex items-center space-x-4">
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileSelect}
                className="hidden"
                accept="image/*,audio/*,.txt"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg text-white transition-colors"
              >
                Choose File
              </button>
              {selectedFile && (
                <div className="flex items-center text-gray-300">
                  <span className="mr-2 text-sm font-bold bg-gray-700 px-2 py-1 rounded">FILE</span>
                  {selectedFile.name}
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="ml-2 text-red-400 hover:text-red-300"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Test Button */}
      <div className="text-center mb-8">
        <button
          onClick={handleTest}
          disabled={isProcessing || (!testInput && !selectedFile)}
          className={`px-8 py-3 rounded-lg font-semibold transition-all duration-300 ${
            isProcessing || (!testInput && !selectedFile)
              ? 'bg-gray-600 cursor-not-allowed'
              : 'bg-green-600 hover:bg-green-700 hover:scale-105'
          }`}
        >
          {isProcessing ? (
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              Processing...
            </div>
          ) : (
            'Run Inference'
          )}
        </button>
      </div>

      {/* Results Section */}
      {results && (
        <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-600">
          <h3 className="text-xl font-semibold text-white mb-4">Results</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-gray-300 font-medium mb-2">Prediction</h4>
              <div className="text-white text-lg font-semibold">
                {results.prediction}
              </div>
            </div>

            <div>
              <h4 className="text-gray-300 font-medium mb-2">Confidence</h4>
              <div className="flex items-center">
                <div className="flex-1 bg-gray-700 rounded-full h-2 mr-2">
                  <div
                    className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                    style={{ width: `${results.confidence}%` }}
                  />
                </div>
                <span className="text-white font-semibold">
                  {Math.round(results.confidence)}%
                </span>
              </div>
            </div>

            <div>
              <h4 className="text-gray-300 font-medium mb-2">Processing Time</h4>
              <div className="text-white">{results.processingTime}</div>
            </div>

            <div>
              <h4 className="text-gray-300 font-medium mb-2">Model Used</h4>
              <div className="text-white">{results.modelUsed}</div>
            </div>
          </div>

          {/* Additional Details */}
          <div className="mt-6 pt-6 border-t border-gray-600">
            <h4 className="text-gray-300 font-medium mb-3">Additional Details</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-2xl mb-2">🎯</div>
                <div className="text-white font-semibold">High Accuracy</div>
                <div className="text-gray-400 text-sm">Model Performance</div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-2xl mb-2">⚡</div>
                <div className="text-white font-semibold">Fast Inference</div>
                <div className="text-gray-400 text-sm">Response Time</div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-2xl mb-2">🔄</div>
                <div className="text-white font-semibold">Real-time</div>
                <div className="text-gray-400 text-sm">Processing Mode</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Placeholder for future features */}
      <div className="mt-12 opacity-50">
        <h3 className="text-xl font-semibold text-gray-400 mb-4">
          Advanced Testing Features (Coming Soon)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-800/30 rounded-lg p-6">
            <h4 className="text-white font-medium mb-2">Batch Testing</h4>
            <p className="text-gray-400 text-sm">Test multiple inputs simultaneously</p>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-6">
            <h4 className="text-white font-medium mb-2">Performance Metrics</h4>
            <p className="text-gray-400 text-sm">Detailed accuracy and speed analysis</p>
          </div>
        </div>
      </div>
    </div>
  )
}
