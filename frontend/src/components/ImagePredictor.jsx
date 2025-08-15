import React, { useState, useEffect } from 'react';

const ImagePredictor = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [fileInputKey, setFileInputKey] = useState(0); // Add this to force re-render of file input

  // Fetch available models on component mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('/api/ml/models');
      const data = await response.json();
      
      if (data.success && data.models) {
        setAvailableModels(data.models);
        // Auto-select the most recent model
        if (data.models.length > 0) {
          setSelectedModel(data.models[0].name);
        }
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
      
      // Clear previous prediction
      setPrediction(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile || !selectedModel) {
      alert('Please select both an image and a model');
      return;
    }

    setLoading(true);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`/api/ml/predict?model_name=${encodeURIComponent(selectedModel)}`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        setPrediction(data);
      } else {
        alert(`Prediction failed: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error making prediction:', error);
      alert('Error making prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setPrediction(null);
    setFileInputKey(prev => prev + 1); // Force file input to re-render
  };

  const handlePredictAnother = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setPrediction(null);
    setFileInputKey(prev => prev + 1); // Reset file input for new selection
  };

  return (
    <div className="card-hover max-w-5xl mx-auto">
      <div className="flex items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Image Prediction</h2>
      </div>
      
      {/* Model Selection */}
      <div className="mb-6 animate-fade-in">
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Select Trained Model:
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="input-modern"
        >
          <option value="">-- Select a model --</option>
          {availableModels.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name} ({model.num_classes} classes, {model.training_accuracy ? 
                `${(model.training_accuracy * 100).toFixed(1)}% accuracy` : 'no accuracy data'})
            </option>
          ))}
        </select>
        
        {availableModels.length === 0 && (
          <div className="mt-3 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center">
              <div>
                <p className="text-yellow-800 font-medium">No trained models available</p>
                <p className="text-yellow-600 text-sm mt-1">Please train a model first in the Train tab</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Image Upload */}
      <div className="mb-6 animate-fade-in">
        <label className="block text-sm font-medium text-gray-700 mb-3">
          📸 Upload Image for Prediction:
        </label>
        <input
          key={fileInputKey}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="input-modern"
        />
      </div>

      {/* Image Preview */}
      {imagePreview && (
        <div className="mb-6 animate-slide-up">
          <h3 className="text-lg font-medium text-gray-700 mb-3 flex items-center">
            <span className="mr-2">🖼️</span>
            Image Preview:
          </h3>
          <div className="flex justify-center">
            <div className="relative">
              <img
                src={imagePreview}
                alt="Preview"
                className="max-w-xs max-h-64 object-contain border border-gray-300 rounded-xl shadow-lg"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent rounded-xl pointer-events-none"></div>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-4 mb-6 animate-fade-in">
        {!prediction ? (
          <>
            <button
              onClick={handlePredict}
              disabled={!selectedFile || !selectedModel || loading}
              className="flex-1 btn-primary"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Predicting...
                </span>
              ) : (
                '🎯 Predict'
              )}
            </button>
            
            <button
              onClick={handleClear}
              className="btn-secondary"
            >
              🗑️ Clear
            </button>
          </>
        ) : (
          <>
            <button
              onClick={handlePredictAnother}
              className="flex-1 btn-primary"
            >
              🔄 Predict Another Image
            </button>
            
            <button
              onClick={() => setPrediction(null)}
              className="btn-secondary"
            >
              👁️ View Image Only
            </button>
          </>
        )}
      </div>

      {/* Prediction Results */}
      {prediction && (
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6 shadow-lg animate-slide-up">
          <div className="flex items-center mb-4">
            <div className="text-2xl mr-3">🎉</div>
            <h3 className="text-xl font-bold text-green-800">Prediction Results</h3>
          </div>
          
          {/* Main Prediction */}
          <div className="mb-6 p-4 bg-white rounded-lg border shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <span className="text-lg font-medium text-gray-700">Predicted Class:</span>
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                {prediction.predicted_class}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Confidence:</span>
              <div className="flex items-center space-x-2">
                <div className="progress-bar w-24">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${parseFloat(prediction.confidence_percentage)}%` }}
                  ></div>
                </div>
                <span className="text-lg font-bold text-green-600">{prediction.confidence_percentage}</span>
              </div>
            </div>
          </div>

          {/* Top Predictions */}
          {prediction.top_predictions && prediction.top_predictions.length > 1 && (
            <div className="mb-4">
              <h4 className="font-medium text-gray-700 mb-3 flex items-center">
                <span className="mr-2">📊</span>
                Top Predictions:
              </h4>
              <div className="space-y-2">
                {prediction.top_predictions.map((pred, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border">
                    <span className="font-medium text-gray-700">{pred.class}</span>
                    <div className="flex items-center space-x-2">
                      <div className="progress-bar w-20">
                        <div 
                          className="h-full bg-gradient-to-r from-blue-400 to-blue-500 rounded-full transition-all duration-500 ease-out" 
                          style={{ width: `${parseFloat(pred.percentage)}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-600 w-12 text-right">{pred.percentage}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center text-xs text-gray-500 bg-white bg-opacity-50 rounded-lg p-2">
            <span className="mr-1">🤖</span>
            Model used: <span className="font-mono ml-1">{prediction.model_used}</span>
          </div>
        </div>
      )}

      {/* Model Info */}
      {selectedModel && availableModels.length > 0 && (
        <div className="mt-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200 animate-fade-in">
          <div className="flex items-center mb-4">
            <div className="text-2xl mr-3">ℹ️</div>
            <h3 className="font-bold text-lg text-gray-800">Selected Model Information</h3>
          </div>
          {(() => {
            const model = availableModels.find(m => m.name === selectedModel);
            return model ? (
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-gray-600 w-20">Type:</span>
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium">
                      {model.model_type}
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-gray-600 w-20">Classes:</span>
                    <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs font-medium">
                      {model.num_classes}
                    </span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-gray-600 w-24">Training:</span>
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-medium">
                      {model.training_accuracy ? `${(model.training_accuracy * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-gray-600 w-24">Validation:</span>
                    <span className="px-2 py-1 bg-orange-100 text-orange-800 rounded text-xs font-medium">
                      {model.validation_accuracy ? `${(model.validation_accuracy * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            ) : null;
          })()}
        </div>
      )}
    </div>
  );
};

export default ImagePredictor;
