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
      const response = await fetch('http://localhost:8000/api/ml/models');
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

      const response = await fetch(`http://localhost:8000/api/ml/predict?model_name=${encodeURIComponent(selectedModel)}`, {
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
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">🔮 Image Prediction</h2>
      
      {/* Model Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Trained Model:
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
          <p className="text-gray-500 text-sm mt-2">
            No trained models available. Please train a model first.
          </p>
        )}
      </div>

      {/* Image Upload */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Upload Image for Prediction:
        </label>
        <input
          key={fileInputKey}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* Image Preview */}
      {imagePreview && (
        <div className="mb-6">
          <h3 className="text-lg font-medium text-gray-700 mb-2">Image Preview:</h3>
          <div className="flex justify-center">
            <img
              src={imagePreview}
              alt="Preview"
              className="max-w-xs max-h-64 object-contain border border-gray-300 rounded-lg"
            />
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-4 mb-6">
        {!prediction ? (
          <>
            <button
              onClick={handlePredict}
              disabled={!selectedFile || !selectedModel || loading}
              className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
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
              className="bg-gray-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-600 transition-colors"
            >
              🗑️ Clear
            </button>
          </>
        ) : (
          <>
            <button
              onClick={handlePredictAnother}
              className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              🔄 Predict Another Image
            </button>
            
            <button
              onClick={() => setPrediction(null)}
              className="bg-gray-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-600 transition-colors"
            >
              👁️ View Image Only
            </button>
          </>
        )}
      </div>

      {/* Prediction Results */}
      {prediction && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <h3 className="text-lg font-bold text-green-800 mb-4">🎉 Prediction Results</h3>
          
          {/* Main Prediction */}
          <div className="mb-4 p-4 bg-white rounded-lg border">
            <div className="flex items-center justify-between">
              <span className="text-lg font-medium text-gray-700">Predicted Class:</span>
              <span className="text-2xl font-bold text-blue-600">{prediction.predicted_class}</span>
            </div>
            <div className="flex items-center justify-between mt-2">
              <span className="text-sm text-gray-600">Confidence:</span>
              <span className="text-lg font-semibold text-green-600">{prediction.confidence_percentage}</span>
            </div>
          </div>

          {/* Top Predictions */}
          {prediction.top_predictions && prediction.top_predictions.length > 1 && (
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Top Predictions:</h4>
              <div className="space-y-2">
                {prediction.top_predictions.map((pred, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="font-medium">{pred.class}</span>
                    <span className="text-sm text-gray-600">{pred.percentage}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4 text-xs text-gray-500">
            Model used: {prediction.model_used}
          </div>
        </div>
      )}

      {/* Model Info */}
      {selectedModel && availableModels.length > 0 && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-medium text-gray-700 mb-2">Selected Model Info:</h3>
          {(() => {
            const model = availableModels.find(m => m.name === selectedModel);
            return model ? (
              <div className="text-sm text-gray-600 grid grid-cols-2 gap-2">
                <div>Type: {model.model_type}</div>
                <div>Classes: {model.num_classes}</div>
                <div>Training Accuracy: {model.training_accuracy ? `${(model.training_accuracy * 100).toFixed(1)}%` : 'N/A'}</div>
                <div>Validation Accuracy: {model.validation_accuracy ? `${(model.validation_accuracy * 100).toFixed(1)}%` : 'N/A'}</div>
              </div>
            ) : null;
          })()}
        </div>
      )}
    </div>
  );
};

export default ImagePredictor;
