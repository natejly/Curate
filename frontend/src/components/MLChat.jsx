import { useState } from 'react';

export default function MLChat({ uploadData, onTrainingStart, onAnalysisComplete }) {
  const [message, setMessage] = useState('');
  const [conversation, setConversation] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [analysis, setAnalysis] = useState(null);

  const handleSendMessage = async () => {
    if (!message.trim() || !uploadData?.upload_path) return;

    const userMessage = { type: 'user', content: message, timestamp: new Date() };
    setConversation(prev => [...prev, userMessage]);
    setIsAnalyzing(true);

    try {
      const response = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          upload_path: uploadData.upload_path
        })
      });

      const result = await response.json();

      if (result.success) {
        const aiMessage = {
          type: 'ai',
          content: result.analysis.ml_plan_text,
          analysis: result.analysis,
          timestamp: new Date()
        };
        setConversation(prev => [...prev, aiMessage]);
        setAnalysis(result.analysis);
        onAnalysisComplete?.(result.analysis);
      } else {
        throw new Error(result.error || 'Analysis failed');
      }
    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: `Error: ${error.message}`,
        timestamp: new Date()
      };
      setConversation(prev => [...prev, errorMessage]);
    } finally {
      setIsAnalyzing(false);
      setMessage('');
    }
  };

  const handleStartTraining = async () => {
    if (!analysis?.ml_plan || !uploadData?.upload_path) return;

    setIsTraining(true);
    onTrainingStart?.();

    try {
      const response = await fetch('/api/ml/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          upload_path: uploadData.upload_path,
          ml_plan: analysis.ml_plan
        })
      });

      const result = await response.json();

      const trainingMessage = {
        type: result.success ? 'success' : 'error',
        content: result.success 
          ? `Training completed! Model "${result.model_name}" achieved ${(result.metadata.training_accuracy * 100).toFixed(2)}% accuracy.`
          : `Training failed: ${result.error}`,
        result: result,
        timestamp: new Date()
      };
      setConversation(prev => [...prev, trainingMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: `Training error: ${error.message}`,
        timestamp: new Date()
      };
      setConversation(prev => [...prev, errorMessage]);
    } finally {
      setIsTraining(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getMessageStyle = (messageType) => {
    switch (messageType) {
      case 'user':
        return 'bg-blue-500 text-white ml-auto';
      case 'ai':
        return 'bg-gray-200 text-gray-800 mr-auto';
      case 'success':
        return 'bg-green-200 text-green-800 mr-auto';
      case 'error':
        return 'bg-red-200 text-red-800 mr-auto';
      default:
        return 'bg-gray-200 text-gray-800 mr-auto';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">AI Training Assistant</h2>
      
      {uploadData && (
        <div className="mb-4 p-3 bg-blue-50 rounded-lg">
          <p className="text-sm text-gray-600">
            Dataset loaded: <span className="font-medium">{uploadData.file_count} files</span>
          </p>
        </div>
      )}

      {/* Chat Messages */}
      <div className="h-96 overflow-y-auto mb-4 border border-gray-200 rounded-lg p-4 space-y-3">
        {conversation.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p>Tell me what you want to do with your uploaded dataset!</p>
            <p className="text-sm mt-2">Example: "Train an image classifier to recognize cats vs dogs"</p>
          </div>
        )}
        
        {conversation.map((msg, index) => (
          <div key={index} className="flex">
            <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${getMessageStyle(msg.type)}`}>
              <p className="text-sm">{msg.content}</p>
              <p className="text-xs opacity-70 mt-1">{formatTimestamp(msg.timestamp)}</p>
              
              {/* Show training button for AI responses with plans */}
              {msg.type === 'ai' && msg.analysis?.ml_plan && (
                <button
                  onClick={handleStartTraining}
                  disabled={isTraining}
                  className="mt-3 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                >
                  {isTraining ? 'Training...' : 'Start Training'}
                </button>
              )}
            </div>
          </div>
        ))}
        
        {isAnalyzing && (
          <div className="flex">
            <div className="bg-gray-200 text-gray-800 mr-auto max-w-xs lg:max-w-md px-4 py-2 rounded-lg">
              <p className="text-sm">Analyzing your dataset...</p>
              <div className="flex space-x-1 mt-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Message Input */}
      <div className="flex space-x-2">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
          placeholder="Describe what you want to do with your data..."
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isAnalyzing || !uploadData}
        />
        <button
          onClick={handleSendMessage}
          disabled={isAnalyzing || !message.trim() || !uploadData}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? 'Analyzing...' : 'Send'}
        </button>
      </div>

      {/* Analysis Summary */}
      {analysis && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-bold text-lg mb-2">Analysis Summary</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p><strong>Model Type:</strong> {analysis.ml_plan?.model_type}</p>
              <p><strong>Task:</strong> {analysis.ml_plan?.task_description}</p>
            </div>
            <div>
              <p><strong>Data Folders:</strong> {analysis.directory_analysis?.folders_with_data?.length || 0}</p>
              <p><strong>File Types:</strong> {analysis.directory_analysis?.file_type_summary ? Object.keys(analysis.directory_analysis.file_type_summary).join(', ') : 'Unknown'}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
