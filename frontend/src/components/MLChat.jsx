import { useState } from 'react';

export default function MLChat({ 
  uploadData, 
  conversation, 
  setConversation, 
  analysis, 
  setAnalysis, 
  onTrainingStart, 
  onAnalysisComplete 
}) {
  const [message, setMessage] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

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
        return 'message-user';
      case 'ai':
        return 'message-ai';
      case 'success':
        return 'message-success';
      case 'error':
        return 'message-error';
      default:
        return 'message-ai';
    }
  };

  return (
    <div className="card-hover max-w-5xl mx-auto">
      <div className="flex items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">AI Training Assistant</h2>
      </div>
      
      {uploadData && (
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200 animate-fade-in">
          <div className="flex items-center">
            <div>
              <p className="font-medium text-gray-700">Dataset Ready</p>
              <p className="text-sm text-gray-600">
                <span className="font-medium text-blue-600">{uploadData.file_count}</span> files loaded successfully
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <div className="h-96 overflow-y-auto mb-6 border border-gray-200 rounded-xl p-6 space-y-4 bg-gradient-to-b from-gray-50 to-white custom-scrollbar">
        {conversation.length === 0 && (
          <div className="text-center text-gray-500 mt-12 animate-fade-in">
            <div className="text-6xl mb-4 opacity-50">Chat</div>
            <p className="text-lg font-medium mb-2">Tell me what you want to do with your dataset!</p>
            <p className="text-sm text-gray-400">
              Example: "Train an image classifier to recognize cats vs dogs"
            </p>
          </div>
        )}
        
        {conversation.map((msg, index) => (
          <div key={index} className="flex animate-fade-in">
            <div className={`${getMessageStyle(msg.type)} transform transition-all duration-300 hover:scale-105`}>
              <p className="text-sm leading-relaxed">{msg.content}</p>
              <p className="text-xs opacity-70 mt-2 flex items-center">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                </svg>
                {formatTimestamp(msg.timestamp)}
              </p>
              
              {/* Show training button for AI responses with plans */}
              {msg.type === 'ai' && msg.analysis?.ml_plan && (
                <button
                  onClick={handleStartTraining}
                  disabled={isTraining}
                  className={`mt-4 ${isTraining ? 'btn-secondary' : 'btn-success'} text-sm`}
                >
                  {isTraining ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Training...
                    </span>
                  ) : (
                    'Start Training'
                  )}
                </button>
              )}
            </div>
          </div>
        ))}
        
        {isAnalyzing && (
          <div className="flex animate-fade-in">
            <div className="message-ai">
              <p className="text-sm">Analyzing your dataset</p>
              <div className="flex space-x-1 mt-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Message Input */}
      <div className="flex space-x-3 mb-6">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
          placeholder="Describe what you want to do with your data..."
          className="input-modern flex-1"
          disabled={isAnalyzing || !uploadData}
        />
        <button
          onClick={handleSendMessage}
          disabled={isAnalyzing || !message.trim() || !uploadData}
          className="btn-primary"
        >
          {isAnalyzing ? (
            <span className="flex items-center">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing
            </span>
          ) : (
            'Send'
          )}
        </button>
      </div>

      {/* Analysis Summary */}
      {analysis && (
        <div className="p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-xl border border-green-200 animate-slide-up">
          <div className="flex items-center mb-4">
            <h3 className="font-bold text-lg text-gray-800">Analysis Summary</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
            <div className="space-y-3">
              <div className="flex items-center">
                <span className="font-medium text-gray-600 w-24">Model:</span>
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                  {analysis.ml_plan?.model_type}
                </span>
              </div>
              <div className="flex items-start">
                <span className="font-medium text-gray-600 w-24 flex-shrink-0">Task:</span>
                <span className="text-gray-700 leading-relaxed">{analysis.ml_plan?.task_description}</span>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-center">
                <span className="font-medium text-gray-600 w-24">Folders:</span>
                <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                  {analysis.directory_analysis?.folders_with_data?.length || 0} data folders
                </span>
              </div>
              <div className="flex items-start">
                <span className="font-medium text-gray-600 w-24 flex-shrink-0">Files:</span>
                <span className="text-gray-700">
                  {analysis.directory_analysis?.file_type_summary ? 
                    Object.keys(analysis.directory_analysis.file_type_summary).join(', ') : 
                    'Unknown types'
                  }
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
