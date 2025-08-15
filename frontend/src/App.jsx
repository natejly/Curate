import React, { useEffect, useState } from "react";
import FolderDropzone from "./components/FolderDropzone";
import MLChat from "./components/MLChat";
import ImagePredictor from "./components/ImagePredictor";

export default function App() {
  const [status, setStatus] = useState("loading");
  const [uploadData, setUploadData] = useState(null);
  const [showChat, setShowChat] = useState(false);
  const [activeTab, setActiveTab] = useState("upload");
  
  // Persistent chat state
  const [conversation, setConversation] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("error"));
  }, []);

  const handleUploadComplete = (data) => {
    setUploadData(data);
    setShowChat(true);
    setActiveTab("train");
  };

  const handleNewUpload = () => {
    setUploadData(null);
    setShowChat(false);
    setActiveTab("upload");
    // Clear chat history when starting new upload
    setConversation([]);
    setAnalysis(null);
  };

  const tabs = [
    { id: "upload", label: "Upload Dataset", icon: "" },
    { id: "train", label: "Train Model", icon: "" },
    { id: "predict", label: "Predict Image", icon: "" }
  ];

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      <div className="max-w-6xl mx-auto py-12 px-4">
        <div className="flex justify-between items-center mb-8 animate-fade-in">
          <div>
            <h1 className="flex items-center text-4xl font-semibold text-gray-900 tracking-tight">
              Curate
              <span className="ml-2 inline-block h-3 w-3 rounded-full bg-indigo-500"></span>
            </h1>
            <div className="flex items-center mt-2 text-gray-600">
              <span>AI-Powered Dataset Training</span>
              <span className="mx-2">•</span>
              <span className="flex items-center gap-2">
                Backend:
                <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                  status === 'healthy'
                    ? 'bg-green-100 text-green-800'
                    : status === 'loading'
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-red-100 text-red-800'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${
                    status === 'healthy'
                      ? 'bg-green-500 animate-pulse'
                      : status === 'loading'
                      ? 'bg-yellow-500 animate-pulse'
                      : 'bg-red-500'
                  }`}></div>
                  {status}
                </span>
              </span>
            </div>
          </div>

          {(showChat || activeTab !== "upload") && (
            <button
              onClick={handleNewUpload}
              className="btn-primary animate-fade-in"
            >
              New Upload
            </button>
          )}
        </div>

        {/* Tab Navigation */}
        <div className="mb-8 animate-slide-up">
          <div className="border-b border-gray-200 bg-white rounded-t-xl shadow-sm">
            <nav className="-mb-px flex space-x-8 px-6">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-2 border-b-2 font-medium text-sm transition-all duration-300 ${
                    activeTab === tab.id
                      ? "border-blue-500 text-blue-600 bg-blue-50 rounded-t-lg -mb-px"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 hover:bg-gray-50 rounded-t-lg"
                  } ${tab.id === "train" && !showChat ? "opacity-50 cursor-not-allowed" : ""}`}
                  disabled={tab.id === "train" && !showChat}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div className="mt-8 animate-fade-in">
          {activeTab === "upload" && (
            <div className="card-hover">
              <FolderDropzone onUploadComplete={handleUploadComplete} />
            </div>
          )}
          
          {activeTab === "train" && showChat && (
            <div className="animate-slide-up">
              <MLChat 
                uploadData={uploadData}
                conversation={conversation}
                setConversation={setConversation}
                analysis={analysis}
                setAnalysis={setAnalysis}
                onTrainingStart={() => console.log('Training started')}
                onAnalysisComplete={(analysis) => console.log('Analysis complete:', analysis)}
              />
            </div>
          )}
          
          {activeTab === "predict" && (
            <div className="animate-slide-up">
              <ImagePredictor />
            </div>
          )}
          
          {activeTab === "train" && !showChat && (
            <div className="text-center py-16 card animate-fade-in">
              <div className="text-6xl mb-4 opacity-50">Dataset</div>
              <div className="text-gray-500 text-lg mb-2">
                Please upload a dataset first to start training
              </div>
              <div className="text-gray-400 text-sm mb-6">
                Your AI model needs data to learn from
              </div>
              <button
                onClick={() => setActiveTab("upload")}
                className="btn-primary"
              >
                Go to Upload
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
