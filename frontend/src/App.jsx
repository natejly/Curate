import React, { useEffect, useState } from "react";
import FolderDropzone from "./components/FolderDropzone";
import MLChat from "./components/MLChat";
import ImagePredictor from "./components/ImagePredictor";

export default function App() {
  const [status, setStatus] = useState("loading");
  const [uploadData, setUploadData] = useState(null);
  const [showChat, setShowChat] = useState(false);
  const [activeTab, setActiveTab] = useState("upload");
  
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
  };

  const tabs = [
    { id: "upload", label: "📁 Upload Dataset", icon: "📁" },
    { id: "train", label: "🤖 Train Model", icon: "🤖" },
    { id: "predict", label: "🔮 Predict Image", icon: "🔮" }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto py-10 px-4">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">Curate</h1>
            <p className="mt-2 text-gray-600">AI-Powered Dataset Training • Backend health: {status}</p>
          </div>
          
          {(showChat || activeTab !== "upload") && (
            <button
              onClick={handleNewUpload}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              📁 New Upload
            </button>
          )}
        </div>

        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? "border-blue-500 text-blue-600"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
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
        <div className="mt-8">
          {activeTab === "upload" && (
            <FolderDropzone onUploadComplete={handleUploadComplete} />
          )}
          
          {activeTab === "train" && showChat && (
            <MLChat 
              uploadData={uploadData}
              onTrainingStart={() => console.log('Training started')}
              onAnalysisComplete={(analysis) => console.log('Analysis complete:', analysis)}
            />
          )}
          
          {activeTab === "predict" && (
            <ImagePredictor />
          )}
          
          {activeTab === "train" && !showChat && (
            <div className="text-center py-12">
              <div className="text-gray-500 text-lg">
                📁 Please upload a dataset first to start training
              </div>
              <button
                onClick={() => setActiveTab("upload")}
                className="mt-4 px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
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
