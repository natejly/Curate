"use client";
import React, { useEffect, useState, useRef, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import LiveMetrics from "./LiveMetrics";

export const dynamic = "force-dynamic";

function TrainingConsoleContent() {
  const router = useRouter(); // eslint-disable-line @typescript-eslint/no-unused-vars
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("sessionId");
  const [consoleLogs, setConsoleLogs] = useState("");
  const [trainStatus, setTrainStatus] = useState("Training started...");
  const [metricsData, setMetricsData] = useState(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isTrainingComplete, setIsTrainingComplete] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const consoleLogsRef = useRef("");


  useEffect(() => {
    if (!sessionId) return;

    console.log("Starting integrated log and metrics stream for session:", sessionId);
    const eventSource = new EventSource(`${process.env.BACKEND_URL || 'http://localhost:8000'}/train-logs/${sessionId}`);

    eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        console.log("Received data:", data);

        if (data.type === 'log') {
          // Handle log messages
          const newLogs = consoleLogsRef.current + data.message + "\n";
          consoleLogsRef.current = newLogs;
          setConsoleLogs(newLogs);

          // Check for training completion
          if (data.message.includes('TRAINING JOB COMPLETED') ||
              data.message.includes('Training completed') ||
              data.message.includes('Training completed for session')) {
            setIsTrainingComplete(true);
            setTrainStatus("Training completed successfully! 🎉");
          }
        } else if (data.type === 'metrics') {
          // Handle metrics data
          setMetricsData(data.data);
          console.log("Received metrics:", data.data);
        }
      } catch {
        // Fallback for non-JSON data (backward compatibility)
        console.log("Received raw log:", e.data);
        const newLogs = consoleLogsRef.current + e.data + "\n";
        consoleLogsRef.current = newLogs;
        setConsoleLogs(newLogs);
      }
    };

    eventSource.onerror = (error) => {
      console.error("EventSource error:", error);
      eventSource.close();

      // Check if training completed based on logs before connection closed
      const hasCompletionMessage = consoleLogsRef.current.includes('TRAINING JOB COMPLETED') ||
                                   consoleLogsRef.current.includes('Training completed');

      if (hasCompletionMessage) {
        setIsTrainingComplete(true);
        setTrainStatus("Training completed successfully! 🎉");
      } else {
        setTrainStatus("Log streaming ended. Training may be finished or failed.");
      }
    };

    eventSource.onopen = () => {
      console.log("Integrated stream connection opened");
      // Connection established - no need to show a message
    };

    return () => {
      console.log("Closing integrated stream");
      eventSource.close();
    };
  }, [sessionId]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (consoleRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = consoleRef.current;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10;

      // If user is at bottom, keep auto-scrolling
      if (isAtBottom || !isUserScrolling) {
        consoleRef.current.scrollTo({
          top: consoleRef.current.scrollHeight,
          behavior: 'smooth'
        });
        setIsUserScrolling(false);
        setShowScrollButton(false);
      }
    }
  }, [consoleLogs, isUserScrolling]);

  // Handle scroll events to detect manual scrolling
  const handleScroll = () => {
    if (consoleRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = consoleRef.current;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10; // 10px threshold
      setIsUserScrolling(!isAtBottom);
      setShowScrollButton(!isAtBottom);
    }
  };

  // Function to manually scroll to bottom
  const scrollToBottom = () => {
    if (consoleRef.current) {
      consoleRef.current.scrollTo({
        top: consoleRef.current.scrollHeight,
        behavior: 'smooth'
      });
      setIsUserScrolling(false);
      setShowScrollButton(false);
    }
  };

  // Function to export model from S3
  const exportModel = async (format: 'onnx' | 'tf') => {
    if (!sessionId) return;

    setIsExporting(true);
    try {
      const response = await fetch(
        `${process.env.BACKEND_URL || 'http://localhost:8000'}/download-model/${sessionId}?format=${format}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        alert(`Export failed: ${data.error}`);
        return;
      }

      // Trigger download using the presigned URL
      const link = document.createElement('a');
      link.href = data.download_url;
      link.download = data.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      console.log(`Model export initiated: ${data.filename}`);
    } catch (error) {
      console.error('Export error:', error);
      alert('Failed to export model. Please check the console for details.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-black">
      {/* Main content area */}
      <div className="flex-1 flex flex-col items-center justify-start p-12">
        <div className="w-full max-w-7xl">
          <div className="text-2xl font-bold text-white mb-6">Training Console</div>

          {/* Console Logs Section */}
          <div className="bg-black border border-white/20 rounded-lg p-6 mb-6 relative">
            <h3 className="text-lg font-semibold text-white mb-4">Training Logs</h3>
            <div
              ref={consoleRef}
              onScroll={handleScroll}
              className="text-green-200 font-mono text-sm h-[400px] w-full overflow-y-auto bg-gray-900 p-4 rounded relative"
              style={{whiteSpace: 'pre-wrap'}}
            >
              {consoleLogs || "Starting Training Instance..."}
            </div>

            {/* Scroll to Bottom Button */}
            {showScrollButton && (
              <button
                onClick={scrollToBottom}
                className="absolute bottom-20 right-6 bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 shadow-lg flex items-center gap-2"
                title="Scroll to bottom"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
                Latest
              </button>
            )}

            <div className="mt-4 p-3 rounded bg-black/30 border border-white/10 text-white text-center">
              {trainStatus}
            </div>
          </div>

          {/* Live Metrics Section */}
          <div className="mb-6">
            {sessionId ? (
              <LiveMetrics sessionId={sessionId} metricsData={metricsData} />
            ) : (
              <div className="bg-black border border-white/20 rounded-lg p-6">
                <div className="text-center text-gray-400">
                  <p>No session ID provided. Please start a training session first.</p>
                </div>
              </div>
            )}
          </div>

          {/* Model Export Section */}
          {isTrainingComplete && sessionId && (
            <div className="bg-black border border-white/20 rounded-lg p-6 mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Export Trained Model</h3>
              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={() => exportModel('onnx')}
                  disabled={isExporting}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  {isExporting ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Exporting...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Export ONNX Model
                    </>
                  )}
                </button>

                <button
                  onClick={() => exportModel('tf')}
                  disabled={isExporting}
                  className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-green-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  {isExporting ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Exporting...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Export TensorFlow Model
                    </>
                  )}
                </button>
              </div>

              <div className="mt-4 p-3 rounded bg-black/30 border border-white/10">
                <p className="text-sm text-gray-300">
                  💡 <strong>Tip:</strong> Models are downloaded from cloud storage. The download link expires after 1 hour for security.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function TrainingConsole() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen bg-black">
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="text-white text-center">
            <div className="text-xl mb-4">Loading Training Console...</div>
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
          </div>
        </div>
      </div>
    }>
      <TrainingConsoleContent />
    </Suspense>
  );
}
