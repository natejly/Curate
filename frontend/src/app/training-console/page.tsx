"use client";
import React, { useEffect, useState, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import LiveMetrics from "./LiveMetrics";

export const dynamic = "force-dynamic";

function TrainingConsoleContent() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("sessionId");
  const [consoleLogs, setConsoleLogs] = useState("");
  const [trainStatus, setTrainStatus] = useState("Waiting for training to start...");
  const [metricsData, setMetricsData] = useState(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isTrainingComplete, setIsTrainingComplete] = useState(false);
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
              {consoleLogs}
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
