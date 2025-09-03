"use client";
import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export const dynamic = "force-dynamic";

export default function TrainingConsole() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("sessionId");
  const [consoleLogs, setConsoleLogs] = useState("");
  const [trainStatus, setTrainStatus] = useState("Training started...");

  useEffect(() => {
    if (!sessionId) return;
    
    console.log("Starting log stream for session:", sessionId);
    const eventSource = new EventSource(`${process.env.BACKEND_URL || 'http://localhost:8000'}/train-logs/${sessionId}`);
    
    eventSource.onmessage = (e) => {
      console.log("Received log:", e.data);
      setConsoleLogs((prev) => prev + e.data + "\n");
    };
    
    eventSource.onerror = (error) => {
      console.error("EventSource error:", error);
      eventSource.close();
      setTrainStatus("Log streaming ended. Training may be finished or failed.");
    };
    
    eventSource.onopen = () => {
      console.log("Log stream connection opened");
      setConsoleLogs("Connected to training logs...\n");
    };
    
    return () => {
      console.log("Closing log stream");
      eventSource.close();
    };
  }, [sessionId]);

  return (
    <div className="flex min-h-screen bg-black">
      {/* Main console only, no sidebar */}
      <div className="flex-1 flex flex-col items-center justify-start p-12">
        <div className="w-full max-w-5xl">
          <div className="text-2xl font-bold text-white mb-6">Training Console</div>
          <div className="bg-black border border-white/20 rounded-lg p-6 text-green-200 font-mono text-sm h-[600px] w-full overflow-y-auto" style={{whiteSpace: 'pre-wrap'}}>
            {consoleLogs || "Starting Training Instance..."}
          </div>
          <div className="mt-6 p-2 rounded bg-black/30 border border-white/10 text-white text-center">
            {trainStatus}
          </div>
        </div>
      </div>
    </div>
  );
}
