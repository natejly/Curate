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
    const eventSource = new EventSource(`http://localhost:8000/train-logs/${sessionId}`);
    eventSource.onmessage = (e) => {
      setConsoleLogs((prev) => prev + e.data);
    };
    eventSource.onerror = () => {
      eventSource.close();
      setTrainStatus("Training finished or failed.");
    };
    return () => eventSource.close();
  }, [sessionId]);

  return (
    <div className="flex min-h-screen bg-black">
      {/* Sidebar menu */}
      <div className="w-64 bg-gray-900 text-white flex flex-col p-6 border-r border-gray-800">
        <div className="text-2xl font-bold mb-8">Curate</div>
        <nav className="flex flex-col gap-4">
          <button className="text-left px-2 py-2 rounded hover:bg-gray-800" onClick={() => router.push("/")}>Upload</button>
          <button className="text-left px-2 py-2 rounded hover:bg-gray-800" disabled>Training</button>
        </nav>
      </div>
      {/* Main console */}
      <div className="flex-1 flex flex-col items-center justify-start p-12">
        <div className="w-full max-w-5xl">
          <div className="text-2xl font-bold text-white mb-6">Training Console</div>
          <div className="bg-black border border-white/20 rounded-lg p-6 text-green-200 font-mono text-sm h-[600px] w-full overflow-y-auto" style={{whiteSpace: 'pre-wrap'}}>
            {consoleLogs
              ? consoleLogs
              : (
                  <span className="text-yellow-300">
                    {trainStatus === "Training started..."
                      ? "Waiting for logs... If you selected an S3 dataset, logs may not be available until the job completes."
                      : "No logs available for this session."}
                  </span>
                )}
          </div>
          <div className="mt-6 p-2 rounded bg-black/30 border border-white/10 text-white text-center">
            {trainStatus}
          </div>
        </div>
      </div>
    </div>
  );
}
