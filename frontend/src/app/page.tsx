"use client";
import React, { useRef, useState } from "react";
import { useRouter } from "next/navigation";

// Accepts only folder uploads
export default function Home() {
  const router = useRouter();
  const [consoleLogs, setConsoleLogs] = useState<string>("");
  const [showConsole, setShowConsole] = useState(false);
  const [testResults, setTestResults] = useState<any | null>(null);
  const [trainStatus, setTrainStatus] = useState<string>("");
  // Remove chatInput state
  // Removed LLM chat state
  const [lastSessionId, setLastSessionId] = useState<string | null>(null);
  const [selectedZip, setSelectedZip] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState<any | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.zip')) {
        setSelectedZip(file);
      } else {
        alert("Only .zip files are accepted.");
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.toLowerCase().endsWith('.zip')) {
        setSelectedZip(file);
      } else {
        alert("Only .zip files are accepted.");
      }
    }
  };

  const handleUpload = async () => {
    if (selectedZip) {
      setUploadProgress(0);
      setUploadComplete(false);
      const formData = new FormData();
      formData.append("file", selectedZip);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "http://localhost:8000/upload/zip", true);
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          setUploadProgress(Math.round((event.loaded / event.total) * 100));
        }
      };
      xhr.onload = async () => {
        setUploadProgress(100);
        setUploadComplete(true);
        // Parse session_id from response
        let sessionId = null;
        try {
          const res = JSON.parse(xhr.responseText);
          sessionId = res.session_id;
          setLastSessionId(sessionId);
        } catch {}
        if (sessionId) {
          // Fetch dataset info
          try {
            const infoRes = await fetch(`http://localhost:8000/dataset-info/${sessionId}`);
            if (infoRes.ok) {
              const info = await infoRes.json();
              setDatasetInfo(info);
              setErrorMsg(null);
            } else {
              const err = await infoRes.json();
              setDatasetInfo(null);
              setErrorMsg(err.detail || "Failed to process dataset.");
            }
          } catch {
            setDatasetInfo(null);
            setErrorMsg("Failed to process dataset.");
          }
        }
        setTimeout(() => {
          setSelectedZip(null);
          setUploadProgress(0);
          setUploadComplete(false);
        }, 1500);
      };
      xhr.onerror = () => {
        setUploadProgress(0);
        setUploadComplete(false);
      };
      xhr.send(formData);
    }
  };

  // Little rounded box for attached file
  const AttachedBox = ({ file }: { file: File }) => (
    <div className="mt-4 px-4 py-2 rounded-lg bg-white/10 border border-white/30 text-white text-sm flex items-center gap-2">
      <span className="font-medium">{file.name}</span>
      <span className="text-xs text-white/60">{(file.size / 1024).toFixed(1)} KB</span>
    </div>
  );

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-black">
      <div className="w-full flex justify-center pt-8 pb-2">
        <span className="text-3xl font-bold text-gray-100 tracking-tight">Curate.</span>
      </div>
      <div
        className={`w-full max-w-md mx-auto p-8 rounded-2xl shadow-lg flex flex-col items-center justify-center border border-white/30 bg-black/60 backdrop-blur-lg transition-all duration-300 ${dragActive ? "border-white" : "border-white/30"}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <div
          className={`w-full h-48 flex flex-col items-center justify-center border-2 border-dashed rounded-xl transition-colors duration-200 ${dragActive ? "border-white bg-white/10" : "border-white/30 bg-white/5"}`}
          onClick={() => inputRef.current?.click()}
          style={{ cursor: "pointer" }}
        >
          <input
            type="file"
            ref={inputRef}
            className="hidden"
            onChange={handleChange}
            accept=".zip"
          />
          <span className="text-lg text-gray-300 font-semibold mb-2">Upload Zipped Dataset</span>
          <span className="text-xs text-gray-400 mt-2">Only .zip files are accepted</span>
          {selectedZip && <AttachedBox file={selectedZip} />}
        </div>
        <button
          className="mt-8 px-6 py-2 rounded-lg bg-white text-black font-bold shadow hover:bg-white/80 transition-colors duration-200 disabled:opacity-50"
          onClick={handleUpload}
          disabled={!selectedZip || uploadProgress > 0}
        >
          Upload
        </button>
        {uploadProgress > 0 && (
          <div className="w-full mt-6">
            <div className="w-full bg-white/10 rounded-full h-3">
              <div
                className="bg-white h-3 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <div className="text-white text-xs mt-2 text-center">{uploadProgress}%</div>
          </div>
        )}
        {/* Flash upload complete message in green, not as a button */}
        <div
          className={`mt-6 px-4 py-2 rounded-lg bg-green-600 text-white text-center font-semibold transition-opacity duration-700 ${uploadComplete ? 'opacity-100' : 'opacity-0'}`}
        >
          Upload complete!
        </div>
      </div>
      {/* Show error message if dataset is invalid */}
      {errorMsg && (
        <div className="w-full max-w-md mt-8 p-4 rounded-xl bg-red-600 text-white text-center font-semibold">
          {errorMsg}
        </div>
      )}
      {/* Show dataset info after upload */}
      {datasetInfo && (
        <div className="w-full max-w-md mt-8 p-6 rounded-xl bg-white/10 border border-white/20 shadow text-white">
          <div className="text-lg font-bold mb-4">Uploaded Dataset Info</div>
          <div className="mb-2">
            <span className="font-semibold">Task:</span> {datasetInfo.task ?? "-"}
          </div>
          <div className="mb-2">
            <span className="font-semibold">Train Dir:</span> {datasetInfo.train_dir ? datasetInfo.train_dir.split('/').slice(-2).join('/') : "-"}
          </div>
          <div className="mb-2">
            <span className="font-semibold">Val Dir:</span> {datasetInfo.val_dir ? datasetInfo.val_dir.split('/').slice(-2).join('/') : "-"}
          </div>
          <div className="mb-2">
            <span className="font-semibold">Test Dir:</span> {datasetInfo.test_dir ? datasetInfo.test_dir.split('/').slice(-2).join('/') : "-"}
          </div>
          <div className="mb-2">
            <span className="font-semibold">Classes:</span> {datasetInfo.classes && datasetInfo.classes.length > 0 ? [...datasetInfo.classes].sort().join(", ") : "-"}
          </div>
          <div className="mb-2">
            <span className="font-semibold">Total Images:</span> {datasetInfo.total_images ?? "-"}
          </div>
          <div className="mb-2">
            <span className="font-semibold">Train Images:</span> {datasetInfo.train_images ?? "-"}
          </div>
          <div className="mb-2">
            <span className="font-semibold">Val Images:</span> {datasetInfo.val_images ?? "-"}
          </div>
          <div className="mb-2">
            <span className="font-semibold">Test Images:</span> {datasetInfo.test_images ?? "-"}
          </div>
          {/* Train button and status log */}
          <button
            className={`mt-6 px-6 py-2 rounded-lg font-bold shadow transition-colors duration-200 ${trainStatus === "Training started..." ? "bg-gray-400 text-gray-200 cursor-not-allowed" : "bg-blue-600 text-white hover:bg-blue-700"}`}
            onClick={async () => {
              if (trainStatus === "Training started...") return;
              setTrainStatus("Training started...");
              setShowConsole(true);
              setConsoleLogs("");
              setTestResults(null);
              const sessionId = datasetInfo.session_id;
              if (!sessionId) {
                setTrainStatus("No session ID found for training.");
                return;
              }
              // Navigate to training console page with sessionId
              router.push(`/training-console?sessionId=${sessionId}`);
            }}
            disabled={trainStatus === "Training started..."}
          >Train</button>
          {showConsole && (
            <div className="mt-6 w-full bg-black/80 border border-white/20 rounded-lg p-4 text-green-200 font-mono text-xs h-64 overflow-y-auto" style={{whiteSpace: 'pre-wrap'}}>
              {consoleLogs || "Waiting for logs..."}
            </div>
          )}
          {testResults && (
            <div className="mt-4 w-full bg-white/10 border border-white/20 rounded-lg p-4 text-white">
              <div className="font-bold mb-2">Test Results</div>
              <pre className="whitespace-pre-wrap text-xs">{JSON.stringify(testResults, null, 2)}</pre>
            </div>
          )}
          {trainStatus && (
            <div className="mt-4 p-2 rounded bg-black/30 border border-white/10 text-white text-center">
              {trainStatus}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
