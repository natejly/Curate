"use client";
import React, { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AvailableDatasetsSection } from "./AvailableDatasetsSection";

interface DatasetInfo {
  session_id: string;
  train_dir?: string;
  val_dir?: string;
  test_dir?: string;
  classes?: string[];
  total_images?: number;
  train_images?: number;
  val_images?: number;
  test_images?: number;
  train_images_per_class?: Record<string, number>;
  val_images_per_class?: Record<string, number>;
  test_images_per_class?: Record<string, number>;
  task?: string;
  processing_status?: string;
  upload_status?: string;
  message?: string;
  error?: string;
}

// Accepts only folder uploads
export default function Home() {
  const router = useRouter();

  // Cleanup polling on unmount
  React.useEffect(() => {
    return () => {
      if (statusPollingRef.current) {
        clearInterval(statusPollingRef.current);
      }
    };
  }, []);
  const [consoleLogs, setConsoleLogs] = useState<string>("");
  const [showConsole, setShowConsole] = useState(false);
  const [testResults, setTestResults] = useState<Record<string, unknown> | null>(null);
  const [trainStatus, setTrainStatus] = useState<string>("");
  // Remove chatInput state
  // Removed LLM chat state
  const [selectedZip, setSelectedZip] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<string>("");
  const statusPollingRef = useRef<NodeJS.Timeout | null>(null);
  const [isUploaded, setIsUploaded] = useState(false);

  const startStatusPolling = (sessionId: string) => {
    setIsProcessing(true);
    setProcessingStatus("File uploaded successfully. Starting processing...");

    const pollStatus = async () => {
      try {
        const response = await fetch(`${process.env.BACKEND_URL || 'http://localhost:8000'}/upload-status/${sessionId}`);
        if (response.ok) {
          const status = await response.json();
          setDatasetInfo(status);

          // Update status message based on upload and processing status
          let statusMessage = "";
          if (status.upload_status === "received") {
            statusMessage = "File uploaded. Starting extraction...";
          } else if (status.upload_status === "extracting") {
            statusMessage = "Extracting zip file...";
          } else if (status.upload_status === "extracted") {
            statusMessage = "Zip extracted. Starting dataset analysis...";
          } else if (status.upload_status === "processing_started") {
            statusMessage = "Dataset analysis in progress...";
          } else if (status.upload_status === "failed") {
            statusMessage = status.message || "Upload failed";
          } else if (status.processing_status === "completed") {
            statusMessage = "Dataset processing completed!";
          } else if (status.processing_status === "failed") {
            statusMessage = status.error || "Dataset processing failed";
          } else {
            statusMessage = status.message || "Processing...";
          }

          setProcessingStatus(statusMessage);

          // Check if everything is complete or failed
          const isComplete = status.processing_status === "completed";
          const isFailed = status.upload_status === "failed" || status.processing_status === "failed";

          if (isComplete || isFailed) {
            setIsProcessing(false);
            if (isFailed) {
              setErrorMsg(statusMessage);
            } else {
              setErrorMsg(null);
            }
            if (statusPollingRef.current) {
              clearInterval(statusPollingRef.current);
              statusPollingRef.current = null;
            }
          }
        } else {
          setErrorMsg("Failed to check processing status");
          setIsProcessing(false);
        }
      } catch {
        setErrorMsg("Failed to check processing status");
        setIsProcessing(false);
      }
    };

    // Poll immediately, then every 2 seconds
    pollStatus();
    statusPollingRef.current = setInterval(pollStatus, 2000);
  };


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
      xhr.open("POST", `${process.env.BACKEND_URL || 'http://localhost:8000'}/upload/zip`, true);
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
          console.log("Upload response:", res);
        } catch (e) {
          console.error("Failed to parse upload response:", e);
        }

        if (sessionId) {
          // Start polling for dataset processing status
          startStatusPolling(sessionId);
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
    <div className="min-h-screen bg-black flex flex-col pt-20">
      <div className="w-full max-w-xl mx-auto p-8 pt-10 rounded-2xl shadow-lg flex flex-col items-center justify-center border border-white/30 bg-black/60 backdrop-blur-lg transition-all duration-300">
        <div
          className={`w-full h-48 flex flex-col items-center justify-center border-2 border-dashed rounded-xl transition-colors duration-200 pt-8 ${dragActive ? "border-white bg-white/10" : "border-white/30 bg-white/5"}`}
          onClick={() => inputRef.current?.click()}
          style={{ cursor: "pointer" }}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
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
        {errorMsg && (
          <div className="w-full mt-8 p-4 rounded-xl bg-red-600 text-white text-center font-semibold">
            {errorMsg}
          </div>
        )}
        {datasetInfo && (
          <div className="w-full mt-8 p-6 rounded-xl bg-white/10 border border-white/20 shadow text-white">
            <div className="text-lg font-bold mb-4">
              Uploaded Dataset Info
            </div>

            {datasetInfo.processing_status === "completed" ? (
              <>
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
              </>
            ) : datasetInfo.upload_status === "failed" ? (
              <div className="p-4 rounded-lg bg-red-600/20 border border-red-400/30 text-red-200">
                <div className="font-semibold mb-2">Upload Failed</div>
                <div>{datasetInfo.message || "An unknown error occurred during upload."}</div>
              </div>
            ) : datasetInfo.processing_status === "failed" ? (
              <div className="p-4 rounded-lg bg-red-600/20 border border-red-400/30 text-red-200">
                <div className="font-semibold mb-2">Processing Failed</div>
                <div>{datasetInfo.error || "An unknown error occurred during processing."}</div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <div className="mb-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-2 border-blue-400 border-t-transparent mx-auto"></div>
                </div>
                <div className="text-sm">
                  {datasetInfo.upload_status === "extracting" && "Extracting zip file..."}
                  {datasetInfo.upload_status === "processing_started" && "Analyzing dataset..."}
                  {datasetInfo.processing_status === "processing" && "Processing dataset..."}
                  {!datasetInfo.upload_status && !datasetInfo.processing_status && "Initializing..."}
                </div>
              </div>
            )}
            {datasetInfo.processing_status === "completed" && !isUploaded && (
              <div className="flex justify-center">
                <button
                  className={`mt-6 px-8 py-3 rounded-lg font-bold shadow transition-colors duration-200 ${trainStatus === "Uploading to cloud..." ? "bg-gray-400 text-gray-200 cursor-not-allowed" : "bg-blue-600 text-white hover:bg-blue-700"}`}
                  onClick={async () => {
                  if (trainStatus === "Uploading to cloud...") return;
                  setTrainStatus("Uploading to cloud...");
                  const sessionId = datasetInfo.session_id;
                  if (!sessionId) {
                    setTrainStatus("No session ID found.");
                    return;
                  }

                  // Upload dataset to S3 only (no training)
                  try {
                    const uploadResponse = await fetch(`${process.env.BACKEND_URL || 'http://localhost:8000'}/upload-to-s3/${sessionId}`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" }
                    });

                    if (!uploadResponse.ok) {
                      const error = await uploadResponse.json();
                      setTrainStatus(`Upload failed: ${error.detail || "Unknown error"}`);
                      return;
                    }

                    const uploadResult = await uploadResponse.json();
                    console.log("Upload to S3 successful:", uploadResult);
                    setTrainStatus("Dataset uploaded to cloud computer successfully!");
                    setIsUploaded(true);
                  } catch (err) {
                    setTrainStatus("Failed to upload to cloud. Check connection.");
                    console.error("Upload error:", err);
                  }
                }}
                disabled={trainStatus === "Uploading to cloud..."}
              >
                {trainStatus === "Uploading to cloud..." ? "Uploading..." : "Upload to Cloud Computer"}
              </button>
              </div>
            )}
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
        <AvailableDatasetsSection />
      </div>
    </div>
  );
}
