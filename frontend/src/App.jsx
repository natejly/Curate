import React, { useEffect, useState } from "react";
import FolderDropzone from "./components/FolderDropzone";

export default function App() {
  const [status, setStatus] = useState("loading");
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("error"));
  }, []);
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-3xl mx-auto py-10 px-4">
        <h1 className="text-2xl font-semibold">Curate</h1>
        <p className="mt-2 text-gray-600">Backend health: {status}</p>

        <div className="mt-8">
          <FolderDropzone />
        </div>
      </div>
    </div>
  );
}
