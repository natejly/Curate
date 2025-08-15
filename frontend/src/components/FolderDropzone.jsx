import React, { useCallback, useState } from "react";

// Helper: Recursively read entries in a DataTransferItemList
async function readDirectoryEntries(entry) {
  const files = [];

  async function walkDirectory(dirEntry, pathPrefix = "") {
    const reader = dirEntry.createReader();
    async function readBatch() {
      return new Promise((resolve, reject) => {
        reader.readEntries((entries) => resolve(entries), reject);
      });
    }
    while (true) {
      const entries = await readBatch();
      if (!entries.length) break;
      for (const e of entries) {
        if (e.isFile) {
          const file = await new Promise((resolve, reject) => e.file(resolve, reject));
          // Create a new File with the full path as the name
          const fileWithPath = new File([file], `${pathPrefix}${file.name}`, {
            type: file.type,
            lastModified: file.lastModified
          });
          files.push(fileWithPath);
        } else if (e.isDirectory) {
          await walkDirectory(e, `${pathPrefix}${e.name}/`);
        }
      }
    }
  }

  if (entry.isDirectory) {
    await walkDirectory(entry, "");
  } else if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    files.push(file);
  }
  return files;
}

// Helper function to render the tree structure from server response
function renderTree(node, basePath = "") {
  if (!node.children || Object.keys(node.children).length === 0) {
    // Terminal directory - show file type counts
    return (
      <div className="ml-4">
        {node.total_files > 0 && (
          <div className="text-sm text-gray-600 mt-1">
            {Object.entries(node.file_counts).map(([ext, count]) => (
              <span key={ext} className="mr-3">
                .{ext}: {count}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Non-terminal directory - show children
  const children = Object.entries(node.children);
  return (
    <ul className="ml-4 list-disc">
      {children.map(([folderName, childNode]) => (
        <li key={basePath + folderName}>
          <span className="font-medium">{folderName}</span>
          {childNode.total_files > 0 && (
            <span className="text-gray-500"> ({childNode.total_files} files)</span>
          )}
          {renderTree(childNode, `${basePath}${folderName}/`)}
        </li>
      ))}
    </ul>
  );
}

export default function FolderDropzone({ onUploadComplete }) {
  const [tree, setTree] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadInfo, setUploadInfo] = useState(null);

  const handleFilesUpload = useCallback(async (files) => {
    if (!files || files.length === 0) return;

    console.log('Starting upload with', files.length, 'files');
    console.log('Files:', files);

    setLoading(true);
    setError("");
    setUploadInfo(null);

    const formData = new FormData();
    files.forEach(file => {
      console.log('Adding file:', file.name, 'size:', file.size);
      formData.append('files', file);
    });

    // Log FormData entries
    for (let pair of formData.entries()) {
      console.log('FormData entry:', pair[0], pair[1]);
    }

    try {
      console.log('Sending request to /api/upload');
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status, response.statusText);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        let errorText;
        try {
          // Try to parse as JSON first
          const errorJson = await response.json();
          errorText = errorJson.detail || errorJson.message || JSON.stringify(errorJson);
        } catch {
          // Fall back to text if not JSON
          errorText = await response.text();
        }
        console.log('Error response:', errorText);
        throw new Error(`Upload failed (${response.status}): ${errorText}`);
      }

      const result = await response.json();
      console.log('Upload successful:', result);
      setTree(result.tree);
      
      // Store upload info for display
      const uploadData = {
        upload_path: result.upload_path,
        file_count: result.file_count,
        tree: result.tree
      };
      setUploadInfo(uploadData);
      
      // Call the callback to notify parent component
      if (onUploadComplete) {
        onUploadComplete(uploadData);
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(`Failed to upload and parse files: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    try {
      const items = e.dataTransfer.items;
      if (!items || items.length === 0) return;

      let allFiles = [];
      for (const item of items) {
        const entry = item.webkitGetAsEntry?.();
        if (entry) {
          const filesFromEntry = await readDirectoryEntries(entry);
          allFiles = allFiles.concat(filesFromEntry);
        } else {
          const file = item.getAsFile?.();
          if (file) allFiles.push(file);
        }
      }
      
      if (allFiles.length > 0) {
        await handleFilesUpload(allFiles);
      }
    } catch (err) {
      console.error(err);
      setError("Failed to read dropped folder. Your browser may restrict folder access.");
    }
  }, [handleFilesUpload]);

  const onFileChange = useCallback(async (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      await handleFilesUpload(files);
    }
  }, [handleFilesUpload]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const hasData = tree !== null;

  return (
    <div className="space-y-6">
      <div
        className={`dropzone ${dragActive ? 'dropzone-active' : ''} ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        <div className="flex flex-col items-center">
          <div className={`text-6xl mb-4 transition-all duration-300 ${dragActive ? 'scale-110' : ''}`}>
            {loading ? 'Processing' : dragActive ? 'Drop Here' : 'Upload'}
          </div>
          <p className="text-gray-600 mb-4 text-lg font-medium text-center">
            {loading ? "Processing your files..." : dragActive ? "Drop your folder here!" : "Drag & drop a folder here or click to browse"}
          </p>
          {!loading && (
            <div className="flex flex-col sm:flex-row gap-3">
              <input
                type="file"
                multiple
                webkitdirectory=""
                onChange={onFileChange}
                className="hidden"
                id="folder-upload"
                disabled={loading}
              />
              <label
                htmlFor="folder-upload"
                className="btn-primary cursor-pointer"
              >
                Choose Folder
              </label>
              <div className="text-center text-sm text-gray-500 px-4 py-2">
                <p>Supports images, documents, and more</p>
              </div>
            </div>
          )}
          {loading && (
            <div className="flex items-center space-x-3 mt-4">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
              <span className="text-sm text-gray-600">Analyzing structure...</span>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-gradient-to-r from-red-50 to-red-100 border border-red-200 rounded-lg animate-fade-in">
          <div className="flex items-center">
            <div>
              <h3 className="font-medium text-red-800">Upload Error</h3>
              <p className="text-sm text-red-600 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {uploadInfo && (
        <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg shadow-lg animate-slide-up">
          <div className="flex items-center mb-3">
            <div>
              <h3 className="text-lg font-bold text-green-800">Upload Successful!</h3>
              <p className="text-sm text-green-700 mt-1">
                <span className="font-medium">{uploadInfo.file_count}</span> files processed and saved
              </p>
            </div>
          </div>
          <div className="mt-3 p-3 bg-white bg-opacity-50 rounded-lg">
            <p className="text-xs text-green-600 font-mono">
              {uploadInfo.upload_path}
            </p>
          </div>
        </div>
      )}

      {hasData && (
        <div className="animate-fade-in space-y-6">
          <div className="card">
            <div className="flex items-center mb-4">
              <h3 className="text-xl font-bold text-gray-800">Dataset Analysis</h3>
            </div>
            
            <div className="max-h-96 overflow-auto border rounded-lg p-4 bg-gradient-to-b from-gray-50 to-white custom-scrollbar">
              <div className="flex items-center mb-3">
                <span className="font-bold text-lg text-gray-700">{tree.name || "Root"}</span>
                {tree.total_files > 0 && (
                  <span className="ml-2 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                    {tree.total_files} files
                  </span>
                )}
              </div>
              {renderTree(tree)}
            </div>
          </div>
          
          {/* Summary of all file types */}
          {tree.file_counts && Object.keys(tree.file_counts).length > 0 && (
            <div className="card">
              <div className="flex items-center mb-4">
                <h4 className="font-bold text-lg text-gray-800">File Type Summary</h4>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {Object.entries(tree.file_counts)
                  .sort((a, b) => b[1] - a[1])
                  .map(([ext, count]) => (
                    <div key={ext} className="bg-gradient-to-br from-blue-50 to-indigo-100 border border-blue-200 px-4 py-3 rounded-lg text-center hover:shadow-md transition-all duration-200">
                      <div className="text-lg font-bold text-blue-700">.{ext}</div>
                      <div className="text-sm text-blue-600">{count} files</div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
