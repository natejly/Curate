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
    <div>
      <div
        className={`border-2 border-dashed rounded-md p-8 text-center transition-colors ${
          dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white"
        } ${loading ? "opacity-50" : ""}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        <p className="text-gray-600 mb-4">
          {loading ? "Processing..." : "Drag & drop a folder here or click to browse files"}
        </p>
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
          className={`inline-block px-4 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 ${
            loading ? "cursor-not-allowed" : ""
          }`}
        >
          {loading ? "Processing..." : "Choose Folder"}
        </label>
      </div>

      {error && <p className="mt-4 text-sm text-red-600">{error}</p>}

      {uploadInfo && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded">
          <h3 className="text-sm font-medium text-green-800">Upload Successful!</h3>
          <p className="text-sm text-green-700 mt-1">
            {uploadInfo.file_count} files saved to: <code className="bg-green-100 px-1 rounded text-xs">{uploadInfo.upload_path}</code>
          </p>
        </div>
      )}

      {hasData && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-4">Folder Structure & File Types</h3>
          <div className="max-h-96 overflow-auto border rounded p-4 bg-gray-50">
            <div>
              <span className="font-medium text-lg">{tree.name || "Root"}</span>
              {tree.total_files > 0 && (
                <span className="text-gray-500"> ({tree.total_files} total files)</span>
              )}
            </div>
            {renderTree(tree)}
          </div>
          
          {/* Summary of all file types */}
          {tree.file_counts && Object.keys(tree.file_counts).length > 0 && (
            <div className="mt-4 p-4 bg-blue-50 rounded">
              <h4 className="font-medium mb-2">Overall File Type Summary:</h4>
              <div className="flex flex-wrap gap-3">
                {Object.entries(tree.file_counts)
                  .sort((a, b) => b[1] - a[1])
                  .map(([ext, count]) => (
                    <span key={ext} className="bg-white px-2 py-1 rounded text-sm">
                      .{ext}: <strong>{count}</strong>
                    </span>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
