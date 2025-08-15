# Curate — File Structure Analyzer

This workspace contains a minimal application for analyzing folder structures:

- FastAPI backend (backend/) - handles file uploads and recursive parsing
- React + Vite frontend with Tailwind CSS (frontend/) - drag & drop folder interface

## Features

- Drag & drop folder upload or browse to select folders
- Python-based recursive parsing of folder structures  
- File type analysis per terminal directory
- Visual folder tree with file counts
- Overall file type summary

Quick start (macOS, zsh):

Backend

```bash
cd backend
source .venv/bin/activate && python -m uvicorn app.main:app --reload --port 8000
```

Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to the backend on port 8000 in `vite.config.js`.

## Usage

1. Start both servers
2. Open the frontend in your browser
3. Drag & drop a folder or click "Choose Folder" to browse
4. View the parsed folder structure and file type analysis

Next steps: add database, auth, Dockerfiles, or CI as needed.
