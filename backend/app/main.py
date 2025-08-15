from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

app = FastAPI(title="Curate Backend (skeleton)")

# Allow dev Vite server origin. In production, restrict this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")


def parse_directory_recursive(base_path: Path) -> Dict[str, Any]:
    """Recursively parse directory and return tree with file type counts per terminal folder."""
    tree = {}
    
    def build_tree(current_path: Path, relative_path: str = "") -> Dict[str, Any]:
        node = {
            "name": current_path.name,
            "type": "directory",
            "children": {},
            "file_counts": defaultdict(int),
            "total_files": 0
        }
        
        try:
            items = list(current_path.iterdir())
        except (PermissionError, OSError):
            return node
            
        # Separate files and directories
        files = [item for item in items if item.is_file()]
        directories = [item for item in items if item.is_dir()]
        
        # Process files in this directory
        for file_path in files:
            ext = file_path.suffix.lower().lstrip('.') or 'no-extension'
            node["file_counts"][ext] += 1
            node["total_files"] += 1
        
        # Process subdirectories
        for dir_path in directories:
            child_relative = f"{relative_path}/{dir_path.name}" if relative_path else dir_path.name
            child_node = build_tree(dir_path, child_relative)
            node["children"][dir_path.name] = child_node
            
            # Bubble up file counts to parent if this is not a terminal directory
            if not child_node["children"]:  # Terminal directory
                # Don't bubble up from terminal directories
                pass
            else:
                # This directory has subdirectories, so bubble up counts
                for ext, count in child_node["file_counts"].items():
                    node["file_counts"][ext] += count
                node["total_files"] += child_node["total_files"]
        
        return node
    
    return build_tree(base_path)


def extract_files_from_uploads(upload_files: List[UploadFile]) -> Path:
    """Save uploaded files to temporary directory, preserving folder structure."""
    temp_dir = Path(tempfile.mkdtemp())
    
    for upload_file in upload_files:
        if not upload_file.filename:
            continue
            
        # Preserve the folder structure from the filename/path
        file_path = temp_dir / upload_file.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(upload_file.file, f)
    
    return temp_dir


@router.post("/test-upload")
async def test_upload(files: List[UploadFile] = File(...)):
    """Simple test endpoint to debug file uploads."""
    return {
        "files_count": len(files) if files else 0,
        "files_info": [{"name": f.filename, "content_type": f.content_type} for f in files] if files else []
    }


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and parse files/folders, returning directory tree with file type analysis."""
    print(f"Received {len(files) if files else 0} files")
    for i, file in enumerate(files):
        print(f"File {i}: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    temp_path = None
    try:
        # Save files to temporary directory
        temp_path = extract_files_from_uploads(files)
        
        # Parse the directory structure
        tree = parse_directory_recursive(temp_path)
        
        return {
            "success": True,
            "file_count": len(files),
            "tree": tree
        }
        
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if temp_path and temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


@router.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(router)
