from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time

# Import our services
from .ai_service import AIAnalysisService
from .ml_service import ModelTrainingService

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    upload_path: str

class TrainingRequest(BaseModel):
    upload_path: str
    ml_plan: Dict[str, Any]

app = FastAPI(title="Curate Backend (skeleton)")

# Configure max file size (100MB)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    print(f"Incoming request: {request.method} {request.url}")
    
    # Don't log all headers for upload requests to avoid spam
    if request.url.path != "/api/upload":
        print(f"Headers: {dict(request.headers)}")
    else:
        print(f"Content-Length: {request.headers.get('content-length', 'unknown')}")
        print(f"Content-Type: {request.headers.get('content-type', 'unknown')}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        print(f"Request completed in {process_time:.4f}s with status {response.status_code}")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        print(f"Request failed after {process_time:.4f}s with error: {e}")
        raise

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


def extract_files_from_uploads(upload_files) -> Path:
    """Save uploaded files to persistent directory, preserving folder structure."""
    # Create uploads directory if it doesn't exist
    uploads_dir = Path(__file__).parent.parent / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    
    # Create a unique subdirectory for this upload
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir = uploads_dir / f"upload_{timestamp}"
    upload_dir.mkdir(exist_ok=True)
    
    print(f"Saving files to persistent directory: {upload_dir}")
    
    for i, upload_file in enumerate(upload_files):
        if not upload_file.filename:
            print(f"Skipping file {i} with no filename")
            continue
            
        # Read file content
        try:
            file_content = upload_file.file.read()
            file_size = len(file_content)
            
            # Skip very large files (>50MB per file)
            if file_size > 50 * 1024 * 1024:  # 50MB
                print(f"Skipping large file: {upload_file.filename} ({file_size} bytes)")
                continue
                
            print(f"Processing file {i}: {upload_file.filename} ({file_size} bytes)")
        except Exception as e:
            print(f"Error reading file {upload_file.filename}: {e}")
            continue
        
        # Preserve the folder structure from the filename/path
        file_path = upload_dir / upload_file.filename
        
        # Ensure parent directories exist
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {file_path.parent}: {e}")
            continue
        
        # Write file content
        try:
            with open(file_path, 'wb') as f:
                f.write(file_content)
            print(f"Successfully wrote file: {file_path}")
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")
            continue
    
    return upload_dir


@router.post("/debug-upload")
async def debug_upload(files: List[UploadFile] = File(...)):
    """Debug endpoint to see exactly what files are being uploaded."""
    print(f"\n=== DEBUG UPLOAD ===")
    print(f"Number of files: {len(files) if files else 0}")
    
    if not files:
        return {"error": "No files received", "files": []}
    
    file_info = []
    for i, file in enumerate(files):
        info = {
            "index": i,
            "filename": file.filename,
            "content_type": file.content_type,
            "headers": dict(file.headers) if hasattr(file, 'headers') else {},
        }
        
        try:
            # Try to read first 50 bytes to check if file is readable
            content_sample = await file.read(50)
            await file.seek(0)  # Reset file pointer
            info["readable"] = True
            info["sample_size"] = len(content_sample)
            info["sample_content"] = content_sample.decode('utf-8', errors='ignore')[:50]
        except Exception as e:
            info["readable"] = False
            info["error"] = str(e)
        
        file_info.append(info)
        print(f"File {i}: {info}")
    
    return {"files": file_info}


@router.post("/test-upload")
async def test_upload(files: List[UploadFile] = File(...)):
    """Simple test endpoint to debug file uploads."""
    return {
        "files_count": len(files) if files else 0,
        "files_info": [{"name": f.filename, "content_type": f.content_type} for f in files] if files else []
    }


@router.post("/upload")
async def upload_files(request: Request):
    """Upload and parse files/folders, returning directory tree with file type analysis."""
    print(f"\n=== UPLOAD START ===")
    
    try:
        # Parse the multipart form data manually with increased limits
        from starlette.formparsers import MultiPartParser
        from starlette.datastructures import FormData
        
        # Create a custom parser with higher limits
        content_type = request.headers.get('content-type', '')
        if not content_type.startswith('multipart/form-data'):
            raise HTTPException(status_code=400, detail="Must be multipart/form-data")
        
        parser = MultiPartParser(
            headers=request.headers,
            stream=request.stream(),
            max_files=50000,  # Allow up to 50,000 files
            max_fields=50000  # Allow up to 50,000 form fields
        )
        
        form_data = await parser.parse()
        form = FormData(form_data)
        print(f"Form keys: {list(form.keys())}")
        
        # Get all files from the form
        files = []
        for key in form.keys():
            values = form.getlist(key)
            for value in values:
                if hasattr(value, 'filename') and value.filename:
                    files.append(value)
                    if len(files) <= 5:  # Only log first 5 files to avoid spam
                        print(f"Found file: {value.filename}")
        
        if len(files) > 5:
            print(f"... and {len(files) - 5} more files (total: {len(files)})")
        
        if not files:
            print("ERROR: No files found in form data")
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        print(f"Valid files: {len(files)}")
        
        # Process all files - no limit (use with caution for large uploads)
        print(f"Processing {len(files)} files...")
        
        temp_path = None
        try:
            # Save files to persistent directory
            upload_path = extract_files_from_uploads(files)
            
            # Parse the directory structure
            tree = parse_directory_recursive(upload_path)
            
            print(f"=== UPLOAD SUCCESS ===")
            return {
                "success": True,
                "file_count": len(files),
                "upload_path": str(upload_path),
                "tree": tree
            }
            
        except Exception as e:
            # Don't delete files on error - keep them for debugging
            print(f"Error during processing: {e}")
            raise
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
@router.get("/uploads")
async def list_uploads():
    """List all uploaded folders."""
    uploads_dir = Path(__file__).parent.parent / "uploads"
    if not uploads_dir.exists():
        return {"uploads": []}
    
    uploads = []
    for upload_dir in uploads_dir.iterdir():
        if upload_dir.is_dir():
            # Get basic info about the upload
            stat = upload_dir.stat()
            uploads.append({
                "name": upload_dir.name,
                "path": str(upload_dir),
                "created": stat.st_ctime,
                "size_mb": sum(f.stat().st_size for f in upload_dir.rglob('*') if f.is_file()) / (1024*1024)
            })
    
    # Sort by creation time (newest first)
    uploads.sort(key=lambda x: x["created"], reverse=True)
    return {"uploads": uploads}


@router.delete("/uploads/{upload_name}")
async def delete_upload(upload_name: str):
    """Delete a specific uploaded folder."""
    uploads_dir = Path(__file__).parent.parent / "uploads"
    upload_path = uploads_dir / upload_name
    
    if not upload_path.exists() or not upload_path.is_dir():
        raise HTTPException(status_code=404, detail="Upload not found")
    
    try:
        shutil.rmtree(upload_path)
        return {"success": True, "message": f"Deleted upload: {upload_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting upload: {str(e)}")


@router.post("/uploads/cleanup")
async def cleanup_old_uploads(days_old: int = 7):
    """Delete uploads older than specified days."""
    uploads_dir = Path(__file__).parent.parent / "uploads"
    if not uploads_dir.exists():
        return {"deleted": 0, "message": "No uploads directory"}
    
    import time
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    deleted = 0
    
    for upload_dir in uploads_dir.iterdir():
        if upload_dir.is_dir() and upload_dir.stat().st_ctime < cutoff_time:
            try:
                shutil.rmtree(upload_dir)
                deleted += 1
                print(f"Deleted old upload: {upload_dir.name}")
            except Exception as e:
                print(f"Error deleting {upload_dir.name}: {e}")
    
    return {"deleted": deleted, "message": f"Cleaned up uploads older than {days_old} days"}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/ai/analyze")
async def analyze_dataset(request: ChatRequest):
    """Analyze uploaded dataset and generate ML training plan."""
    try:
        upload_path = Path(request.upload_path)
        if not upload_path.exists():
            raise HTTPException(status_code=404, detail="Upload path not found")
        
        ai_service = AIAnalysisService()
        analysis = ai_service.analyze_directory(upload_path, request.message)
        
        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")


@router.post("/ml/train")
async def train_model(request: TrainingRequest):
    """Train a machine learning model based on the analysis plan."""
    try:
        upload_path = Path(request.upload_path)
        if not upload_path.exists():
            raise HTTPException(status_code=404, detail="Upload path not found")
        
        ml_service = ModelTrainingService(upload_path)
        
        # Determine model type and train accordingly
        model_type = request.ml_plan.get("model_type", "image_classification")
        
        if model_type == "image_classification":
            result = ml_service.train_image_classification_model(request.ml_plan)
        elif model_type == "text_classification":
            result = ml_service.train_text_classification_model(request.ml_plan)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@router.get("/ml/status")
async def get_training_status():
    """Get the status of model training."""
    try:
        ml_service = ModelTrainingService(Path("."))  # Just for status check
        status = ml_service.get_training_status()
        return {"success": True, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training status: {str(e)}")


@router.get("/models")
async def list_models():
    """List all trained models."""
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            return {"models": []}
        
        models = []
        for model_file in models_dir.glob("*.h5"):
            metadata_file = models_dir / f"{model_file.stem}_metadata.json"
            metadata = {}
            
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            models.append({
                "name": model_file.stem,
                "path": str(model_file),
                "metadata": metadata
            })
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.get("/ml/models")
async def get_available_models():
    """Get list of available trained models with metadata."""
    try:
        ml_service = ModelTrainingService(Path("."))  # Just for getting models
        result = ml_service.get_available_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")


@router.post("/ml/predict")
async def predict_image(file: UploadFile = File(...), model_name: str = ""):
    """Predict a single image using a trained model."""
    try:
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create temporary file for the uploaded image
        temp_dir = Path("temp_inference")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        temp_file = temp_dir / f"temp_image_{int(time.time())}{Path(file.filename).suffix}"
        
        try:
            with open(temp_file, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Create ML service and make prediction
            ml_service = ModelTrainingService(Path("."))  # Just for prediction
            result = ml_service.predict_image(temp_file, model_name)
            
            return result
            
        finally:
            # Cleanup temp file
            if temp_file.exists():
                temp_file.unlink()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting image: {str(e)}")


app.include_router(router)
