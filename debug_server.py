from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="Debug Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    print(f"\n=== {request.method} {request.url} ===")
    print(f"Content-Length: {request.headers.get('content-length', 'unknown')}")
    print(f"Content-Type: {request.headers.get('content-type', 'unknown')}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        print(f"✅ Completed in {process_time:.4f}s with status {response.status_code}")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        print(f"❌ Failed after {process_time:.4f}s with error: {e}")
        raise

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/debug-request")
async def debug_request(request: Request):
    """Debug endpoint to see what we're receiving."""
    print("\n=== DEBUG REQUEST ===")
    
    # Get basic info
    content_type = request.headers.get('content-type', '')
    content_length = request.headers.get('content-length', '0')
    
    result = {
        "content_type": content_type,
        "content_length": content_length,
        "headers": dict(request.headers),
    }
    
    try:
        if 'multipart/form-data' in content_type:
            # Try to read the raw body
            body = await request.body()
            result["body_size"] = len(body)
            result["body_preview"] = body[:500].decode('utf-8', errors='ignore')
            
            # Try to parse form data manually
            from fastapi import Form, File, UploadFile
            try:
                form = await request.form()
                result["form_keys"] = list(form.keys())
                result["form_values"] = {}
                
                for key, value in form.items():
                    if hasattr(value, 'filename'):  # It's a file
                        result["form_values"][key] = {
                            "type": "file",
                            "filename": value.filename,
                            "content_type": getattr(value, 'content_type', None),
                            "size": len(await value.read()) if hasattr(value, 'read') else 'unknown'
                        }
                        # Reset file pointer if possible
                        if hasattr(value, 'seek'):
                            await value.seek(0)
                    else:
                        result["form_values"][key] = str(value)[:100]  # Truncate long values
                        
            except Exception as form_error:
                result["form_parse_error"] = str(form_error)
        else:
            body = await request.body()
            result["body_size"] = len(body)
            result["body_preview"] = body[:500].decode('utf-8', errors='ignore')
            
    except Exception as e:
        result["error"] = str(e)
    
    print(f"Debug result: {result}")
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
