import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Load environment variables
load_dotenv()

class AIAnalysisService:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.openai_api_key and self.openai_api_key != 'your_openai_api_key_here':
            try:
                # Initialize OpenAI client
                self.client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            print("OpenAI API key not found or is placeholder. Using fallback analysis.")
    
    def analyze_directory_structure(self, upload_path: Path) -> Dict:
        """Analyze the uploaded directory structure"""
        structure = self._get_directory_structure(upload_path)
        return {
            "structure": structure,
            "file_count": self._count_files(upload_path),
            "folders": self._get_folder_list(upload_path),
            "file_types": self._get_file_types(upload_path)
        }
    
    def _get_directory_structure(self, path: Path, max_depth: int = 3, current_depth: int = 0) -> Dict:
        """Get directory structure with limited depth"""
        if current_depth >= max_depth:
            return {"...": "truncated"}
        
        structure = {}
        try:
            for item in path.iterdir():
                if item.is_dir():
                    structure[f"{item.name}/"] = self._get_directory_structure(
                        item, max_depth, current_depth + 1
                    )
                else:
                    structure[item.name] = f"file ({item.stat().st_size} bytes)"
        except PermissionError:
            structure["error"] = "Permission denied"
        
        return structure
    
    def _count_files(self, path: Path) -> Dict:
        """Count files by type"""
        counts = {"total": 0, "images": 0, "text": 0, "other": 0}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        text_extensions = {'.txt', '.csv', '.json', '.xml', '.md', '.py', '.js', '.html'}
        
        for file_path in path.rglob('*'):
            if file_path.is_file():
                counts["total"] += 1
                ext = file_path.suffix.lower()
                if ext in image_extensions:
                    counts["images"] += 1
                elif ext in text_extensions:
                    counts["text"] += 1
                else:
                    counts["other"] += 1
        
        return counts
    
    def _get_folder_list(self, path: Path) -> List[str]:
        """Get list of all folders"""
        folders = []
        for item in path.rglob('*'):
            if item.is_dir():
                rel_path = item.relative_to(path)
                folders.append(str(rel_path))
        return sorted(folders)
    
    def _get_file_types(self, path: Path) -> Dict[str, int]:
        """Get count of each file type"""
        file_types = {}
        for file_path in path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower() or 'no_extension'
                file_types[ext] = file_types.get(ext, 0) + 1
        return dict(sorted(file_types.items()))
    
    def generate_ml_plan(self, user_description: str, directory_analysis: Dict) -> Dict:
        """Use OpenAI to generate an ML training plan"""
        
        # Create a concise summary of the directory for the AI
        dir_summary = {
            "file_count": directory_analysis["file_count"],
            "folders": directory_analysis["folders"][:20],  # Limit folders shown
            "file_types": directory_analysis["file_types"]
        }
        
        prompt = f"""
        Analyze this uploaded dataset and user request to create a machine learning training plan.

        USER REQUEST: {user_description}

        DATASET ANALYSIS:
        - Total files: {dir_summary['file_count']['total']}
        - Images: {dir_summary['file_count']['images']}
        - Text files: {dir_summary['file_count']['text']}
        - Other files: {dir_summary['file_count']['other']}
        
        FOLDER STRUCTURE:
        {json.dumps(dir_summary['folders'], indent=2)}
        
        FILE TYPES:
        {json.dumps(dir_summary['file_types'], indent=2)}

        Based on this information, provide a JSON response with:
        1. model_type: "image_classification", "text_classification", "object_detection", "regression", etc.
        2. training_folders: List of folder names that should be used for training
        3. validation_strategy: How to split data for validation
        4. preprocessing_steps: What preprocessing is needed
        5. model_architecture: Suggested model architecture
        6. model_name: A descriptive, semantic name for this model based on the dataset content and purpose (e.g., "handwritten_digit_classifier", "dog_breed_identifier", "medical_xray_analyzer")
        7. training_parameters: Suggested parameters including:
           - epochs: intelligent number based on dataset size
           - batch_size: optimal batch size for the dataset
           - learning_rate: appropriate learning rate
        8. feasibility_score: Score from 1-10 how feasible this is
        9. recommendations: Additional recommendations or warnings

        IMPORTANT: The model_name should be descriptive and semantic, reflecting what the model actually does.
        Examples: "mnist_digit_classifier", "cat_dog_binary_classifier", "flower_species_identifier"

        Respond ONLY with valid JSON.
        """
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert ML engineer. Analyze datasets and create training plans. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                
                ml_plan = json.loads(response.choices[0].message.content)
                return ml_plan
            else:
                # Use fallback when OpenAI is not available
                raise Exception("OpenAI client not available")
                
        except Exception as e:
            # Fallback plan if OpenAI fails or is not available
            print(f"Using fallback ML plan generation. Error: {str(e)}")
            
            # Intelligent fallback based on directory analysis
            folders = directory_analysis.get("folders", [])
            file_counts = directory_analysis.get("file_count", {})
            images = file_counts.get("images", 0)
            total_files = file_counts.get("total", 0)
            
            # Filter folders to only include those that likely contain training data
            # Look for leaf folders (folders that contain files, not other folders)
            training_folders = []
            for folder in folders:
                # For digit classification, we want folders like "sorted_digits/0", "sorted_digits/1", etc.
                if '/' in folder and not any(f.startswith(folder + '/') for f in folders):
                    training_folders.append(folder)
            
            # If no nested folders, use direct folders
            if not training_folders:
                training_folders = [f for f in folders if '/' not in f][:10]
            else:
                training_folders = training_folders[:100]  # Limit to 10 classes for demo
            
            # Determine model type based on data
            if images > total_files * 0.8:  # Mostly images
                model_type = "image_classification"
                model_architecture = "CNN" if images < 1000 else "transfer_learning"
                task_description = f"Image classification with {len(training_folders)} classes"
            else:
                model_type = "data_analysis"
                model_architecture = "dense"
                task_description = "General data classification"
            
            # Intelligent training parameter selection
            def calculate_epochs(image_count, class_count):
                """Calculate optimal epochs based on dataset characteristics"""
                if image_count < 100:
                    return 15  # Small dataset needs more epochs
                elif image_count < 1000:
                    return 10  # Medium dataset
                elif image_count < 5000:
                    return 8   # Large dataset
                else:
                    return 5   # Very large dataset - fewer epochs to prevent overfitting
            
            def calculate_batch_size(image_count):
                """Calculate optimal batch size based on dataset size and memory efficiency"""
                if image_count < 100:
                    return 8   # Small batch for small datasets
                elif image_count < 500:
                    return 16  # Small-medium batch
                elif image_count < 2000:
                    return 32  # Medium batch
                elif image_count < 10000:
                    return 64  # Large batch
                else:
                    return 128 # Very large batch for efficiency
            
            def generate_model_name(folders, image_count, model_architecture):
                """Generate semantic, descriptive model name based on dataset characteristics"""
                import datetime
                
                # Get class names from folder paths
                class_names = []
                for folder in folders[:5]:  # Look at more classes for better context
                    # Extract class name from path (e.g., "sorted_digits/0" -> "0")
                    class_name = folder.split('/')[-1] if '/' in folder else folder
                    class_names.append(class_name.lower())
                
                class_count = len(folders)
                
                # Generate semantic base name based on detected patterns
                def infer_domain_and_task(class_names, class_count):
                    """Infer the domain and task from class names"""
                    # Check for common patterns
                    if all(name.isdigit() for name in class_names[:10]):
                        return "digit_classifier"
                    elif any(word in " ".join(class_names) for word in ["cat", "dog", "animal"]):
                        return "animal_classifier" 
                    elif any(word in " ".join(class_names) for word in ["flower", "plant", "leaf"]):
                        return "plant_classifier"
                    elif any(word in " ".join(class_names) for word in ["face", "person", "human"]):
                        return "person_classifier"
                    elif any(word in " ".join(class_names) for word in ["car", "vehicle", "truck", "bike"]):
                        return "vehicle_classifier"
                    elif any(word in " ".join(class_names) for word in ["food", "fruit", "vegetable"]):
                        return "food_classifier"
                    elif class_count == 2:
                        # Binary classification - use both class names
                        return f"{class_names[0]}_vs_{class_names[1]}_classifier"
                    elif class_count <= 5:
                        # Small multiclass - use descriptive name
                        return f"{class_names[0]}_multiclass_classifier"
                    else:
                        # Large multiclass - generic but descriptive
                        return f"{class_count}class_image_classifier"
                
                # Get semantic base name
                base_name = infer_domain_and_task(class_names, class_count)
                
                # Add dataset size indicator for context
                size_indicator = ""
                if image_count < 100:
                    size_indicator = "small"
                elif image_count < 1000:
                    size_indicator = "medium"
                elif image_count < 10000:
                    size_indicator = "large"
                else:
                    size_indicator = "xlarge"
                
                # Add architecture type
                arch_type = "transfer" if model_architecture == "transfer_learning" else "custom"
                
                # Add timestamp for uniqueness (shorter format)
                timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
                
                # Combine parts semantically
                model_name = f"{base_name}_{size_indicator}_{arch_type}_{timestamp}"
                
                # Clean up the name (remove special characters, limit length)
                model_name = "".join(c for c in model_name if c.isalnum() or c in "_-")[:50]
                
                return model_name
            
            # Calculate intelligent parameters
            epochs = calculate_epochs(images, len(training_folders))
            batch_size = calculate_batch_size(images)
            model_name = generate_model_name(training_folders, images, model_architecture)
            
            # Generate intelligent recommendations
            recommendations = []
            if len(training_folders) > 10:
                recommendations.append("Large number of classes detected - consider grouping similar categories")
            if images > 5000:
                recommendations.append("Large dataset - transfer learning recommended for faster training")
            if len(training_folders) < 2:
                recommendations.append("Need at least 2 classes for classification - check data structure")
            
            # Add parameter recommendations
            recommendations.append(f"📊 Intelligent parameters: {epochs} epochs, batch size {batch_size}")
            recommendations.append(f"🏷️ Generated model name: {model_name}")
            
            # Check if it's an API key issue
            if "401" in str(e) or "invalid_api_key" in str(e):
                recommendations.append("⚠️ OpenAI API key is invalid or expired - using intelligent fallback analysis")
            else:
                recommendations.append("⚠️ AI analysis temporarily unavailable - using intelligent fallback analysis")
            
            return {
                "model_type": model_type,
                "task_description": task_description,
                "training_folders": training_folders,  # Use the filtered training folders
                "validation_strategy": "80/20 split",
                "preprocessing_steps": ["normalize", "resize"] if model_type == "image_classification" else ["normalize"],
                "model_architecture": model_architecture,
                "model_name": model_name,
                "training_parameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": 0.001
                },
                "feasibility_score": min(10, 5 + len(training_folders)),
                "recommendations": recommendations,
                "analysis_status": "fallback_mode",
                "analysis_note": "Using intelligent rule-based analysis. For full AI analysis, please check your OpenAI API key."
            }
    
    def analyze_directory(self, upload_path: Path, user_message: str) -> Dict:
        """Main analysis method that combines directory analysis with AI-powered ML plan generation"""
        try:
            # Analyze directory structure
            directory_analysis = self.analyze_directory_structure(upload_path)
            
            # Generate ML plan using AI
            ml_plan = self.generate_ml_plan(user_message, directory_analysis)
            
            # Create comprehensive analysis result
            return {
                "directory_analysis": directory_analysis,
                "ml_plan": ml_plan,
                "ml_plan_text": self._format_ml_plan_text(ml_plan),
                "folders_with_data": directory_analysis["folders"],
                "file_type_summary": directory_analysis["file_types"]
            }
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "directory_analysis": None,
                "ml_plan": None
            }
    
    def _format_ml_plan_text(self, ml_plan: Dict) -> str:
        """Format the ML plan into readable text"""
        if not ml_plan:
            return "Failed to generate ML plan"
        
        # Show analysis status
        status_icon = "🤖" if ml_plan.get("analysis_status") != "fallback_mode" else "🔧"
        
        text = f"{status_icon} **ML Training Plan**\n\n"
        
        # Add status note if in fallback mode
        if ml_plan.get("analysis_status") == "fallback_mode":
            text += f"📋 **Analysis Mode**: Rule-based fallback\n"
            text += f"💡 **Note**: {ml_plan.get('analysis_note', 'Using intelligent fallback analysis')}\n\n"
        
        text += f"**Task**: {ml_plan.get('task_description', ml_plan.get('model_type', 'Unknown'))}\n"
        text += f"**Training Data**: {len(ml_plan.get('training_folders', []))} folders\n"
        text += f"**Model Architecture**: {ml_plan.get('model_architecture', 'Auto-detect')}\n"
        text += f"**Validation Strategy**: {ml_plan.get('validation_strategy', '80/20 split')}\n"
        text += f"**Feasibility**: {ml_plan.get('feasibility_score', 0)}/10\n\n"
        
        if ml_plan.get('recommendations'):
            text += "**Recommendations**:\n"
            for rec in ml_plan['recommendations']:
                text += f"• {rec}\n"
        
        return text