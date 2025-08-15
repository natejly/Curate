import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional
import shutil
from datetime import datetime

class ModelTrainingService:
    def __init__(self, upload_path: Path):
        self.upload_path = upload_path
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
    def train_image_classification_model(self, ml_plan: Dict) -> Dict:
        """Train an image classification model"""
        try:
            # Prepare data
            train_data, val_data, class_names, detected_shape = self._prepare_image_data(ml_plan)
            
            if train_data is None or detected_shape is None:
                return {"error": "Failed to prepare image data", "success": False}
            
            # Build model
            architecture = ml_plan.get("model_architecture", "CNN")
            print(f"Architecture from ml_plan: {architecture} (type: {type(architecture)})")
            
            # Ensure architecture is a string, not a dict
            if isinstance(architecture, dict):
                print("Warning: model_architecture is a dict, using CNN as fallback")
                architecture = "CNN"  # Default fallback
            elif not isinstance(architecture, str):
                print(f"Warning: model_architecture is {type(architecture)}, converting to string")
                architecture = str(architecture) if architecture else "CNN"
                
            model = self._build_image_model(
                input_shape=detected_shape,
                num_classes=len(class_names),
                architecture=architecture
            )
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=ml_plan.get("training_parameters", {}).get("epochs", 5),
                verbose=1
            )
            
            # Save model with AI-generated name
            model_name = ml_plan.get("model_name", f"image_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            model_save_path = self.model_path / f"{model_name}.h5"
            model.save(model_save_path)
            
            # Save metadata
            metadata = {
                "model_name": model_name,
                "model_type": "image_classification",
                "classes": class_names,
                "input_shape": detected_shape,
                "training_accuracy": float(max(history.history['accuracy'])),
                "validation_accuracy": float(max(history.history['val_accuracy'])) if 'val_accuracy' in history.history else None,
                "epochs_trained": len(history.history['accuracy']),
                "created_at": datetime.now().isoformat()
            }
            
            metadata_path = self.model_path / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "success": True,
                "model_name": model_name,
                "model_path": str(model_save_path),
                "metadata": metadata,
                "training_history": {
                    "accuracy": history.history['accuracy'],
                    "val_accuracy": history.history.get('val_accuracy', []),
                    "loss": history.history['loss'],
                    "val_loss": history.history.get('val_loss', [])
                }
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
        finally:
            # Cleanup temp training directory after training is complete
            temp_dir = Path("temp_training_data")
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print("Cleaned up temporary training directory")
                except Exception as e:
                    print(f"Error cleaning up temp directory: {e}")
    
    def _prepare_image_data(self, ml_plan: Dict) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], List[str]]:
        """Prepare image data for training"""
        try:
            # Find training folders
            training_folders = ml_plan.get("training_folders", [])
            if not training_folders:
                # Auto-detect class folders - look for folders with images
                training_folders = []
                for folder_path in self.upload_path.rglob('*'):
                    if folder_path.is_dir():
                        # Check if this folder contains images
                        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
                        has_images = any(f.suffix.lower() in image_extensions 
                                       for f in folder_path.iterdir() if f.is_file())
                        if has_images:
                            rel_path = folder_path.relative_to(self.upload_path)
                            training_folders.append(str(rel_path))
            
            print(f"Training folders found: {training_folders}")
            
            # Create temporary organized structure
            temp_dir = Path("temp_training_data")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            
            class_names = []
            for folder_name in training_folders:
                # Handle both relative paths and direct folder names
                folder_path = self.upload_path / folder_name
                if folder_path.exists() and folder_path.is_dir():
                    # Use the last part of the path as class name (e.g., "0", "1", "2" from "sorted_digits/0")
                    class_name = folder_path.name
                    class_names.append(class_name)
                    class_dir = temp_dir / class_name
                    class_dir.mkdir(exist_ok=True)
                    
                    # Copy images to class directory
                    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
                    copied_count = 0
                    for img_file in folder_path.rglob('*'):
                        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                            try:
                                # Use a unique filename to avoid conflicts
                                dest_filename = f"{copied_count}_{img_file.name}"
                                dest_path = class_dir / dest_filename
                                shutil.copy2(img_file, dest_path)
                                copied_count += 1
                                
                                # Verify the file was copied successfully
                                if not dest_path.exists():
                                    print(f"Warning: File {dest_path} was not created successfully")
                                    
                            except Exception as e:
                                print(f"Error copying {img_file} to {dest_path}: {e}")
                    
                    print(f"Copied {copied_count} images for class '{class_name}'")
            
            if not class_names:
                print("No class folders with images found!")
                return None, None, []
            
            print(f"Final class names: {class_names}")
            
            # Detect image shape from a few sample images
            detected_shape = self._detect_image_shape(temp_dir)
            print(f"Detected image shape: {detected_shape}")
            
            # Verify temp directory structure before creating datasets
            print(f"Temp directory structure:")
            for class_name in class_names:
                class_dir = temp_dir / class_name
                if class_dir.exists():
                    file_count = len([f for f in class_dir.iterdir() if f.is_file()])
                    print(f"  {class_name}/: {file_count} files")
                    
                    # List a few files to verify they exist
                    files = list(class_dir.iterdir())[:3]
                    for f in files:
                        if f.exists():
                            print(f"    ✓ {f.name} ({f.stat().st_size} bytes)")
                        else:
                            print(f"    ✗ {f.name} (missing!)")
                else:
                    print(f"  {class_name}/: directory missing!")
            
            # Create datasets with AI-determined batch size
            batch_size = ml_plan.get("training_parameters", {}).get("batch_size", 32)
            try:
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    temp_dir,
                    validation_split=0.2,
                    subset="training",
                    seed=123,
                    image_size=detected_shape[:2],  # Use detected width and height
                    batch_size=batch_size
                )
                
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    temp_dir,
                    validation_split=0.2,
                    subset="validation",
                    seed=123,
                    image_size=detected_shape[:2],  # Use detected width and height
                    batch_size=batch_size
                )
            except Exception as e:
                print(f"Error creating datasets: {e}")
                return None, None, []
            
            # Normalize data
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
            val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
            
            # Performance optimization
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            
            return train_ds, val_ds, class_names, detected_shape
            
        except Exception as e:
            print(f"Error preparing image data: {e}")
            # Only cleanup on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, [], None
    
    def _detect_image_shape(self, temp_dir: Path) -> Tuple[int, int, int]:
        """Detect the common image shape from sample images"""
        try:
            image_shapes = []
            sample_count = 0
            max_samples = 10  # Check up to 10 images per class
            
            # Sample images from each class
            for class_dir in temp_dir.iterdir():
                if class_dir.is_dir():
                    class_sample_count = 0
                    for img_file in class_dir.iterdir():
                        if img_file.is_file() and class_sample_count < max_samples:
                            try:
                                with Image.open(img_file) as img:
                                    # Convert to RGB if needed (for PNG with transparency, etc.)
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    width, height = img.size
                                    # Store as (height, width, channels) for TensorFlow
                                    image_shapes.append((height, width, 3))
                                    sample_count += 1
                                    class_sample_count += 1
                            except Exception as e:
                                print(f"Error reading image {img_file}: {e}")
                                continue
                        
                        if class_sample_count >= max_samples:
                            break
            
            if not image_shapes:
                print("No valid images found, using default shape (28, 28, 3)")
                return (28, 28, 3)
            
            # Find the most common shape
            from collections import Counter
            shape_counts = Counter(image_shapes)
            most_common_shape = shape_counts.most_common(1)[0][0]
            
            # If images are very small (like MNIST), keep original size
            # If images are large, resize to a reasonable size for training
            height, width, channels = most_common_shape
            
            # Adaptive sizing based on image dimensions
            if height <= 32 and width <= 32:
                # Small images like MNIST digits (28x28) - keep original size
                final_shape = most_common_shape
            elif height <= 64 and width <= 64:
                # Small-medium images - keep original or slightly reduce
                final_shape = most_common_shape
            elif height <= 128 and width <= 128:
                # Medium images - might keep original
                final_shape = most_common_shape
            else:
                # Large images - resize to more manageable size for training
                aspect_ratio = width / height
                if aspect_ratio > 1:
                    # Wider than tall
                    new_width = min(224, width)
                    new_height = int(new_width / aspect_ratio)
                else:
                    # Taller than wide or square
                    new_height = min(224, height)
                    new_width = int(new_height * aspect_ratio)
                final_shape = (new_height, new_width, channels)
            
            print(f"Detected {len(image_shapes)} sample images")
            print(f"Most common shape: {most_common_shape}")
            print(f"Final shape for training: {final_shape}")
            
            return final_shape
            
        except Exception as e:
            print(f"Error detecting image shape: {e}")
            print("Using default shape (28, 28, 3)")
            return (28, 28, 3)
    
    def _build_image_model(self, input_shape: Tuple[int, int, int], num_classes: int, architecture: str = "CNN") -> tf.keras.Model:
        """Build image classification model"""
        
        # Ensure architecture is a string and handle edge cases
        if not isinstance(architecture, str):
            architecture = "CNN"
        
        architecture = architecture.strip().upper()
        
        if architecture == "TRANSFER_LEARNING":
            # Use pre-trained MobileNetV2
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        else:
            # Simple CNN
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        
        return model
    
    def train_text_classification_model(self, ml_plan: Dict) -> Dict:
        """Train a text classification model"""
        # Placeholder for text classification
        return {
            "error": "Text classification not yet implemented",
            "success": False
        }
    
    def get_training_status(self) -> Dict:
        """Get status of current training"""
        return {
            "models_trained": len(list(self.model_path.glob("*.h5"))),
            "available_models": [f.stem for f in self.model_path.glob("*.h5")]
        }
    
    def predict_image(self, image_file: Path, model_name: str) -> Dict:
        """Predict a single image using a trained model"""
        try:
            # Load the model
            model_path = self.model_path / f"{model_name}.h5"
            if not model_path.exists():
                return {"error": f"Model {model_name} not found", "success": False}
            
            model = tf.keras.models.load_model(model_path)
            
            # Load metadata to get class names and input shape
            metadata_path = self.model_path / f"{model_name}_metadata.json"
            input_shape = (224, 224)  # Default fallback
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                class_names = metadata.get("classes", [])
                # Get input shape from metadata (height, width, channels) -> (width, height)
                stored_shape = metadata.get("input_shape", (224, 224, 3))
                input_shape = (stored_shape[1], stored_shape[0])  # Convert to (width, height) for load_img
            else:
                # Fallback class names for digit classification
                class_names = [str(i) for i in range(10)]
            
            # Load and preprocess the image
            image = tf.keras.utils.load_img(image_file, target_size=input_shape)
            image_array = tf.keras.utils.img_to_array(image)
            image_array = tf.expand_dims(image_array, 0)  # Add batch dimension
            
            # Normalize the image (same as training)
            image_array = image_array / 255.0
            
            # Make prediction
            predictions = model.predict(image_array, verbose=0)
            predicted_class_idx = int(tf.argmax(predictions[0]).numpy())
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get top 3 predictions
            top_indices = tf.nn.top_k(predictions[0], k=min(3, len(class_names))).indices.numpy()
            top_predictions = []
            for idx in top_indices:
                class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
                conf = float(predictions[0][idx])
                top_predictions.append({
                    "class": class_name,
                    "confidence": conf,
                    "percentage": f"{conf * 100:.1f}%"
                })
            
            return {
                "success": True,
                "predicted_class": class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Class_{predicted_class_idx}",
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.1f}%",
                "top_predictions": top_predictions,
                "model_used": model_name
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}", "success": False}
    
    def get_available_models(self) -> Dict:
        """Get list of available trained models with metadata"""
        try:
            models = []
            for model_file in self.model_path.glob("*.h5"):
                model_name = model_file.stem
                metadata_path = self.model_path / f"{model_name}_metadata.json"
                
                model_info = {
                    "name": model_name,
                    "file_size": model_file.stat().st_size,
                    "created_at": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                }
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_info.update({
                        "model_type": metadata.get("model_type", "unknown"),
                        "classes": metadata.get("classes", []),
                        "num_classes": len(metadata.get("classes", [])),
                        "training_accuracy": metadata.get("training_accuracy", 0),
                        "validation_accuracy": metadata.get("validation_accuracy", 0),
                        "epochs_trained": metadata.get("epochs_trained", 0)
                    })
                
                models.append(model_info)
            
            return {
                "success": True,
                "models": sorted(models, key=lambda x: x["created_at"], reverse=True),
                "total_models": len(models)
            }
            
        except Exception as e:
            return {"error": f"Failed to get models: {str(e)}", "success": False}
