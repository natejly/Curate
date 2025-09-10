#!/usr/bin/env python3
"""
Extract comprehensive knowledge from prompts.py and upload to Pinecone knowledge base
This allows prompts to be simplified and rely on RAG for context
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_knowledge_from_prompts():
    """Extract comprehensive structured knowledge from prompts.py"""
    
    knowledge_base = [
        # Training Strategy Knowledge
        {
            "id": "training_approaches_overview",
            "text": "Machine learning training approaches for image classification: Single-stage training trains the entire model from the start with unfrozen layers, good for larger datasets where you want immediate feature adaptation. Dual-stage training first trains only the top layers (feature extractor frozen), then unfreezes a percentage of layers for fine-tuning. Better for smaller datasets or when transfer learning benefits are important.",
            "category": "training_strategy",
            "topic": "single_vs_dual_stage",
            "source": "prompts.py"
        },
        {
            "id": "dual_stage_training_details",
            "text": "Dual-stage training implementation: Stage 1 - Train only classifier head while keeping backbone frozen, use higher learning rate (1e-3 to 1e-2). Stage 2 - Unfreeze percentage of backbone layers, use lower learning rate (1e-4 to 1e-5). Monitor validation accuracy between stages. Switch to stage 2 when stage 1 validation accuracy plateaus.",
            "category": "training_strategy",
            "topic": "dual_stage_implementation",
            "source": "prompts.py"
        },
        {
            "id": "transfer_learning_best_practices",
            "text": "Transfer learning best practices: Start with pre-trained weights from ImageNet. Freeze early layers initially to preserve low-level features. Gradually unfreeze layers during training. Use lower learning rates for pre-trained layers. Fine-tune classifier head first, then gradually include more layers. Monitor validation performance to prevent overfitting.",
            "category": "training_strategy",
            "topic": "transfer_learning",
            "source": "prompts.py"
        },
        {
            "id": "progressive_unfreezing",
            "text": "Progressive unfreezing strategy: Start with completely frozen backbone, train classifier for 3-5 epochs. Gradually unfreeze layers from top to bottom in groups of 10-20%. Allow 2-3 epochs between unfreezing steps. Monitor validation loss at each step. Stop unfreezing if validation performance degrades. Use discriminative learning rates with lower rates for earlier layers.",
            "category": "training_strategy",
            "topic": "progressive_unfreezing",
            "source": "prompts.py"
        },
        {
            "id": "curriculum_learning",
            "text": "Curriculum learning for image classification: Start training with easier examples (clear, well-lit, centered objects) before introducing difficult cases (occluded, poor lighting, unusual angles). Gradually increase dataset complexity over epochs. Can improve convergence speed and final accuracy. Particularly effective for complex datasets with high intra-class variation.",
            "category": "training_strategy",
            "topic": "curriculum_learning",
            "source": "prompts.py"
        },

        # Hyperparameter Optimization Knowledge
        {
            "id": "hyperparameter_priorities",
            "text": "Hyperparameter optimization priority order: 1. Learning rates (most impactful) - make order-of-magnitude adjustments if problematic. 2. Batch size - make significant changes based on dataset size and memory constraints. 3. Epochs - substantial increases/decreases if under/overfitting detected. 4. Unfreeze percent - bold adjustments if fine-tuning issues. 5. Image size - optimize input dimensions. 6. Training approach - switch between single/dual stage.",
            "category": "hyperparameters",
            "topic": "optimization_priority",
            "source": "prompts.py"
        },
        {
            "id": "learning_rate_guidelines",
            "text": "Learning rate optimization for transfer learning: Use lower learning rate (1e-4 to 1e-5) for pre-trained layers and higher learning rate (1e-3 to 1e-2) for new classifier head. This prevents destroying learned features while allowing new layers to adapt quickly. For fine-tuning, use even lower rates to preserve pre-trained weights.",
            "category": "hyperparameters",
            "topic": "learning_rate_transfer_learning",
            "source": "prompts.py"
        },
        {
            "id": "learning_rate_schedules",
            "text": "Learning rate scheduling strategies: Step decay - reduce LR by factor (0.1-0.5) every N epochs. Cosine annealing - smooth reduction following cosine curve. Exponential decay - gradual exponential reduction. OneCycleLR - ramp up then down for super-convergence. ReduceLROnPlateau - reduce when validation metric plateaus. Warm restarts - periodically reset to higher LR.",
            "category": "hyperparameters",
            "topic": "learning_rate_scheduling",
            "source": "prompts.py"
        },
        {
            "id": "batch_size_optimization",
            "text": "Batch size selection strategy: For limited GPU memory, start with batch size 16 or 32. Smaller batch sizes (8-16) provide more gradient updates but may be noisier. Larger batch sizes (64-128) are more stable but require more memory. Use gradient accumulation if you need effective large batch sizes with limited memory. Power of 2 values are optimal.",
            "category": "hyperparameters",
            "topic": "batch_size_selection",
            "source": "prompts.py"
        },
        {
            "id": "batch_size_effects",
            "text": "Batch size effects on training: Large batches (>64) provide stable gradients but may converge to sharp minima with poor generalization. Small batches (<16) add noise that can help escape local minima but make training less stable. Medium batches (16-64) balance stability and generalization. Increase learning rate proportionally when increasing batch size.",
            "category": "hyperparameters",
            "topic": "batch_size_effects",
            "source": "prompts.py"
        },
        {
            "id": "epoch_recommendations",
            "text": "Epoch selection guidelines: For initial training, start with 10-20 epochs and monitor validation loss. For fine-tuning, use 5-10 epochs with careful monitoring to prevent overfitting. Increase epochs substantially if underfitting is detected. Decrease if overfitting occurs early. Use early stopping for automatic optimization.",
            "category": "hyperparameters",
            "topic": "epoch_selection",
            "source": "prompts.py"
        },
        {
            "id": "early_stopping_strategies",
            "text": "Early stopping implementation: Monitor validation loss with patience of 3-7 epochs. Save best model weights based on validation metric. Use min_delta threshold (0.001-0.01) to avoid stopping on noise. For small datasets, use higher patience. For large datasets, lower patience is acceptable. Consider monitoring multiple metrics (accuracy, F1, precision).",
            "category": "hyperparameters",
            "topic": "early_stopping",
            "source": "prompts.py"
        },
        {
            "id": "image_size_optimization",
            "text": "Image size selection for optimal performance: Match input size to model architecture - 224x224 for EfficientNet-B0, 300x300 for B3/B4. Larger images provide more detail but require more computation and memory. Consider dataset characteristics - if images have fine details, use larger input sizes. Square dimensions are typically optimal for most architectures.",
            "category": "hyperparameters",
            "topic": "image_dimensions",
            "source": "prompts.py"
        },
        {
            "id": "resolution_scaling_effects",
            "text": "Resolution scaling effects: Higher resolution (384x384, 512x512) improves accuracy for fine-grained classification but increases computation 4x per dimension doubling. Use progressive resizing - start training at lower resolution, then fine-tune at higher resolution. Test-time augmentation with multiple scales can improve accuracy without training overhead.",
            "category": "hyperparameters",
            "topic": "resolution_scaling",
            "source": "prompts.py"
        },
        {
            "id": "unfreeze_percent_strategy",
            "text": "Layer unfreezing strategy for transfer learning: Start with unfreezing top 20-50% of layers for fine-tuning. Higher unfreeze percentage (0.7-1.0) for larger datasets or when significant adaptation is needed. Lower percentage (0.2-0.4) for smaller datasets to prevent overfitting. Monitor validation performance to adjust unfreezing strategy.",
            "category": "hyperparameters",
            "topic": "layer_unfreezing",
            "source": "prompts.py"
        },

        # Model Architecture Knowledge
        {
            "id": "architecture_change_guidelines",
            "text": "Architecture changes should be EXTREMELY RARE - only when hyperparameters cannot solve fundamental capacity issues. DO NOT change architecture unless hyperparameters have been thoroughly optimized first. Only consider architecture changes if current model has fundamental capacity limitations. Hyperparameter optimization should solve 95% of performance issues. Provide compelling justification if architecture change is truly required.",
            "category": "model_architecture",
            "topic": "change_guidelines",
            "source": "prompts.py"
        },
        {
            "id": "efficientnet_selection",
            "text": "EfficientNet model selection guidelines: EfficientNet-B0 is good for quick experiments and limited resources with 224x224 input. EfficientNet-B3 or B4 provide better accuracy for production use with 300x300 input. Scale the input image size to match the model's expected input dimensions for optimal performance.",
            "category": "model_architecture", 
            "topic": "efficientnet_variants",
            "source": "prompts.py"
        },
        {
            "id": "model_capacity_assessment",
            "text": "Model capacity assessment: Underfitting signs - training and validation loss both high and decreasing slowly, low training accuracy. Overfitting signs - training accuracy high but validation accuracy low, validation loss increasing while training loss decreases. Right capacity - both losses converge to similar low values with good generalization gap.",
            "category": "model_architecture",
            "topic": "capacity_assessment",
            "source": "prompts.py"
        },
        {
            "id": "architecture_selection_criteria",
            "text": "Architecture selection criteria: Dataset size - larger datasets can support bigger models (ResNet152, EfficientNet-B7). Computational constraints - mobile deployment needs efficient models (MobileNet, EfficientNet-B0). Accuracy requirements - high accuracy needs larger models. Inference speed - real-time applications need optimized architectures. Memory constraints affect model choice.",
            "category": "model_architecture",
            "topic": "selection_criteria",
            "source": "prompts.py"
        },
        {
            "id": "regularization_techniques",
            "text": "Regularization techniques for overfitting: Dropout (0.2-0.5) in classifier head, avoid in convolutional layers. L2 weight decay (1e-4 to 1e-2) on all parameters except batch norm. Data augmentation as implicit regularization. Early stopping prevents overtraining. Batch normalization provides regularization effect. Label smoothing (0.1) reduces overconfidence.",
            "category": "model_architecture",
            "topic": "regularization",
            "source": "prompts.py"
        },

        # Dataset Analysis Knowledge
        {
            "id": "dataset_complexity_assessment",
            "text": "Dataset complexity assessment criteria: Low complexity - simple objects, clear backgrounds, limited variations. Medium complexity - moderate variations in lighting, pose, background. High complexity - complex scenes, occlusion, high intra-class variation. Complexity affects choice of training approach, learning rates, and model architecture.",
            "category": "dataset_analysis",
            "topic": "complexity_assessment",
            "source": "prompts.py"
        },
        {
            "id": "dataset_size_implications",
            "text": "Dataset size implications for training: Small datasets (<1000 per class) - use aggressive data augmentation, lower learning rates, transfer learning essential. Medium datasets (1000-10000 per class) - moderate augmentation, can fine-tune more layers. Large datasets (>10000 per class) - can train from scratch, less augmentation needed, higher learning rates acceptable.",
            "category": "dataset_analysis",
            "topic": "size_implications",
            "source": "prompts.py"
        },
        {
            "id": "class_imbalance_handling",
            "text": "Class imbalance handling strategies: Weighted loss functions - assign higher weights to minority classes. Resampling techniques - oversample minority or undersample majority classes. Focal loss - focuses on hard examples, reduces easy example contribution. Data augmentation targeting minority classes. Ensemble methods with balanced subsets. Monitor per-class metrics, not just overall accuracy.",
            "category": "dataset_analysis",
            "topic": "class_imbalance",
            "source": "prompts.py"
        },
        {
            "id": "data_quality_assessment",
            "text": "Data quality assessment checklist: Label accuracy - manually verify sample of labels. Image quality - check for corruption, blur, artifacts. Class distribution - identify severe imbalances. Duplicate detection - remove near-duplicate images. Outlier detection - identify mislabeled or irrelevant images. Annotation consistency - ensure consistent labeling standards.",
            "category": "dataset_analysis",
            "topic": "quality_assessment",
            "source": "prompts.py"
        },
        {
            "id": "train_val_test_splits",
            "text": "Dataset splitting best practices: Standard split - 70% train, 15% validation, 15% test. For small datasets - use cross-validation instead of fixed splits. Stratified splitting - maintain class distributions across splits. Time-based splitting for temporal data. Ensure no data leakage between splits. Validation set should represent real-world distribution.",
            "category": "dataset_analysis",
            "topic": "data_splitting",
            "source": "prompts.py"
        },

        # Data Augmentation Knowledge
        {
            "id": "augmentation_strategies",
            "text": "Data augmentation strategies: Geometric transforms - rotation (±15°), horizontal flip, zoom (0.8-1.2), translation (±10%). Photometric transforms - brightness (±20%), contrast (±20%), saturation (±20%), hue (±10%). Advanced techniques - mixup, cutmix, cutout, autoaugment. Match augmentation intensity to dataset size and complexity.",
            "category": "data_augmentation",
            "topic": "augmentation_types",
            "source": "prompts.py"
        },
        {
            "id": "augmentation_intensity",
            "text": "Augmentation intensity guidelines: Light augmentation - minimal transforms, preserve object structure. Moderate augmentation - standard geometric and color transforms. Heavy augmentation - aggressive transforms, mixup techniques. For small datasets use heavy augmentation. For large datasets use light-moderate augmentation. Monitor validation performance to tune intensity.",
            "category": "data_augmentation",
            "topic": "intensity_levels",
            "source": "prompts.py"
        },
        {
            "id": "test_time_augmentation",
            "text": "Test-time augmentation (TTA) techniques: Apply multiple augmentations to test images and average predictions. Common TTA - horizontal flip, multiple crops, rotation. Improves accuracy but increases inference time. Use 4-10 augmentations for good balance. Particularly effective for medical imaging and fine-grained classification.",
            "category": "data_augmentation",
            "topic": "test_time_augmentation",
            "source": "prompts.py"
        },

        # Performance Optimization Knowledge
        {
            "id": "performance_troubleshooting",
            "text": "Performance troubleshooting guidelines: If validation accuracy plateaus, try reducing learning rate or increasing model capacity. If training loss decreases but validation loss increases, reduce learning rate or add regularization. If both training and validation loss plateau, increase learning rate or model capacity. Monitor gradient norms to detect vanishing/exploding gradients.",
            "category": "troubleshooting",
            "topic": "performance_issues",
            "source": "prompts.py"
        },
        {
            "id": "convergence_issues",
            "text": "Convergence troubleshooting: Loss not decreasing - learning rate too high or too low, check gradients. Loss oscillating - reduce learning rate or batch size. Slow convergence - increase learning rate, check data loading bottlenecks. Gradient explosion - reduce learning rate, add gradient clipping. Vanishing gradients - check activation functions, consider residual connections.",
            "category": "troubleshooting",
            "topic": "convergence_issues",
            "source": "prompts.py"
        },
        {
            "id": "memory_optimization",
            "text": "Memory optimization strategies: Reduce batch size if out of memory. Use gradient accumulation for effective large batch training. Mixed precision training (fp16) reduces memory usage. Gradient checkpointing trades computation for memory. Use smaller input resolution. Consider model compression techniques. Monitor GPU memory usage throughout training.",
            "category": "optimization",
            "topic": "memory_optimization",
            "source": "prompts.py"
        },
        {
            "id": "training_speed_optimization",
            "text": "Training speed optimization: Use DataLoader with multiple workers and pin_memory. Optimize data pipeline - preprocess and cache data. Use appropriate batch size for GPU utilization. Mixed precision training for speed gains. Profile training to identify bottlenecks. Consider distributed training for very large models.",
            "category": "optimization",
            "topic": "speed_optimization",
            "source": "prompts.py"
        },

        # Evaluation and Metrics Knowledge
        {
            "id": "metric_selection",
            "text": "Metric selection guidelines: Balanced datasets - accuracy is appropriate. Imbalanced datasets - use precision, recall, F1-score, AUC-ROC. Multi-class problems - macro/micro averaged metrics. Top-k accuracy for large number of classes. Confusion matrices for detailed per-class analysis. Monitor multiple metrics simultaneously.",
            "category": "evaluation",
            "topic": "metric_selection",
            "source": "prompts.py"
        },
        {
            "id": "validation_strategies",
            "text": "Validation strategies: Hold-out validation - standard train/val/test split. Cross-validation - k-fold for small datasets. Stratified validation - maintain class distributions. Time-series validation - temporal splits for time-dependent data. Leave-one-out for very small datasets. Bootstrap validation for uncertainty estimation.",
            "category": "evaluation",
            "topic": "validation_strategies",
            "source": "prompts.py"
        },
        {
            "id": "overfitting_detection",
            "text": "Overfitting detection indicators: Training accuracy much higher than validation accuracy (>10% gap). Validation loss increases while training loss decreases. Model performs well on training data but poorly on new data. High variance in validation metrics across epochs. Early peak in validation accuracy followed by decline.",
            "category": "evaluation",
            "topic": "overfitting_detection",
            "source": "prompts.py"
        },

        # Production and Deployment Knowledge
        {
            "id": "model_optimization_deployment",
            "text": "Model optimization for deployment: Quantization - reduce precision from fp32 to int8. Pruning - remove unimportant weights/neurons. Knowledge distillation - train smaller student model. ONNX export for cross-platform deployment. TensorRT/OpenVINO for optimized inference. Batch inference for throughput optimization.",
            "category": "deployment",
            "topic": "model_optimization",
            "source": "prompts.py"
        },
        {
            "id": "inference_optimization",
            "text": "Inference optimization techniques: Model serving with batching for efficiency. Caching for repeated queries. Multi-threading for CPU inference. GPU memory management for batch inference. Pipeline parallelism for large models. Edge deployment considerations - model size, latency, power consumption.",
            "category": "deployment",
            "topic": "inference_optimization",
            "source": "prompts.py"
        },

        # Response Format Knowledge
        {
            "id": "json_response_format",
            "text": "Required JSON response format for hyperparameter recommendations: Use exactly the key 'value' for parameter values. Return proper data types: batch_size as integer, learning rates as float, epochs as integer, image_size as array of two integers, unfreeze_percent as float between 0 and 1, dual_stage as boolean. Include reasoning for each recommendation.",
            "category": "response_format",
            "topic": "json_structure",
            "source": "prompts.py"
        },
        {
            "id": "optimization_impact_requirements",
            "text": "Optimization impact requirements: Make changes that will create measurable performance differences. Avoid minor tweaks - recommend bold adjustments. Prioritize changes with highest expected impact. Focus on hyperparameter changes with clear reasoning. Make substantial optimizations that will significantly impact performance.",
            "category": "optimization_strategy",
            "topic": "impact_requirements",
            "source": "prompts.py"
        },
        {
            "id": "recommendation_justification",
            "text": "Recommendation justification requirements: Provide clear reasoning for each hyperparameter choice. Explain expected impact on training dynamics. Reference dataset characteristics in recommendations. Consider computational constraints and trade-offs. Prioritize changes with highest probability of success. Base recommendations on established best practices.",
            "category": "response_format",
            "topic": "justification_requirements",
            "source": "prompts.py"
        }
    ]
    
    return knowledge_base

def upload_to_pinecone(knowledge_base):
    """Upload knowledge base to Pinecone"""
    try:
        from pinecone import Pinecone
        import openai
        
        # Initialize clients
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        knowledge_index_name = os.getenv('PINECONE_KNOWLEDGE_INDEX_NAME', 'curate-knowledge')
        
        if not pinecone_api_key:
            print("❌ PINECONE_API_KEY not found")
            return False
            
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found")
            return False
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Create or connect to knowledge index
        existing_indexes = [index.name for index in pc.list_indexes()]
        if knowledge_index_name not in existing_indexes:
            print(f"🔨 Creating knowledge index: {knowledge_index_name}")
            pc.create_index(
                name=knowledge_index_name,
                dimension=512,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        
        index = pc.Index(knowledge_index_name)
        print(f"✅ Connected to knowledge index: {knowledge_index_name}")
        
        # Generate embeddings and upload
        vectors_to_upsert = []
        
        for i, knowledge in enumerate(knowledge_base):
            print(f"Processing {i+1}/{len(knowledge_base)}: {knowledge['id']}")
            
            # Generate embeddings
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=knowledge['text'],
                dimensions=512
            )
            
            embeddings = response.data[0].embedding
            
            # Prepare vector
            vector = {
                "id": knowledge['id'],
                "values": embeddings,
                "metadata": {
                    "text": knowledge['text'],
                    "category": knowledge['category'],
                    "topic": knowledge['topic'],
                    "source": knowledge['source']
                }
            }
            
            vectors_to_upsert.append(vector)
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            print(f"📤 Upserting batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}")
            upsert_response = index.upsert(vectors=batch)
            print(f"✅ Batch upserted: {upsert_response.upserted_count} vectors")
        
        print(f"✅ Successfully upserted {len(vectors_to_upsert)} total vectors")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload to Pinecone: {str(e)}")
        return False

def test_knowledge_retrieval():
    """Test retrieving knowledge from Pinecone"""
    try:
        from pinecone import Pinecone
        import openai
        
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        knowledge_index_name = os.getenv('PINECONE_KNOWLEDGE_INDEX_NAME', 'curate-knowledge')
        
        pc = Pinecone(api_key=pinecone_api_key)
        openai_client = openai.OpenAI(api_key=openai_api_key)
        index = pc.Index(knowledge_index_name)
        
        # Test queries
        test_queries = [
            "What is dual stage training?",
            "How do I choose learning rates for transfer learning?",
            "When should I change model architecture?",
            "How to select batch size for limited GPU memory?",
            "What augmentation strategies should I use?",
            "How to handle class imbalance in datasets?",
            "What are signs of overfitting?",
            "How to optimize model for deployment?",
            "What learning rate schedule should I use?",
            "How to assess dataset complexity?"
        ]
        
        print("\n🔍 Testing knowledge retrieval...")
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Generate query embedding
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                dimensions=512
            )
            
            query_embedding = response.data[0].embedding
            
            # Search index
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            print(f"Found {len(results.matches)} results:")
            for i, match in enumerate(results.matches):
                score = match.score
                category = match.metadata.get('category', 'Unknown')
                topic = match.metadata.get('topic', 'Unknown')
                text = match.metadata.get('text', 'No text')[:150] + "..."
                print(f"  {i+1}. Score: {score:.3f} | {category}/{topic}")
                print(f"     {text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge retrieval test failed: {str(e)}")
        return False

def generate_knowledge_summary():
    """Generate summary of knowledge base contents"""
    knowledge_base = extract_knowledge_from_prompts()
    
    print("\n📊 Knowledge Base Summary")
    print("=" * 50)
    
    # Category breakdown
    categories = {}
    for knowledge in knowledge_base:
        category = knowledge['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(knowledge['topic'])
    
    print(f"Total Knowledge Entries: {len(knowledge_base)}")
    print(f"Categories: {len(categories)}")
    print()
    
    for category, topics in categories.items():
        print(f"📂 {category.upper().replace('_', ' ')}")
        for topic in topics:
            print(f"   • {topic.replace('_', ' ').title()}")
        print()

def main():
    """Main function to extract and upload comprehensive prompt knowledge"""
    print("🧠 Comprehensive ML Training Knowledge Extraction and Upload")
    print("=" * 60)
    
    # Extract knowledge
    print("📚 Extracting comprehensive knowledge from prompts.py...")
    knowledge_base = extract_knowledge_from_prompts()
    print(f"✅ Extracted {len(knowledge_base)} knowledge entries")
    
    # Generate summary
    generate_knowledge_summary()
    
    # Upload to Pinecone
    print(f"\n📤 Uploading to Pinecone...")
    success = upload_to_pinecone(knowledge_base)
    
    if success:
        print("\n✅ Comprehensive knowledge base upload complete!")
        
        # Test retrieval
        print("\n🧪 Testing knowledge retrieval...")
        test_success = test_knowledge_retrieval()
        
        if test_success:
            print("\n🎉 Comprehensive ML training knowledge base is ready!")
            print("Your prompts can now be simplified and rely on RAG for detailed context.")
            print("The knowledge base covers:")
            print("  • Training strategies and approaches")
            print("  • Hyperparameter optimization")
            print("  • Model architecture guidance") 
            print("  • Dataset analysis and preprocessing")
            print("  • Data augmentation techniques")
            print("  • Performance troubleshooting")
            print("  • Evaluation and metrics")
            print("  • Deployment optimization")
        else:
            print("\n⚠️  Upload successful but retrieval test failed")
    else:
        print("\n❌ Failed to upload knowledge base")

if __name__ == "__main__":
    # main()
    test_knowledge_retrieval()