"""
LangChain + Pinecone RAG Advisor for hyperparameter optimization.
Preserves the TrainingAdvisor API used by the workflow while switching to LangChain chains.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain / Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
try:
    # Prefer new location
    from langchain_core.documents import Document
except Exception:  # fallback for older versions
    from langchain.schema import Document

# Prompts (simplified, rely on RAG)
try:
    from prompts import (
        HYPERPARAMETER_ADVISOR_SYSTEM_PROMPT,
        get_hyperparameter_prompt,
        OPTIMIZATION_SYSTEM_PROMPT,
        get_optimization_prompt,
    )
except ImportError:
    # Fallback for relative import
    from .prompts import (
        HYPERPARAMETER_ADVISOR_SYSTEM_PROMPT,
        get_hyperparameter_prompt,
        OPTIMIZATION_SYSTEM_PROMPT,
        get_optimization_prompt,
    )

# Import the LLM logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_llm_logger

logger = logging.getLogger(__name__)


class TrainingAdvisor:
    """AI-powered training advisor using LangChain RAG."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", session_id: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.model = model
        self.session_id = session_id
        self.llm_logger = setup_llm_logger(session_id=session_id)

        # LLM and embeddings
        self.llm = ChatOpenAI(model=self.model, temperature=0.2, api_key=self.api_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512, api_key=self.api_key)

        # Vector store / retriever (Pinecone)
        knowledge_index_name = os.getenv("PINECONE_KNOWLEDGE_INDEX_NAME", "curate-knowledge")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            self.llm_logger.warning("⚠️ RAG: PINECONE_API_KEY not found - RAG disabled")
            self.vector_store = None
            self.retriever = None
        else:
            try:
                self.llm_logger.info(f"🔗 RAG: Initializing Pinecone vector store with index '{knowledge_index_name}'")
                self.vector_store = PineconeVectorStore(
                    index_name=knowledge_index_name,
                    embedding=self.embeddings,
                    pinecone_api_key=pinecone_api_key,
                )
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
                self.llm_logger.info("✅ RAG: Successfully initialized Pinecone vector store and retriever (k=5)")
                
                # Test retriever connectivity only if we have OpenAI API key too
                if self.api_key:
                    try:
                        test_query = "test connection"
                        test_results = self.retriever.invoke(test_query)
                        self.llm_logger.info(f"✅ RAG: Retriever connectivity test successful - found {len(test_results)} documents")
                    except Exception as test_e:
                        self.llm_logger.warning(f"⚠️ RAG: Retriever connectivity test failed: {str(test_e)}")
                else:
                    self.llm_logger.warning("⚠️ RAG: Skipping connectivity test - no OpenAI API key for embeddings")
                    
            except Exception as e:
                self.llm_logger.error(f"❌ RAG: Failed to initialize Pinecone vector store: {str(e)}")
                self.vector_store = None
                self.retriever = None

        # Chains
        self.llm_logger.info("🔧 RAG: Setting up LangChain chains with RAG context integration")
        self._setup_chains()
        self.llm_logger.info("✅ RAG: LangChain chains setup complete")

    # ---------- Chain setup ----------
    def _format_docs(self, docs: List[Document]) -> str:
        """Format RAG documents with comprehensive logging."""
        self.llm_logger.info(f"📄 RAG: Formatting {len(docs)} retrieved documents")
        
        parts = []
        categories_found = {}
        total_content_length = 0
        
        for i, d in enumerate(docs):
            meta = d.metadata or {}
            category = meta.get("category", "general")
            topic = meta.get("topic", "unknown")
            content_length = len(d.page_content)
            total_content_length += content_length
            
            # Track categories for summary
            if category not in categories_found:
                categories_found[category] = 0
            categories_found[category] += 1
            
            formatted_doc = f"[{category}/{topic}] {d.page_content}"
            parts.append(formatted_doc)
            
            self.llm_logger.debug(f"📄 RAG Doc {i+1}: category='{category}', topic='{topic}', length={content_length}chars")
            
            # Log document metadata if available
            if meta:
                meta_summary = {k: v for k, v in meta.items() if k not in ['category', 'topic']}
                if meta_summary:
                    self.llm_logger.debug(f"📄 RAG Doc {i+1} metadata: {meta_summary}")
        
        # Log summary statistics
        self.llm_logger.info(f"📄 RAG Summary: {len(docs)} docs, {total_content_length} total chars")
        self.llm_logger.info(f"📄 RAG Categories found: {dict(categories_found)}")
        
        formatted_context = "\n\n".join(parts)
        self.llm_logger.debug(f"📄 RAG Final context length: {len(formatted_context)} characters")
        
        return formatted_context

    def _setup_chains(self) -> None:
        parser = JsonOutputParser()

        # Hyperparameter chain - expects dataset_info and current_config
        hyper_prompt = ChatPromptTemplate.from_template("""
You are an expert ML engineer. Use the provided context to recommend optimal hyperparameters.

Context from knowledge base:
{context}

Dataset Information:
{dataset_info}

Current Configuration:
{current_config}

Provide hyperparameter recommendations in valid JSON format with detailed reasoning.
Focus on the most impactful optimizations based on the dataset characteristics.

Required JSON structure:
{{
  "analysis": {{
    "dataset_complexity": "low|medium|high",
    "recommended_approach": "single_stage|dual_stage",
    "key_insights": ["insight1", "insight2"]
  }},
  "hyperparameters": {{
    "training_config": {{
      "batch_size": {{"value": 32, "reasoning": "explanation"}},
      "initial_learning_rate": {{"value": 0.001, "reasoning": "explanation"}},
      "initial_epochs": {{"value": 20, "reasoning": "explanation"}},
      "image_size": {{"value": [224, 224], "reasoning": "explanation"}},
      "dual_stage": {{"value": true, "reasoning": "explanation"}}
    }},
    "fine_tuning_config": {{
      "fine_tune_learning_rate": {{"value": 0.0001, "reasoning": "explanation"}},
      "fine_tune_epochs": {{"value": 10, "reasoning": "explanation"}},
      "unfreeze_percent": {{"value": 0.5, "reasoning": "explanation"}}
    }}
  }}
}}
""")

        # Context retrieval function
        def get_context_for_query(inputs):
            if self.retriever:
                # Create a search query from the inputs
                dataset_preview = str(inputs.get('dataset_info', ''))[:200]
                search_query = f"hyperparameter optimization {dataset_preview}"
                
                self.llm_logger.info(f"🔍 RAG: Searching for hyperparameter context")
                self.llm_logger.debug(f"🔍 RAG Query: '{search_query[:100]}...' (truncated)")
                
                try:
                    docs = self.retriever.invoke(search_query)
                    self.llm_logger.info(f"🔍 RAG: Retrieved {len(docs)} documents for hyperparameter query")
                    
                    # Log document scores if available
                    for i, doc in enumerate(docs):
                        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                            score = doc.metadata['score']
                            self.llm_logger.debug(f"🔍 RAG Doc {i+1} similarity score: {score:.4f}")
                    
                    return self._format_docs(docs)
                    
                except Exception as e:
                    self.llm_logger.error(f"❌ RAG: Failed to retrieve hyperparameter context: {str(e)}")
                    return "No relevant context available due to retrieval error."
            else:
                self.llm_logger.warning("⚠️ RAG: No retriever available for hyperparameter context")
                return "No RAG context available - retriever not initialized."

        self.hyperparam_chain = (
            {
                "context": get_context_for_query,
                "dataset_info": lambda x: x["dataset_info"],
                "current_config": lambda x: x["current_config"]
            }
            | hyper_prompt
            | self.llm
            | parser
        )

        # Optimization chain - expects training_log and current_config
        opt_prompt = ChatPromptTemplate.from_template("""
You are an ML optimization expert. Use the context to analyze training issues and recommend improvements.

Context from knowledge base:
{context}

Training Log Analysis:
{training_log}

Current Configuration:
{current_config}

Analyze the training performance and recommend SUBSTANTIAL optimizations that will significantly impact results.
Provide your analysis and recommendations in valid JSON format.

Focus on hyperparameter optimization first. Only recommend architecture changes if absolutely necessary.
""")

        def get_context_for_optimization(inputs):
            if self.retriever:
                # Create a search query from the training log
                training_preview = str(inputs.get('training_log', ''))[:200]
                search_query = f"training optimization {training_preview}"
                
                self.llm_logger.info(f"🔍 RAG: Searching for optimization context")
                self.llm_logger.debug(f"🔍 RAG Query: '{search_query[:100]}...' (truncated)")
                
                try:
                    docs = self.retriever.invoke(search_query)
                    self.llm_logger.info(f"🔍 RAG: Retrieved {len(docs)} documents for optimization query")
                    
                    # Log document scores if available
                    for i, doc in enumerate(docs):
                        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                            score = doc.metadata['score']
                            self.llm_logger.debug(f"🔍 RAG Doc {i+1} similarity score: {score:.4f}")
                    
                    return self._format_docs(docs)
                    
                except Exception as e:
                    self.llm_logger.error(f"❌ RAG: Failed to retrieve optimization context: {str(e)}")
                    return "No relevant context available due to retrieval error."
            else:
                self.llm_logger.warning("⚠️ RAG: No retriever available for optimization context")
                return "No RAG context available - retriever not initialized."

        self.optimization_chain = (
            {
                "context": get_context_for_optimization,
                "training_log": lambda x: x["training_log"],
                "current_config": lambda x: x["current_config"]
            }
            | opt_prompt
            | self.llm
            | parser
        )

    # ---------- Helpers ----------
    def _extract_value(self, param_dict: Dict[str, Any]):
        """
        Extract value from AI recommendation parameter dictionary.
        Expects "value" key as specified in the prompt.
        """
        if not isinstance(param_dict, dict):
            return param_dict  # If it's already a value, return it

        if "value" in param_dict:
            value = param_dict["value"]

            # Type conversion for common parameter types
            if isinstance(value, str):
                # Try to convert string numbers to appropriate numeric types
                if value.isdigit():
                    return int(value)
                try:
                    return float(value)
                except ValueError:
                    pass

            return value

        # Log available keys for debugging if "value" is missing
        logger.warning(f"Expected 'value' key not found in parameter dict. Available keys: {list(param_dict.keys())}")
        return None
        
    def extract_dataset_info(self, data_parser, trainer) -> Dict[str, Any]:
        """
        Extract relevant information from ImgClassData and ImgClassTrainer.
        
        Args:
            data_parser: ImgClassData instance
            trainer: ImgClassTrainer instance
            
        Returns:
            Dictionary containing dataset characteristics
        """
        try:
            # Extract basic dataset info
            dataset_info = {
                "dataset_path": data_parser.filepath,
                "total_images": data_parser.total_images if hasattr(data_parser, 'total_images') else "unknown",
                "num_classes": len(data_parser.classes),
                "class_names": data_parser.classes,
                "image_dimensions": {
                    "original": data_parser.IMSIZE,
                    "processed": trainer.IMG_SIZE
                },
                "file_tree_structure": data_parser.json_tree,
                "directory_structure": {
                    "train_dir": data_parser.train_dir,
                    "val_dir": data_parser.val_dir,
                    "test_dir": data_parser.test_dir
                }
            }
            
            # Calculate class distribution if possible
            try:
                class_distribution = {}
                if hasattr(data_parser, 'json_tree') and data_parser.json_tree:
                    for class_name in data_parser.classes:
                        class_path = os.path.join(data_parser.train_dir, class_name)
                        if os.path.exists(class_path):
                            class_count = len([f for f in os.listdir(class_path) 
                                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                            class_distribution[class_name] = class_count
                
                dataset_info["class_distribution"] = class_distribution
                dataset_info["total_training_images"] = sum(class_distribution.values()) if class_distribution else "unknown"
                
                # Calculate class balance metrics
                if class_distribution:
                    counts = list(class_distribution.values())
                    dataset_info["class_balance"] = {
                        "min_class_size": min(counts),
                        "max_class_size": max(counts),
                        "mean_class_size": sum(counts) / len(counts),
                        "imbalance_ratio": max(counts) / min(counts) if min(counts) > 0 else "infinite"
                    }
            except Exception as e:
                logger.warning(f"Could not calculate class distribution: {str(e)}")
                dataset_info["class_distribution"] = "calculation_failed"
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Failed to extract dataset info: {str(e)}")
            return {"error": str(e)}
    
    def get_current_config(self, trainer) -> Dict[str, Any]:
        """
        Extract current training configuration from trainer.
        
        Args:
            trainer: ImgClassTrainer instance
            
        Returns:
            Dictionary containing current configuration
        """
        try:
            return {
                "base_model_name": trainer.base_model_name,
                "batch_size": trainer.batch_size,
                "initial_learning_rate": trainer.initial_learning_rate,
                "fine_tune_learning_rate": trainer.fine_tune_learning_rate,
                "initial_epochs": trainer.initial_epochs,
                "fine_tune_epochs": trainer.fine_tune_epochs,
                "dual_stage": trainer.dual_stage,
                "custom_img_size": trainer.custom_img_size,
                "img_size_used": trainer.IMG_SIZE,
                "unfreeze_percent": trainer.unfreeze_percent,
                "num_classes": trainer.NUM_CLASSES
            }
        except Exception as e:
            logger.error(f"Failed to extract current config: {str(e)}")
            return {"error": str(e)}
    
    # ---------- LangChain-powered recommendation methods ----------
    def recommend_hyperparameters(self, dataset_info: str, current_config: dict) -> dict:
        """Recommend hyperparameters using LangChain RAG."""
        self.llm_logger.info("🚀 RAG: Starting hyperparameter recommendation with LangChain")
        
        try:
            if not self.hyperparam_chain:
                self.llm_logger.warning("⚠️ RAG: Chain not initialized, using fallback response")
                return self._fallback_hyperparameters()

            # Log input details
            dataset_preview = dataset_info[:200] + "..." if len(dataset_info) > 200 else dataset_info
            self.llm_logger.info(f"📊 RAG: Dataset info length: {len(dataset_info)} chars")
            self.llm_logger.debug(f"📊 RAG: Dataset preview: {dataset_preview}")
            self.llm_logger.info(f"⚙️ RAG: Current config keys: {list(current_config.keys())}")

            # Ensure inputs are properly formatted
            inputs = {
                "dataset_info": str(dataset_info),
                "current_config": str(current_config)  # Convert dict to string
            }
            
            # Validate inputs structure
            if not isinstance(inputs, dict):
                self.llm_logger.error("❌ RAG: Inputs must be a dictionary")
                return self._fallback_hyperparameters()
            
            if "dataset_info" not in inputs or "current_config" not in inputs:
                self.llm_logger.error("❌ RAG: Missing required input keys")
                return self._fallback_hyperparameters()
            
            self.llm_logger.info("🔗 RAG: Invoking hyperparameter chain with RAG context...")
            self.llm_logger.debug(f"🔗 RAG: Input keys: {list(inputs.keys())}")
            
            result = self.hyperparam_chain.invoke(inputs)
            
            if isinstance(result, dict):
                self.llm_logger.info("✅ RAG: Successfully received hyperparameter recommendations")
                self.llm_logger.debug(f"✅ RAG: Result keys: {list(result.keys())}")
                
                # Log analysis if available
                if "analysis" in result:
                    analysis = result["analysis"]
                    complexity = analysis.get("dataset_complexity", "unknown")
                    approach = analysis.get("recommended_approach", "unknown")
                    self.llm_logger.info(f"📊 RAG Analysis: complexity={complexity}, approach={approach}")
                
                return result
            else:
                self.llm_logger.warning(f"⚠️ RAG: Unexpected result type: {type(result)}")
                return self._fallback_hyperparameters()
                
        except TypeError as te:
            self.llm_logger.error(f"❌ RAG: Type error in LangChain hyperparameter chain: {te}")
            self.llm_logger.error(f"❌ RAG: This often indicates incorrect input format to the chain")
            return self._fallback_hyperparameters()
        except Exception as e:
            self.llm_logger.error(f"❌ RAG: Error in LangChain hyperparameter chain: {e}")
            self.llm_logger.error(f"❌ RAG: Exception type: {type(e).__name__}")
            return self._fallback_hyperparameters()

    def recommend_optimizations(self, training_log: str, current_config: dict) -> dict:
        """Recommend optimizations using LangChain RAG."""
        self.llm_logger.info("🚀 RAG: Starting optimization recommendation with LangChain")
        
        try:
            if not self.optimization_chain:
                self.llm_logger.warning("⚠️ RAG: Optimization chain not initialized, using fallback")
                return self._fallback_optimizations()

            # Log input details
            log_preview = training_log[:200] + "..." if len(training_log) > 200 else training_log
            self.llm_logger.info(f"📊 RAG: Training log length: {len(training_log)} chars")
            self.llm_logger.debug(f"📊 RAG: Training log preview: {log_preview}")
            self.llm_logger.info(f"⚙️ RAG: Current config keys: {list(current_config.keys())}")

            # Ensure inputs are properly formatted
            inputs = {
                "training_log": str(training_log),
                "current_config": str(current_config)  # Convert dict to string
            }
            
            # Validate inputs structure
            if not isinstance(inputs, dict):
                self.llm_logger.error("❌ RAG: Inputs must be a dictionary")
                return self._fallback_optimizations()
            
            if "training_log" not in inputs or "current_config" not in inputs:
                self.llm_logger.error("❌ RAG: Missing required input keys")
                return self._fallback_optimizations()
            
            self.llm_logger.info("🔗 RAG: Invoking optimization chain with RAG context...")
            self.llm_logger.debug(f"🔗 RAG: Input keys: {list(inputs.keys())}")
            
            result = self.optimization_chain.invoke(inputs)
            
            if isinstance(result, dict):
                self.llm_logger.info("✅ RAG: Successfully received optimization recommendations")
                self.llm_logger.debug(f"✅ RAG: Result keys: {list(result.keys())}")
                
                # Log analysis if available
                if "analysis" in result:
                    analysis = result["analysis"]
                    if "key_issues" in analysis:
                        issues = analysis["key_issues"]
                        self.llm_logger.info(f"📊 RAG Analysis: Found {len(issues)} key issues")
                
                return result
            else:
                self.llm_logger.warning(f"⚠️ RAG: Unexpected optimization result type: {type(result)}")
                return self._fallback_optimizations()
                
        except TypeError as te:
            self.llm_logger.error(f"❌ RAG: Type error in LangChain optimization chain: {te}")
            self.llm_logger.error(f"❌ RAG: This often indicates incorrect input format to the chain")
            return self._fallback_optimizations()
        except Exception as e:
            self.llm_logger.error(f"❌ RAG: Error in LangChain optimization chain: {e}")
            self.llm_logger.error(f"❌ RAG: Exception type: {type(e).__name__}")
            return self._fallback_optimizations()

    def _fallback_hyperparameters(self) -> dict:
        """Fallback hyperparameter recommendations when chain fails."""
        self.llm_logger.warning("🔄 RAG: Using fallback hyperparameter recommendations")
        return {
            "analysis": {
                "dataset_complexity": "medium",
                "recommended_approach": "dual_stage",
                "key_insights": ["Using fallback recommendations due to chain failure"]
            },
            "hyperparameters": {
                "training_config": {
                    "batch_size": {"value": 32, "reasoning": "Conservative default for medium datasets"},
                    "initial_learning_rate": {"value": 0.001, "reasoning": "Standard learning rate for transfer learning"},
                    "initial_epochs": {"value": 20, "reasoning": "Sufficient epochs for initial training"},
                    "image_size": {"value": [224, 224], "reasoning": "Standard ImageNet input size"},
                    "dual_stage": {"value": True, "reasoning": "Recommended for most image classification tasks"}
                },
                "fine_tuning_config": {
                    "fine_tune_learning_rate": {"value": 0.0001, "reasoning": "Lower learning rate for fine-tuning"},
                    "fine_tune_epochs": {"value": 10, "reasoning": "Conservative epochs for fine-tuning"},
                    "unfreeze_percent": {"value": 0.5, "reasoning": "Unfreeze half the layers for fine-tuning"}
                }
            }
        }

    def _fallback_optimizations(self) -> dict:
        """Fallback optimization recommendations when chain fails."""
        self.llm_logger.warning("🔄 RAG: Using fallback optimization recommendations")
        return {
            "analysis": {
                "training_performance": "unknown",
                "key_issues": ["Unable to analyze due to chain failure"],
                "recommendations": ["Manual hyperparameter tuning recommended"]
            },
            "optimization_recommendations": {
                "training_config": {
                    "batch_size": {"recommended_value": 32, "reasoning": "Conservative default"},
                    "initial_learning_rate": {"recommended_value": 0.001, "reasoning": "Standard transfer learning rate"}
                }
            }
        }

    # ---------- RAG-powered tasks ----------
    def _invoke_hyperparams_chain(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        try:
            self.llm_logger.info("🔗 RAG: Invoking LangChain hyperparameter chain with RAG context")
            self.llm_logger.debug(f"🔗 RAG: Prompt length: {len(user_prompt)} characters")
            
            result = self.hyperparam_chain.invoke(user_prompt)
            
            if result:
                self.llm_logger.info("✅ RAG: Hyperparameter chain completed successfully")
                return result
            else:
                self.llm_logger.warning("⚠️ RAG: Hyperparameter chain returned empty result")
                return None
                
        except Exception as e:
            self.llm_logger.error(f"❌ RAG: LangChain hyperparam chain failed: {str(e)}")
            self.llm_logger.error(f"❌ RAG: Exception type: {type(e).__name__}")
            return None

    def _invoke_optimization_chain(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        try:
            self.llm_logger.info("🔗 RAG: Invoking LangChain optimization chain with RAG context")
            self.llm_logger.debug(f"🔗 RAG: Prompt length: {len(user_prompt)} characters")
            
            result = self.optimization_chain.invoke(user_prompt)
            
            if result:
                self.llm_logger.info("✅ RAG: Optimization chain completed successfully")
                return result
            else:
                self.llm_logger.warning("⚠️ RAG: Optimization chain returned empty result")
                return None
                
        except Exception as e:
            self.llm_logger.error(f"❌ RAG: LangChain optimization chain failed: {str(e)}")
            self.llm_logger.error(f"❌ RAG: Exception type: {type(e).__name__}")
            return None
            
    
    def _validate_optimization_response(self, recommendations: Dict[str, Any]) -> None:
        """Validate the structure and types of optimization recommendations."""
        if "training_config" in recommendations:
            for param, details in recommendations["training_config"].items():
                if isinstance(details, dict) and "recommended_value" in details:
                    value = details["recommended_value"]
                    if param in ["batch_size", "initial_epochs", "fine_tune_epochs"]:
                        if not isinstance(value, int):
                            raise ValueError(f"Parameter {param} must be integer, got {type(value)}: {value}")
                    elif param in ["initial_learning_rate", "fine_tune_learning_rate", "unfreeze_percent"]:
                        if not isinstance(value, (int, float)):
                            raise ValueError(f"Parameter {param} must be number, got {type(value)}: {value}")
                    elif param == "image_size":
                        if not isinstance(value, list) or len(value) != 2:
                            raise ValueError(f"Parameter {param} must be list of 2 integers, got {type(value)}: {value}")
                    elif param == "base_model_name":
                        if not isinstance(value, str):
                            raise ValueError(f"Parameter {param} must be string, got {type(value)}: {value}")
    
    def get_hyperparameter_recommendations(self, data_parser, trainer) -> Optional[Dict[str, Any]]:
        """
        Get AI-powered hyperparameter recommendations.
        
        Args:
            data_parser: ImgClassData instance
            trainer: ImgClassTrainer instance
            
        Returns:
            Dictionary containing recommendations or None if failed
        """
        self.llm_logger.info("🚀 RAG: Starting hyperparameter recommendation workflow")
        
        self.llm_logger.info("📊 RAG: Extracting dataset information for AI analysis...")
        dataset_info = self.extract_dataset_info(data_parser, trainer)
        current_config = self.get_current_config(trainer)

        # Log extracted info summary
        if isinstance(dataset_info, dict):
            self.llm_logger.info(f"📊 RAG: Dataset extracted - classes: {dataset_info.get('num_classes', 'unknown')}, images: {dataset_info.get('total_images', 'unknown')}")
        if isinstance(current_config, dict):
            self.llm_logger.info(f"⚙️ RAG: Config extracted - model: {current_config.get('base_model_name', 'unknown')}, batch_size: {current_config.get('batch_size', 'unknown')}")

        self.llm_logger.info("🔗 RAG: Calling LangChain RAG for hyperparameter recommendations...")
        user_prompt = get_hyperparameter_prompt(
            json.dumps(dataset_info, indent=2),
            json.dumps(current_config, indent=2),
        )

        self.llm_logger.debug(f"🔗 RAG: Generated prompt length: {len(user_prompt)} characters")
        recommendations = self._invoke_hyperparams_chain(user_prompt)
        
        if recommendations:
            self.llm_logger.info("✅ RAG: Successfully received AI recommendations")
            if isinstance(recommendations, dict) and "analysis" in recommendations:
                analysis = recommendations["analysis"]
                self.llm_logger.info(f"📊 RAG: Recommended complexity: {analysis.get('dataset_complexity', 'unknown')}")
                self.llm_logger.info(f"📊 RAG: Recommended approach: {analysis.get('recommended_approach', 'unknown')}")
            return recommendations
        else:
            self.llm_logger.error("❌ RAG: Failed to get recommendations from AI advisor")
            return None
    
    def save_recommendations(self, recommendations: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """
        Save recommendations to JSON file.
        
        Args:
            recommendations: Recommendations dictionary
            filepath: Optional custom filepath
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"ai_recommendations_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            logger.info(f"AI recommendations saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save recommendations: {str(e)}")
            raise
    
    def apply_recommendations(self, trainer, recommendations: Dict[str, Any]) -> bool:
        """
        Apply AI recommendations to trainer configuration.
        
        Args:
            trainer: ImgClassTrainer instance
            recommendations: AI recommendations dictionary
            
        Returns:
            True if successfully applied, False otherwise
        """
        try:
            # Debug: Log the structure of recommendations
            logger.info(f"Recommendations structure: {list(recommendations.keys())}")
            
            # Check if recommendations have the expected structure
            if "hyperparameters" in recommendations:
                params = recommendations["hyperparameters"]
                logger.info("Using 'hyperparameters' structure")
            elif "ai_recommendations" in recommendations:
                # Handle the structure from format_recommendations_for_logging
                params = recommendations["ai_recommendations"]
                logger.info("Using 'ai_recommendations' structure")
            else:
                logger.error("No hyperparameters or ai_recommendations found in recommendations")
                logger.error(f"Available keys: {list(recommendations.keys())}")
                return False
            
            logger.info(f"Params structure: {list(params.keys()) if isinstance(params, dict) else 'Not a dict'}")
            
            # Apply training config
            if "training_config" in params:
                config = params["training_config"]
                
                if "batch_size" in config:
                    value = self._extract_value(config["batch_size"])
                    logger.info(f"Raw batch_size from AI: {config['batch_size']}")
                    logger.info(f"Extracted batch_size value: {value} (type: {type(value)})")

                    # Ensure batch_size is a valid integer
                    if value:
                        try:
                            trainer.batch_size = int(value)
                            logger.info(f"Updated batch_size to: {trainer.batch_size} (type: {type(trainer.batch_size)})")
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid batch_size value: {value}, keeping original value")
                            logger.error(f"Error: {e}")
                    else:
                        logger.warning("No valid batch_size value extracted from AI recommendations")
                
                if "initial_learning_rate" in config:
                    value = self._extract_value(config["initial_learning_rate"])
                    if value:
                        trainer.initial_learning_rate = value
                        logger.info(f"Updated initial_learning_rate to: {trainer.initial_learning_rate}")
                
                if "initial_epochs" in config:
                    value = self._extract_value(config["initial_epochs"])
                    if value:
                        trainer.initial_epochs = value
                        logger.info(f"Updated initial_epochs to: {trainer.initial_epochs}")
                
                if "image_size" in config:
                    value = self._extract_value(config["image_size"])
                    logger.info(f"Raw image_size from AI: {config['image_size']}")
                    logger.info(f"Extracted image_size value: {value} (type: {type(value)})")

                    # Ensure image_size is a valid tuple/list of two integers
                    if value:
                        try:
                            if isinstance(value, (list, tuple)) and len(value) == 2:
                                trainer.custom_img_size = tuple(int(x) for x in value)
                                trainer.IMG_SIZE = trainer.custom_img_size
                                logger.info(f"Updated image_size to: {trainer.IMG_SIZE}")
                            else:
                                logger.warning(f"Invalid image_size format: {value}, expected [width, height]")
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid image_size value: {value}, keeping original value")
                            logger.error(f"Error: {e}")
                    else:
                        logger.warning("No valid image_size value extracted from AI recommendations")
                
                if "dual_stage" in config:
                    value = self._extract_value(config["dual_stage"])
                    logger.info(f"Raw dual_stage from AI: {config['dual_stage']}")
                    logger.info(f"Extracted dual_stage value: {value} (type: {type(value)})")

                    # Ensure dual_stage is a valid boolean
                    if value is not None:
                        try:
                            if isinstance(value, bool):
                                trainer.dual_stage = value
                                logger.info(f"Updated dual_stage to: {trainer.dual_stage}")
                            elif isinstance(value, str):
                                # Handle string representations of boolean
                                if value.lower() in ['true', '1', 'yes']:
                                    trainer.dual_stage = True
                                elif value.lower() in ['false', '0', 'no']:
                                    trainer.dual_stage = False
                                else:
                                    logger.warning(f"Invalid dual_stage string value: {value}, keeping original value")
                                logger.info(f"Updated dual_stage to: {trainer.dual_stage}")
                            else:
                                logger.warning(f"Invalid dual_stage type: {type(value)}, expected boolean")
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid dual_stage value: {value}, keeping original value")
                            logger.error(f"Error: {e}")
                    else:
                        logger.warning("No valid dual_stage value extracted from AI recommendations")
            
            # Apply fine-tuning config
            if "fine_tuning_config" in params:
                config = params["fine_tuning_config"]
                
                if "fine_tune_learning_rate" in config:
                    value = self._extract_value(config["fine_tune_learning_rate"])
                    if value:
                        trainer.fine_tune_learning_rate = value
                        logger.info(f"Updated fine_tune_learning_rate to: {trainer.fine_tune_learning_rate}")
                
                if "fine_tune_epochs" in config:
                    value = self._extract_value(config["fine_tune_epochs"])
                    if value:
                        trainer.fine_tune_epochs = value
                        logger.info(f"Updated fine_tune_epochs to: {trainer.fine_tune_epochs}")
                
                if "unfreeze_percent" in config:
                    value = self._extract_value(config["unfreeze_percent"])
                    if value:
                        trainer.unfreeze_percent = value
                        logger.info(f"Updated unfreeze_percent to: {trainer.unfreeze_percent}")
            
            # Apply model architecture
            if "model_architecture" in params:
                model_config = params["model_architecture"]
                if "base_model" in model_config:
                    value = self._extract_value(model_config["base_model"])
                    if value:
                        trainer.base_model_name = value
                        logger.info(f"Updated base_model_name to: {trainer.base_model_name}")
                elif "recommended_model" in model_config:
                    trainer.base_model_name = model_config["recommended_model"]
                    logger.info(f"Updated base_model_name to: {trainer.base_model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply recommendations: {str(e)}")
            return False
    
    def optimize(self, trainer) -> Optional[Dict[str, Any]]:
        """
        Analyze training logs and suggest optimized hyperparameters for better performance.
        
        Args:
            trainer: ImgClassTrainer instance with training history
            
        Returns:
            Dictionary containing optimization recommendations or None if failed
        """
        try:
            # Extract training log data
            if not hasattr(trainer, 'training_log') or trainer.training_log is None:
                logger.error("No training log found in trainer")
                return None
            
            # Get the training history and current configuration
            training_log = trainer.training_log.get_log_data()
            current_config = self.get_current_config(trainer)
            
            logger.info("Analyzing training performance for optimization...")
            
            # Create optimization prompt and invoke chain
            optimization_prompt = get_optimization_prompt(
                json.dumps(training_log, indent=2),
                json.dumps(current_config, indent=2),
            )

            recommendations = self._invoke_optimization_chain(optimization_prompt)
            
            if recommendations:
                logger.info("Successfully received optimization recommendations")
                
                # Apply recommendations using trainer.edit_config to maintain training log updates
                if "optimization_recommendations" in recommendations:
                    success = self._apply_optimization_recommendations(trainer, recommendations["optimization_recommendations"])
                    if not success:
                        logger.error("Failed to apply optimization recommendations")
                        return None
                
                return recommendations
            else:
                logger.error("Failed to get optimization recommendations from AI")
                return None
                
        except Exception as e:
            logger.error(f"Failed to optimize training: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            return None
    
    def _apply_optimization_recommendations(self, trainer, recommendations: Dict[str, Any]) -> bool:
        """
        Apply optimization recommendations using trainer.edit_config to maintain log updates.
        
        Args:
            trainer: ImgClassTrainer instance
            recommendations: Optimization recommendations dictionary
            
        Returns:
            True if successfully applied, False otherwise
        """
        try:
            changes_applied = {}
            
            # Collect all parameters for edit_config
            config_params = {}
            
            # Get current values as defaults
            config_params['base_model_name'] = trainer.base_model_name
            config_params['batch_size'] = trainer.batch_size
            config_params['initial_learning_rate'] = trainer.initial_learning_rate
            config_params['fine_tune_learning_rate'] = trainer.fine_tune_learning_rate
            config_params['initial_epochs'] = trainer.initial_epochs
            config_params['fine_tune_epochs'] = trainer.fine_tune_epochs
            config_params['dual_stage'] = trainer.dual_stage
            config_params['custom_img_size'] = trainer.custom_img_size
            config_params['unfreeze_percent'] = trainer.unfreeze_percent
            
            # Update with recommendations from training_strategy
            if "training_strategy" in recommendations:
                for param, details in recommendations["training_strategy"].items():
                    if isinstance(details, dict) and "recommended_value" in details:
                        old_value = config_params.get(param)
                        new_value = details["recommended_value"]
                        
                        # Type validation and conversion
                        try:
                            if param == "dual_stage":
                                new_value = bool(new_value)
                                config_params['dual_stage'] = new_value
                                changes_applied[param] = {
                                    "old_value": old_value,
                                    "new_value": new_value,
                                    "reasoning": details.get("reasoning", "N/A")
                                }
                        except (ValueError, TypeError) as e:
                            logger.error(f"Type conversion error for {param}: {new_value} -> {str(e)}")
                            continue

            # Update with recommendations from training_config
            if "training_config" in recommendations:
                for param, details in recommendations["training_config"].items():
                    if isinstance(details, dict) and "recommended_value" in details:
                        old_value = config_params.get(param)
                        new_value = details["recommended_value"]
                        
                        # Type validation and conversion
                        try:
                            if param == "batch_size":
                                new_value = int(float(str(new_value).strip()))
                                if new_value <= 0:
                                    logger.warning(f"Invalid batch_size {new_value}, skipping")
                                    continue
                            elif param == "initial_epochs":
                                new_value = int(float(str(new_value).strip()))
                                if new_value <= 0:
                                    logger.warning(f"Invalid initial_epochs {new_value}, skipping")
                                    continue
                            elif param == "fine_tune_epochs":
                                new_value = int(float(str(new_value).strip()))
                                if new_value <= 0:
                                    logger.warning(f"Invalid fine_tune_epochs {new_value}, skipping")
                                    continue
                            elif param == "initial_learning_rate":
                                new_value = float(str(new_value).strip())
                                if new_value <= 0 or new_value > 1:
                                    logger.warning(f"Invalid initial_learning_rate {new_value}, skipping")
                                    continue
                            elif param == "fine_tune_learning_rate":
                                new_value = float(str(new_value).strip())
                                if new_value <= 0 or new_value > 1:
                                    logger.warning(f"Invalid fine_tune_learning_rate {new_value}, skipping")
                                    continue
                            elif param == "unfreeze_percent":
                                new_value = float(str(new_value).strip())
                                if new_value < 0 or new_value > 1:
                                    logger.warning(f"Invalid unfreeze_percent {new_value}, skipping")
                                    continue
                            elif param == "image_size":
                                if isinstance(new_value, list) and len(new_value) == 2:
                                    new_value = [int(float(str(x).strip())) for x in new_value]
                                else:
                                    logger.warning(f"Invalid image_size format {new_value}, skipping")
                                    continue
                            elif param == "base_model_name":
                                new_value = str(new_value).strip()
                                if not new_value.startswith("EfficientNet"):
                                    logger.warning(f"Invalid base_model_name {new_value}, skipping")
                                    continue
                        except (ValueError, TypeError) as e:
                            logger.error(f"Type conversion error for {param}: {new_value} -> {str(e)}")
                            continue
                        
                        # Special handling for image_size -> custom_img_size
                        if param == "image_size":
                            old_custom_img_size = config_params.get('custom_img_size')
                            config_params['custom_img_size'] = tuple(new_value) if isinstance(new_value, list) else new_value
                            changes_applied['custom_img_size'] = {
                                "old_value": old_custom_img_size,
                                "new_value": new_value,
                                "reasoning": details.get("reasoning", "N/A")
                            }
                        else:
                            config_params[param] = new_value
                            changes_applied[param] = {
                                "old_value": old_value,
                                "new_value": new_value,
                                "reasoning": details.get("reasoning", "N/A")
                            }
            
            # Update with recommendations from fine_tuning_config
            if "fine_tuning_config" in recommendations:
                for param, details in recommendations["fine_tuning_config"].items():
                    if isinstance(details, dict) and "recommended_value" in details:
                        old_value = config_params.get(param)
                        new_value = details["recommended_value"]
                        
                        # Type validation and conversion
                        try:
                            if param == "fine_tune_epochs":
                                new_value = int(float(str(new_value).strip()))
                                if new_value <= 0:
                                    logger.warning(f"Invalid fine_tune_epochs {new_value}, skipping")
                                    continue
                            elif param == "fine_tune_learning_rate":
                                new_value = float(str(new_value).strip())
                                if new_value <= 0 or new_value > 1:
                                    logger.warning(f"Invalid fine_tune_learning_rate {new_value}, skipping")
                                    continue
                            elif param == "unfreeze_percent":
                                new_value = float(str(new_value).strip())
                                if new_value < 0 or new_value > 1:
                                    logger.warning(f"Invalid unfreeze_percent {new_value}, skipping")
                                    continue
                        except (ValueError, TypeError) as e:
                            logger.error(f"Type conversion error for {param}: {new_value} -> {str(e)}")
                            continue
                        
                        config_params[param] = new_value
                        changes_applied[param] = {
                            "old_value": old_value,
                            "new_value": new_value,
                            "reasoning": details.get("reasoning", "N/A")
                        }
            
            # Update with recommendations from model_architecture
            if "model_architecture" in recommendations:
                for param, details in recommendations["model_architecture"].items():
                    if isinstance(details, dict) and "recommended_value" in details:
                        old_value = config_params.get(param)
                        new_value = details["recommended_value"]
                        # Map base_model_name parameter
                        if param == "base_model_name":
                            config_params['base_model_name'] = new_value
                            changes_applied[param] = {
                                "old_value": old_value,
                                "new_value": new_value,
                                "reasoning": details.get("reasoning", "N/A")
                            }
            
            # Handle recommendations from analysis section (dual_stage from recommended_approach)
            if "analysis" in recommendations and "recommended_approach" in recommendations["analysis"]:
                approach = recommendations["analysis"]["recommended_approach"]
                old_dual_stage = config_params.get('dual_stage')
                new_dual_stage = (approach == "dual_stage")
                if old_dual_stage != new_dual_stage:
                    config_params['dual_stage'] = new_dual_stage
                    changes_applied['dual_stage'] = {
                        "old_value": old_dual_stage,
                        "new_value": new_dual_stage,
                        "reasoning": f"AI recommended {approach} training approach based on analysis"
                    }
            
            # Apply all changes at once using edit_config
            if changes_applied and hasattr(trainer, 'edit_config'):
                try:
                    trainer.edit_config(
                        base_model_name=config_params['base_model_name'],
                        batch_size=config_params['batch_size'],
                        initial_learning_rate=config_params['initial_learning_rate'],
                        fine_tune_learning_rate=config_params['fine_tune_learning_rate'],
                        initial_epochs=config_params['initial_epochs'],
                        fine_tune_epochs=config_params['fine_tune_epochs'],
                        dual_stage=config_params['dual_stage'],
                        custom_img_size=config_params['custom_img_size'],
                        unfreeze_percent=config_params['unfreeze_percent']
                    )
                    
                    # Log all applied changes
                    for param, change_info in changes_applied.items():
                        logger.info(f"Applied optimization: {param} = {change_info['new_value']} (was {change_info['old_value']})")
                    
                    logger.info(f"Successfully applied {len(changes_applied)} optimization changes using edit_config")
                    return True
                    
                except Exception as edit_error:
                    logger.error(f"Failed to apply optimizations using edit_config: {str(edit_error)}")
                    # Fallback to direct assignment
                    for param, change_info in changes_applied.items():
                        try:
                            setattr(trainer, param, change_info['new_value'])
                            logger.info(f"Applied optimization (fallback): {param} = {change_info['new_value']}")
                        except Exception as fallback_error:
                            logger.warning(f"Failed to apply {param}: {str(fallback_error)}")
                    return True
            else:
                logger.info("No optimization changes to apply")
                return False
            
        except Exception as e:
            logger.error(f"Failed to apply optimization recommendations: {str(e)}")
            return False

    def format_recommendations_for_logging(self, recommendations: Dict[str, Any], original_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format AI recommendations for inclusion in training logs.
        
        Args:
            recommendations: Full AI recommendations
            original_config: Original trainer configuration
            
        Returns:
            Formatted recommendations for logging
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "dataset_analysis": recommendations.get("analysis", {}),
                "original_configuration": original_config,
                "ai_recommendations": {},
                "applied_changes": {},
                "recommendation_summary": {}
            }
            
            # Extract hyperparameter recommendations with reasoning
            if "hyperparameters" in recommendations:
                params = recommendations["hyperparameters"]
                
                # Training config recommendations
                if "training_config" in params:
                    log_entry["ai_recommendations"]["training_config"] = {}
                    for param, details in params["training_config"].items():
                        if isinstance(details, dict) and "value" in details:
                            log_entry["ai_recommendations"]["training_config"][param] = {
                                "recommended_value": details["value"],
                                "confidence": details.get("confidence", "N/A"),
                                "reasoning": details.get("reasoning", "No reasoning provided")
                            }
                
                # Fine-tuning config recommendations
                if "fine_tuning_config" in params:
                    log_entry["ai_recommendations"]["fine_tuning_config"] = {}
                    for param, details in params["fine_tuning_config"].items():
                        if isinstance(details, dict) and "value" in details:
                            log_entry["ai_recommendations"]["fine_tuning_config"][param] = {
                                "recommended_value": details["value"],
                                "confidence": details.get("confidence", "N/A"),
                                "reasoning": details.get("reasoning", "No reasoning provided")
                            }
                
                # Model architecture recommendations
                if "model_architecture" in params:
                    arch = params["model_architecture"]
                    if isinstance(arch, dict):
                        log_entry["ai_recommendations"]["model_architecture"] = {
                            "recommended_model": arch.get("base_model", "N/A"),
                            "confidence": arch.get("confidence", "N/A"),
                            "reasoning": arch.get("reasoning", "No reasoning provided")
                        }
                
                # Optimization recommendations
                if "optimization" in params:
                    opt = params["optimization"]
                    log_entry["ai_recommendations"]["optimization"] = {}
                    for opt_type, details in opt.items():
                        if isinstance(details, dict):
                            log_entry["ai_recommendations"]["optimization"][opt_type] = {
                                "recommended_settings": details,
                                "confidence": details.get("confidence", "N/A"),
                                "reasoning": details.get("reasoning", "No reasoning provided")
                            }
            
            # SageMaker recommendations
            if "sagemaker_recommendations" in recommendations:
                sm_rec = recommendations["sagemaker_recommendations"]
                log_entry["ai_recommendations"]["sagemaker"] = {}
                for rec_type, details in sm_rec.items():
                    if isinstance(details, dict) and "value" in details:
                        log_entry["ai_recommendations"]["sagemaker"][rec_type] = {
                            "recommended_value": details["value"],
                            "confidence": details.get("confidence", "N/A"),
                            "reasoning": details.get("reasoning", "No reasoning provided")
                        }
                    else:
                        log_entry["ai_recommendations"]["sagemaker"][rec_type] = details
            
            # Data augmentation recommendations
            if "data_augmentation" in recommendations:
                aug_rec = recommendations["data_augmentation"]
                log_entry["ai_recommendations"]["data_augmentation"] = aug_rec
            
            # Applied changes will be added by the caller
            log_entry["applied_changes"] = {}
            
            # Create summary statistics
            total_recommendations = 0
            
            for category in log_entry["ai_recommendations"].values():
                if isinstance(category, dict):
                    total_recommendations += len(category)
            
            log_entry["recommendation_summary"] = {
                "total_parameters_analyzed": len(original_config),
                "total_recommendations_made": total_recommendations,
                "ai_model_used": self.model,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Failed to format recommendations for logging: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "raw_recommendations": recommendations
            }


def create_advisor_summary(recommendations: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of AI recommendations.
    
    Args:
        recommendations: AI recommendations dictionary
        
    Returns:
        Formatted summary string
    """
    try:
        summary = "=== AI ADVISOR RECOMMENDATIONS ===\n\n"
        
        if "analysis" in recommendations:
            analysis = recommendations["analysis"]
            summary += f"Dataset Complexity: {analysis.get('dataset_complexity', 'Unknown')}\n"
            summary += f"Recommended Approach: {analysis.get('recommended_approach', 'Unknown')}\n\n"
            
            if "key_insights" in analysis:
                summary += "Key Insights:\n"
                for insight in analysis["key_insights"]:
                    summary += f"  - {insight}\n"
                summary += "\n"
        
        if "hyperparameters" in recommendations:
            params = recommendations["hyperparameters"]
            summary += "RECOMMENDED HYPERPARAMETERS:\n\n"
            
            if "training_config" in params:
                config = params["training_config"]
                summary += "Training Configuration:\n"
                for key, value in config.items():
                    if isinstance(value, dict) and "value" in value:
                        conf = value.get("confidence", "N/A")
                        summary += f"  {key}: {value['value']} (confidence: {conf}%)\n"
                summary += "\n"
            
            if "fine_tuning_config" in params:
                config = params["fine_tuning_config"]
                summary += "Fine-tuning Configuration:\n"
                for key, value in config.items():
                    if isinstance(value, dict) and "value" in value:
                        conf = value.get("confidence", "N/A")
                        summary += f"  {key}: {value['value']} (confidence: {conf}%)\n"
                summary += "\n"
        
        if "sagemaker_recommendations" in recommendations:
            sm_rec = recommendations["sagemaker_recommendations"]
            summary += "SageMaker Recommendations:\n"
            if "instance_type" in sm_rec:
                summary += f"  Instance Type: {sm_rec['instance_type']['value']}\n"
            if "estimated_training_time" in sm_rec:
                summary += f"  Estimated Training Time: {sm_rec['estimated_training_time'].get('dual_stage', 'N/A')}\n"
            if "cost_estimate" in sm_rec:
                summary += f"  Estimated Cost: {sm_rec['cost_estimate'].get('approximate_cost', 'N/A')}\n"
        
        return summary
        
    except Exception as e:
        return f"Error creating summary: {str(e)}"
    

