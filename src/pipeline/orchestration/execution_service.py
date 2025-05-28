"""
AutoLM System Runner

This is orchestrator that implements AutoLM workflow following the data flow design:



Usage:
    python runner.py --goal "Your goal description" --config config/main_config.json
"""
import os
import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import all necessary modules
from src.pipeline.shared.logging import get_logger
from src.pipeline.shared.utility import DataUtility, AIUtility, StatisticsUtility
from src.pipeline.processing.generator import Generator, MetaGenerator
from src.pipeline.processing.evaluator import Evaluator
from src.pipeline.processing.retriever import VectorRetriever, GraphRetriever, MemoryRetriever

logger = get_logger(__name__)


class ExecutionService:
    """
    Ocrchestrator for providing the prompt execution service.
    Inputs:
    - Task Prompts Orchestration in config/plan.json (conform to db/schema/schema_plan.json)
    - Library of Task Prompts in db/prompts/task_prompt_library.json (conform to db/schema/schema_task_prompt_library.json)
    - Library of Meta Prompts in db/prompts/meta_prompt_library.json (conform to db/schema/schema_meta_prompt_library.json)
    - Configuration of how to select models, topologies (tts), hyperparameters as in config/mle_config.json
    
    Processes:
    1. Load the plan and extract the task prompts in scope from the task prompt library
    2. Execute the task prompts in scope following the sequencing in the plan
    3. Before executing each task prompt, retrieve relevant contexts from Vector DB and / or Graph DB and store to Memory DB (collection episodic memory).
    4. Load all relevant contexts from Memory DB for each task prompt.
    5. From Memory DB, retrieve the iteration parameters for each task prompt. (e.g. iteration time: 5, iteration value: [a, b, c, d, e])
    6. Contextualise the task prompt with the iteration value for the iteration.
    7. Classify / categorise the task prompt for task type and task complexity by using meta prompt.
    8. Based on mle.config, task type and task complexity, select the model, topology (tts), hyperparameters.
    9. Based on the task prompt, retrieve the useful MBTI and behavior traits from Memory DB (personality memory collection).
    10. Apply the MBTI and behavior traits to the system prompt in associated with the task prompt.
    11. Execute the task prompt with the uplifted system prompt, the task prompt optimistion topoology, the selected model, topology (tts), and hyperparameters.
    12. Parse the output and store in episodic memory, reflecting the entity name as suggested in the task prompt (to be added, which corresponds to entity description of the response).
    13. Evaluate the output from the task prompt against the expected output from the task prompt.
    14. Collate the final collection of outputs corresponding to the task prompts in the scope of the plan. 
    """    
    def __init__(self, 
                 config_file_path: Optional[Union[str, Path]] = None,
                 vector_db_json_path: Optional[Union[str, Path]] = None,
                 episodic_memory_json_path: Optional[Union[str, Path]] = None,
                 personality_memory_json_path: Optional[Union[str, Path]] = None,
                 graph_db_path: Optional[Union[str, Path]] = None, # For GraphRetriever (PKL)
                 plan_json_path: Optional[Union[str, Path]] = None,
                 task_library_json_path: Optional[Union[str, Path]] = None
                 ):
        """Initialize the ExecutionService with necessary components and DB paths.
        
        Args:
            config_file_path: Path to configuration file. If None, uses default path.
            vector_db_json_path: Path to the VectorDB JSON file for retriever.
            episodic_memory_json_path: Path to the Episodic Memory JSON file.
            personality_memory_json_path: Path to the Personality Memory JSON file.
            graph_db_path: Path to the GraphDB PKL file.
            plan_json_path: Path to the plan JSON file (e.g., prompt_flow_config.json).
            task_library_json_path: Path to the task prompt library JSON file (e.g., shortlisted_prompts.json).
        """
        logger.info("Initializing ExecutionService")
        start_time = time.time()
        
        try:
            # Set up directory structure (defaults if not overridden by specific paths)
            self.config_dir = Path.cwd() / "config"
            self.db_dir = Path.cwd() / "db"
            self.prompts_dir = self.db_dir / "prompt" 
            self.memory_dir = self.db_dir / "memory" 
            self.vector_db_dir = self.db_dir / "vector" 
            self.graph_db_dir = self.db_dir / "graph" 

            # Ensure base directories exist
            self.memory_dir.mkdir(exist_ok=True, parents=True)
            self.vector_db_dir.mkdir(exist_ok=True, parents=True)
            self.graph_db_dir.mkdir(exist_ok=True, parents=True)
            self.prompts_dir.mkdir(exist_ok=True, parents=True)

            # Store DB paths
            self.vector_db_json_path = Path(vector_db_json_path) if vector_db_json_path else self.vector_db_dir / "vector_db_for_retriever.json"
            self.episodic_memory_json_path = Path(episodic_memory_json_path) if episodic_memory_json_path else self.memory_dir / "episodic_memory_for_retriever.json"
            self.personality_memory_json_path = Path(personality_memory_json_path) if personality_memory_json_path else self.memory_dir / "personality_memory_for_retriever.json"
            self.graph_db_pkl_path = Path(graph_db_path) if graph_db_path else self.graph_db_dir / "g_complete.pkl" 

            # Load configuration
            self.config_path = Path(config_file_path) if config_file_path else self.config_dir / "main_config.json"
            self.config = self._load_config()
            
            # Load MLE configuration
            self.mle_config_path = self.config_dir / "mle_config.json"
            self.mle_config = self._load_mle_config()
            
            # Load plan configuration
            self.plan_path = Path(plan_json_path) if plan_json_path else self.config_dir / "prompt_flow_config.json"
            self.plan = self._load_plan() 
            
            # Store task library path
            self.task_library_path = Path(task_library_json_path) if task_library_json_path else self.config_dir / "shortlisted_prompts.json"

            # Initialize utilities
            self.data_utility = DataUtility()
            self.ai_utility = AIUtility() 
            self.stats_utility = StatisticsUtility()
            
            # Initialize components
            self.generator = Generator()
            self.meta_generator = MetaGenerator(generator=self.generator)
            self.evaluator = Evaluator() 
            
            # Initialize retrievers with JSON paths (or PKL for Graph)
            self.vector_retriever = VectorRetriever(vector_db_path=self.vector_db_json_path, generator=self.generator)
            self.graph_retriever = GraphRetriever(graph_db=self.graph_db_pkl_path, generator=self.generator, graph_type="standard")
            
            self.episodic_memory_retriever = MemoryRetriever(memory_db_path=self.episodic_memory_json_path, generator=self.generator, memory_type="episodic")
            self.personality_memory_retriever = MemoryRetriever(memory_db_path=self.personality_memory_json_path, generator=self.generator, memory_type="personality")

            # Track execution state
            self.current_task_id = None 
            self.execution_results: Dict[str, Any] = {} 
            self.task_outputs: Dict[str, Any] = {} 
            
            logger.info(f"ExecutionService initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"ExecutionService initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration from file with fallback to defaults."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.debug(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            "system": {
                "paths": {
                    "knowledge_base": "db",
                    "source": "data"
                }
            },
            "execution": {
                "default_model": "gpt-4o-mini",
                "default_temperature": 0.7,
                "max_tokens": 4000
            }
        }

    def _load_mle_config(self) -> Dict[str, Any]:
        """Load MLE configuration for model selection."""
        try:
            if self.mle_config_path.exists():
                with open(self.mle_config_path, 'r') as f:
                    config = json.load(f)
                logger.debug(f"Loaded MLE configuration from {self.mle_config_path}")
                return config
            else:
                logger.warning(f"MLE config file not found at {self.mle_config_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load MLE configuration: {e}")
            return {}

    def _load_plan(self) -> Dict[str, Any]:
        """Load execution plan from file."""
        try:
            if self.plan_path.exists():
                with open(self.plan_path, 'r') as f:
                    plan_data = json.load(f) 
                logger.debug(f"Loaded plan from {self.plan_path}")
                # prompt_flow_config.json is expected to be the direct graph structure.
                if "nodes" in plan_data and "edges" in plan_data:
                    return plan_data 
                else: 
                    logger.warning(f"Plan file {self.plan_path} does not have expected 'nodes' and 'edges' keys.")
                    # Return a structure that won't cause downstream get calls to fail catastrophically
                    return {"nodes": {}, "edges": [], "goal_description": "Unknown (Plan format error)"}
            else:
                logger.warning(f"Plan file not found at {self.plan_path}")
                return {"nodes": {}, "edges": [], "goal_description": "Unknown (Plan file not found)"}
        except Exception as e:
            logger.error(f"Failed to load plan: {e}")
            return {"nodes": {}, "edges": [], "goal_description": f"Unknown (Error loading plan: {e})"}

    def _load_task_prompt_library(self) -> Dict[str, Any]:
        """Load task prompt library from file, trying shortlisted first."""
        # Try loading shortlisted_prompts.json first
        if self.task_library_path.exists():
            try:
                with open(self.task_library_path, 'r') as f:
                    library = json.load(f)
                logger.info(f"Loaded task prompt library from specified path: {self.task_library_path}")
                # This file is expected to be a direct dictionary of {prompt_id: details}
                return library
            except Exception as e:
                logger.error(f"Failed to load task prompt library from {self.task_library_path}: {e}")
        
        # Fallback to the main library if shortlisted isn't found or fails
        main_library_path = self.prompts_dir / "task_prompt_library.json"
        logger.warning(f"Specified task library {self.task_library_path} not found or failed to load. Falling back to main library: {main_library_path}")
        try:
            if main_library_path.exists():
                with open(main_library_path, 'r') as f:
                    # The main library might be nested under "collections" > "task_prompt_templates"
                    # or it could be a direct dictionary. For now, assume it's a direct dictionary.
                    # If it's nested, this needs adjustment or the PlanService needs to save shortlisted_prompts.json
                    # as a direct dictionary.
                    library = json.load(f)
                logger.debug(f"Loaded main task prompt library from {main_library_path}")
                return library # Assuming it's {prompt_id: details}
            else:
                logger.error(f"Main task prompt library not found at {main_library_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load main task prompt library: {e}")
            return {}
    
    def execute_plan(self) -> Dict[str, Any]: # Renamed from execute
        """Execute the chain of task prompts according to the loaded plan.
        
        Returns:
            Dict containing execution results for all tasks.
        """
        logger.info("Starting plan execution process")
        start_time = time.time()
        
        try:
            task_library = self._load_task_prompt_library()
            if not task_library:
                raise ValueError("Task prompt library is empty or could not be loaded.")
            
            if not self.plan or "nodes" not in self.plan:
                raise ValueError("Plan is not loaded or has no nodes.")

            nodes = self.plan.get("nodes", {})
            # This is a simplified execution order. A real implementation would use graph traversal (e.g., topological sort)
            # based on self.plan.get("edges", []).
            # For now, iterate through nodes in the order they appear (if dict) or as a list
            # This assumes node IDs like "node_1", "node_2" can be sorted for sequential flow if needed.
            
            # A more robust way to determine execution order would be from `self.plan.get("edges", [])`
            # For this iteration, we will assume nodes are processed in the order they are defined or sorted by key.
            # This is a significant simplification of a true graph execution engine.
            
            # Get a list of node_ids. If they are named "node_1", "node_2", etc., sorting them
            # might give a semblance of order, but this is not guaranteed for complex graphs.
            # A true topological sort based on edges is needed for robust execution.
            # For now, let's assume the plan provider gives nodes in a processable order or
            # that dependencies are handled by tasks checking `self.task_outputs`.
            
            node_ids_to_process = list(nodes.keys()) # Or some sorted/topologically_sorted list
            logger.info(f"Executing plan with node order: {node_ids_to_process}")


            for node_id in node_ids_to_process:
                task_config = nodes[node_id] # This is the task configuration from the plan's node
                self.current_task_id = node_id 
                
                prompt_id = task_config.get("prompt_id")
                if not prompt_id:
                    logger.warning(f"Task node {node_id} has no prompt_id, skipping.")
                    self.execution_results[node_id] = {"error": "Missing prompt_id", "successful": False, "task_id": node_id}
                    continue
                
                # Get the full prompt details from the library using prompt_id
                task_prompt_details = task_library.get(prompt_id)
                if not task_prompt_details:
                    logger.warning(f"Task prompt with ID {prompt_id} for node {node_id} not found in library, skipping.")
                    self.execution_results[node_id] = {"error": f"Prompt {prompt_id} not found", "successful": False, "task_id": node_id}
                    continue
                
                # Merge task_config (from plan node) and task_prompt_details (from library)
                # task_config might have specific overrides or additional parameters for this instance of the prompt
                current_task_info = {**task_prompt_details, **task_config} # task_config overrides library details

                task_result = self._execute_task_prompt(node_id, current_task_info)
                self.execution_results[node_id] = task_result
                
                # Store specific outputs if defined by the prompt (e.g., if prompt has an 'output_variable_name')
                # This is a simple way to pass data between tasks.
                if task_result.get("successful") and current_task_info.get("output_variable_name"):
                    self.task_outputs[current_task_info["output_variable_name"]] = task_result.get("parsed_output")

                # Basic failure handling: stop if a task fails. Could be made more sophisticated based on edge conditions.
                if not self._check_task_conditions(node_id, task_result): 
                    logger.error(f"Task {node_id} (Prompt: {prompt_id}) failed or conditions not met. Halting execution.")
                    break 
            
            final_summary = self._collate_results()
            logger.info(f"Plan execution completed in {time.time() - start_time:.2f} seconds")
            return final_summary
            
        except Exception as e:
            logger.error(f"Plan execution failed: {str(e)}")
            logger.debug(f"Plan execution error details: {traceback.format_exc()}")
            return {"error": str(e), "results": self.execution_results, "status": "failed"}
    
    # _get_task_prompt removed as its logic is now integrated into execute_plan
    
    def _check_task_conditions(self, task_node_id: str, current_task_result: Dict[str, Any]) -> bool:
        """Check if task conditions are met to continue execution."""
        # Check if task execution was successful
        if not result.get("successful", False):
            return False
        
        # Check for specific conditions in task config
        conditions = task.get("conditions", [])
        for condition in conditions:
            condition_type = condition.get("type")
            
            if condition_type == "result_equals":
                # Check if a specific result value equals the expected value
                path = condition.get("path", "").split('.')
                expected = condition.get("value")
                
                # Navigate through the result dictionary
                actual = result
                for key in path:
                    if key in actual:
                        actual = actual[key]
                    else:
                        return False
                
                if actual != expected:
                    return False
            
            elif condition_type == "result_contains":
                # Check if a specific result contains the expected value
                path = condition.get("path", "").split('.')
                expected = condition.get("value")
                
                # Navigate through the result dictionary
                actual = result
                for key in path:
                    if key in actual:
                        actual = actual[key]
                    else:
                        return False
                
                if isinstance(actual, list) and expected not in actual:
                    return False
                elif isinstance(actual, str) and expected not in actual:
                    return False
                elif isinstance(actual, dict) and expected not in actual:
                    return False
        
        return True
    
    def _collate_results(self) -> Dict[str, Any]:
        """Collate final results from all task executions."""
        # Basic collation - can be extended for more complex result processing
        return {
            "goal": self.goal,
            "tasks_executed": self.current_task_index + 1,
            "tasks_successful": sum(1 for result in self.execution_results.values() if result.get("successful", False)),
            "results": self.execution_results,
            "timestamp": time.time()
        }
        
    def _execute_task_prompt(self, task: Dict[str, Any], task_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task prompt with all required steps.
        
        Args:
            task: Task configuration from the plan
            task_prompt: Task prompt details from the library
            
        Returns:
            Dict containing execution results for this task
        """
        prompt_id = task_prompt.get("prompt_id")
        logger.info(f"Executing task prompt: {prompt_id}")
        
        try:
            # Step 3.1: Retrieve contexts from Vector and Graph DBs
            # 'task' for _retrieve_contexts_from_dbs refers to the node config from the plan.
            # 'task_prompt' refers to the prompt details from the library.
            # current_task_info holds merged details.
            contexts = self._retrieve_contexts_from_dbs(current_task_info, current_task_info) 
            
            # Step 3.2: Store contexts to Memory DB (Episodic)
            # This should ideally use a working memory or a specific episodic memory instance for the current execution run.
            # For now, using self.episodic_memory_retriever.save_memory
            # This assumes _store_contexts_to_memory is adapted or save_memory handles this structure.
            self._store_contexts_to_memory(contexts, current_task_info, memory_type="episodic") # Specify memory type
            
            # Step 3.3: Load relevant contexts from Memory DB (Episodic)
            memory_contexts = self._retrieve_contexts_from_memory(current_task_info, memory_type="episodic")
            
            # Step 3.4: Get iteration parameters if available
            iterations = self._get_iteration_parameters(current_task_info, current_task_info) # Pass current_task_info
            
            task_results_summary = [] # To store results from each iteration
            
            # Step 3.5: Execute for each iteration
            for iteration_index, iteration_value in enumerate(iterations):
                contextualized_prompt_text = self._contextualize_prompt(current_task_info, iteration_value, memory_contexts)
                task_type, task_complexity = self._classify_task(contextualized_prompt_text) # Pass text
                personality_traits = self._get_personality_traits(current_task_info) # Pass current_task_info
                
                system_prompt_template = current_task_info.get("components", {}).get("system_prompt", "") # Get from components
                system_prompt = self._apply_personality_to_system_prompt(
                    system_prompt_template, # Use template
                    personality_traits
                )
                
                execution_config = self._select_execution_config(task_type, task_complexity)
                response = self._execute_prompt(
                    prompt_text=contextualized_prompt_text, # Pass the contextualized text
                    system_prompt=system_prompt,
                    execution_config=execution_config # This should contain model name, params etc.
                )
                
                parsed_output = self._parse_output(response, current_task_info) # Pass current_task_info
                evaluation = self._evaluate_output(parsed_output, current_task_info) # Pass current_task_info
                
                iteration_summary = {
                    "iteration_index": iteration_index,
                    "iteration_value": iteration_value,
                    "parsed_output": parsed_output,
                    "evaluation": evaluation,
                    "model_config_used": execution_config # Store what was actually used
                }
                task_results_summary.append(iteration_summary)
                
                self._store_result_to_memory(iteration_summary, current_task_info, iteration_value, memory_type="episodic")
            
            overall_success = all(r.get("evaluation", {}).get("success", False) for r in task_results_summary)
            return {
                "prompt_id": prompt_id,
                "task_id": task_id,
                "iterations_count": len(iterations),
                "iteration_results": task_results_summary,
                "successful": overall_success,
                "parsed_output": task_results_summary[-1]["parsed_output"] if task_results_summary else None 
            }
            
        except Exception as e:
            logger.error(f"Task execution failed for task_id {task_id} (Prompt {prompt_id}): {str(e)}")
            logger.debug(f"Task execution error details: {traceback.format_exc()}")
            return {
                "prompt_id": prompt_id,
                "task_id": task_id,
                "error": str(e),
                "successful": False
            }
    
    def _retrieve_contexts_from_dbs(self, task_config: Dict[str, Any], task_prompt_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant contexts from Vector and Graph DBs."""
        contexts = []
        # Use task_summary or description from task_prompt_details as query if context_query not present
        query_text = task_config.get("context_query", task_prompt_details.get("description", task_config.get("task_summary", "")))
        
        retrieval_config = task_config.get("retrieval_config", {})
        vector_ret_config = retrieval_config.get("vector_config", {}) # Default to empty dict if not present
        graph_ret_config = retrieval_config.get("graph_config", {})   # Default to empty dict if not present

        # Retrieve from vector DB
        if self.vector_retriever and self.vector_retriever.vector_df is not None and not self.vector_retriever.vector_df.empty:
            logger.info(f"Retrieving from VectorDB for query: {query_text[:100]}")
            vector_results = self.vector_retriever.retrieve(query_text, vector_ret_config)
            for result in vector_results:
                contexts.append({
                    "source_db": "vector_db", 
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0)
                })
        
        # Retrieve from graph DB
        if self.graph_retriever and self.graph_retriever.graph is not None and self.graph_retriever.graph.number_of_nodes() > 0:
            logger.info(f"Retrieving from GraphDB for query: {query_text[:100]}")
            graph_results = self.graph_retriever.retrieve(query_text, graph_ret_config)
            for result in graph_results:
                contexts.append({
                    "source_db": "graph_db", 
                    "content": result.get("content", ""), 
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0)
                })
        
        contexts.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return contexts

    def _store_contexts_to_memory(self, contexts: List[Dict[str, Any]], task_info: Dict[str, Any], memory_type: str = "episodic") -> None:
        """Store retrieved contexts to the specified Memory DB type."""
        retriever_to_use = None
        if memory_type == "episodic":
            retriever_to_use = self.episodic_memory_retriever
        # Add elif for "working" or other types if they have dedicated retrievers
        # elif memory_type == "personality":
        #     retriever_to_use = self.personality_memory_retriever

        if not retriever_to_use:
            logger.warning(f"No retriever configured for memory_type '{memory_type}'. Cannot store contexts.")
            return

        for i, context_item in enumerate(contexts):
            memory_entry = {
                "memory_id": str(uuid.uuid4()), # Ensure uuid is imported: import uuid
                "entity": f"context_for_task_{self.current_task_id}_item_{i}", 
                "query": task_info.get("task_summary", task_info.get("prompt_id")), 
                "context": context_item, 
                "timestamp": time.time(), # Use time.time() for Unix timestamp
                "source_db_type": context_item.get("source_db") 
            }
            retriever_to_use.save_memory(memory_entry) # Assumes save_memory can handle this dict
        logger.info(f"Stored {len(contexts)} contexts to {memory_type} memory for task {self.current_task_id}.")


    def _retrieve_contexts_from_memory(self, task_info: Dict[str, Any], memory_type: str = "episodic") -> List[Dict[str, Any]]:
        """Load relevant contexts from the specified Memory DB type."""
        retriever_to_use = None
        if memory_type == "episodic":
            retriever_to_use = self.episodic_memory_retriever
        elif memory_type == "personality":
             retriever_to_use = self.personality_memory_retriever
        # Add elif for "working" or other types

        if not retriever_to_use:
            logger.warning(f"No retriever configured for memory_type '{memory_type}'. Cannot retrieve contexts.")
            return []

        query = task_info.get("context_query", task_info.get("task_summary", task_info.get("prompt_id", "")))
        
        config = {
            "search": {"symbolic": True, "semantic": True, "hybrid_weight": 0.6},
            "limits": {"top_k": 5} 
        }
        
        logger.info(f"Retrieving from {memory_type} memory for task {self.current_task_id} using query: {query[:100]}")
        results = retriever_to_use.retrieve(query, config)
        
        retrieved_contexts = [res.get("context", {}) for res in results if "context" in res]
        logger.info(f"Retrieved {len(retrieved_contexts)} contexts from {memory_type} memory.")
        return retrieved_contexts
        
    def _get_iteration_parameters(self, task_config: Dict[str, Any], task_prompt_details: Dict[str, Any]) -> List[Any]:
        """Get iteration parameters for the task. Default to single iteration with None value."""
        if "iterations" in task_config: 
            logger.debug(f"Iterations found in task_config for task {self.current_task_id}")
            return task_config["iterations"]
        
        prompt_components = task_prompt_details.get("components", {})
        # Iteration logic based on prompt_components can be added here if schema supports it
        # For now, if not in task_config, default to single iteration
            
        logger.debug(f"No iteration parameters specified for task {self.current_task_id}. Defaulting to single iteration.")
        return [None] 
        
    def _contextualize_prompt(self, task_prompt_details: Dict[str, Any], iteration_value: Any, 
                          contexts: List[Dict[str, Any]]) -> str:
        """Contextualize the prompt with iteration value and contexts."""
        prompt_components = task_prompt_details.get("components", {})
        base_prompt_text = prompt_components.get("instructions", prompt_components.get("task", ""))
        
        # Construct the prompt header
        final_prompt_text = f"Role: {prompt_components.get('role', 'AI Assistant')}\n"
        final_prompt_text += f"Primary Task: {prompt_components.get('task', '')}\n"
        final_prompt_text += f"Purpose: {prompt_components.get('purpose', '')}\n"
        final_prompt_text += f"Audience: {prompt_components.get('audience', '')}\n"
        if prompt_components.get('context'): # Scenario context from prompt library
             final_prompt_text += f"Scenario Context: {prompt_components.get('context')}\n"
        
        # Handle iteration value
        if iteration_value is not None:
            if "{{iteration_value}}" in base_prompt_text:
                base_prompt_text = base_prompt_text.replace("{{iteration_value}}", str(iteration_value))
            else: 
                final_prompt_text += f"Current Iteration Value: {str(iteration_value)}\n"

        final_prompt_text += f"\nInstructions:\n{base_prompt_text}\n"

        # Add retrieved contexts
        if contexts: # Assuming contexts should always be added if present
            context_str = "\n--- Relevant Context Start ---\n"
            for i, ctx in enumerate(contexts[:3]): # Limit to top 3 contexts for brevity
                ctx_content = ctx.get('content', 'No content')
                ctx_source = ctx.get('source_db', 'unknown_source') # Changed from 'source'
                ctx_score = ctx.get('score', 0.0)
                context_str += f"Context Item {i+1} (Source: {ctx_source}, Relevance: {ctx_score:.2f}):\n{ctx_content}\n\n"
            context_str += "--- Relevant Context End ---\n"
            final_prompt_text += context_str
        
        final_prompt_text += f"\nExpected Response Format: {prompt_components.get('response_format', 'Clear, concise text.')}\n"
        
        logger.debug(f"Contextualized prompt for task {self.current_task_id}: {final_prompt_text[:300]}...") # Increased log length
        return final_prompt_text

    def _classify_task(self, contextualized_prompt_text: str) -> Tuple[str, str]:
        """Classify the task type and complexity using meta prompt."""
        # Use meta generator to classify
        try:
            # Ensure that the meta prompt for task classification is correctly defined and loaded.
            # For now, using a placeholder prompt_id=100 as in the original.
            classification_response = self.meta_generator.get_meta_generation(
                application="metaworkflow", # This should match a definition in meta_prompt_library.json
                category="task_classifier", # This should match a definition
                action="classify",          # This should match a definition
                prompt_id=100, # This specific prompt_id should exist for this app/cat/action
                task_prompt=contextualized_prompt_text, # The actual prompt text to classify
                model="Qwen2.5-1.5B", # Or from config
                temperature=0.3 # Low temperature for classification
            )
            
            # Attempt to parse the classification response (assuming it's JSON)
            if isinstance(classification_response, str):
                try:
                    classification_data = json.loads(classification_response)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from task classification response: {classification_response}")
                    classification_data = {} # Fallback
            elif isinstance(classification_response, dict):
                classification_data = classification_response
            else:
                classification_data = {}

            task_type = classification_data.get("task_type", "general")
            task_complexity = classification_data.get("task_complexity", "medium")
                
        except Exception as e:
            logger.error(f"Task classification failed: {e}")
            task_type = "general" # Default values if classification fails
            task_complexity = "medium"
        
        logger.debug(f"Classified task as Type: {task_type}, Complexity: {task_complexity}")
        return task_type, task_complexity

    def _get_personality_traits(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get personality traits from Personality Memory DB for the task."""
        # Use the dedicated personality memory retriever
        if not self.personality_memory_retriever:
            logger.warning("Personality memory retriever not available.")
            return {}
            
        query = task_info.get("task_summary", task_info.get("prompt_id", "")) # Query based on task summary or ID
        
        config = {
            "search": {"symbolic": True, "semantic": True, "hybrid_weight": 0.5},
            "limits": {"top_k": 1} # Usually, one dominant personality profile is sought
        }
        
        logger.info(f"Retrieving personality traits for task {self.current_task_id} using query: {query[:100]}")
        results = self.personality_memory_retriever.retrieve(query, config)
        
        if results:
            # Assuming the first result is the most relevant personality profile
            # The fields here should match the schema of personality_memory.json
            trait_data = results[0] 
            return {
                "mbti_type": trait_data.get("mbti_type"),
                "personality_type": trait_data.get("personality_type"),
                "cognitive_style": trait_data.get("cognitive_style"),
                "mode_description": trait_data.get("mode_description")
                # Add other relevant fields from personality memory schema
            }
        
        logger.info(f"No specific personality traits found for task {self.current_task_id}.")
        return {} # Default empty traits

    def _apply_personality_to_system_prompt(self, system_prompt_template: str, 
                                        personality_traits: Dict[str, Any]) -> str:
        """Apply personality traits to the system prompt."""
        """Apply personality traits to the system prompt."""
        if not personality_traits or not any(personality_traits.values()): # Check if traits dict is empty or all values are None/empty
            return system_prompt_template
        
        # Add personality information to system prompt
        # This could involve placeholders in the system_prompt_template like {{mbti_type}}
        # Or simply appending the information. For now, appending.
        
        personality_addon = "\n\n--- AI Personality Profile ---\n"
        if personality_traits.get("mbti_type"):
            personality_addon += f"MBTI Type: {personality_traits['mbti_type']}\n"
        if personality_traits.get("personality_type"):
            personality_addon += f"Personality Type: {personality_traits['personality_type']}\n"
        if personality_traits.get("cognitive_style"):
            personality_addon += f"Cognitive Style: {personality_traits['cognitive_style']}\n"
        if personality_traits.get("mode_description"):
            personality_addon += f"Current Mode: {personality_traits['mode_description']}\n"
        personality_addon += "--- End AI Personality Profile ---\n"
        
        return system_prompt_template + personality_addon
    
    def _select_execution_config(self, task_type: str, task_complexity: str) -> Dict[str, Any]:
        """Select model, topology, and parameters based on task classification and MLE config."""
        # Default config from main_config.json
        default_exec_config = self.config.get("execution", {
            "default_model": "gpt-4o-mini", 
            "default_temperature": 0.7, 
            "max_tokens": 4000
        })

        # Try to find specific config in mle_config.json
        # mle_config structure: {"task_types": {"type": {"complexities": {"level": {config}}}}}
        specific_config = self.mle_config.get("task_types", {}).get(task_type, {}).get("complexities", {}).get(task_complexity)

        if specific_config:
            logger.info(f"Using specific MLE config for task type '{task_type}', complexity '{task_complexity}'.")
            # Merge with defaults: specific_config overrides defaults
            return {
                "model_provider": specific_config.get("model_provider", default_exec_config.get("default_model_provider", "azure_openai")), # Assuming a provider field
                "model_name": specific_config.get("model_name", default_exec_config.get("default_model")),
                "parameters": {**default_exec_config, **specific_config.get("parameters", {})}, # Merge params
                "tts_topology": specific_config.get("tts_topology") # This might be a string or dict
            }
        else:
            logger.info(f"No specific MLE config for task type '{task_type}', complexity '{task_complexity}'. Using defaults.")
            return {
                "model_provider": default_exec_config.get("default_model_provider", "azure_openai"),
                "model_name": default_exec_config.get("default_model"),
                "parameters": default_exec_config, # Use all default execution params
                "tts_topology": None # No specific topology
            }

    def _execute_prompt(self, prompt_text: str, system_prompt: str, 
                       execution_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the prompt with selected model and parameters."""
        # This method was incomplete, providing a basic default execution.
        # For now, it will use direct execution. Topology handling would be more complex.
        
        model_name = execution_config.get("model_name", self.config.get("execution", {}).get("default_model", "gpt-4o-mini"))
        parameters = execution_config.get("parameters", self.config.get("execution", {}))
        
        logger.debug(f"Executing prompt with model: {model_name}, params: {parameters}")

        # Default direct execution
        result = self.generator.get_completion(
            prompt_id=str(time.time_ns()), # Unique ID for the call
            prompt=prompt_text,
            system_prompt=system_prompt,
            model=model_name,
            temperature=parameters.get("temperature", self.config.get("execution", {}).get("default_temperature",0.7)),
            max_tokens=parameters.get("max_tokens", self.config.get("execution", {}).get("max_tokens",4000)),
            top_p=parameters.get("top_p", 1.0),
            frequency_penalty=parameters.get("frequency_penalty", 0.0),
            presence_penalty=parameters.get("presence_penalty", 0.0),
            # seed=parameters.get("seed", None), # Seed might not be supported by all models/APIs
            # json_schema=parameters.get("json_schema", None), # JSON schema might not be part of basic config
            return_full_response=True # Ensure we get usage data
        )
            
        return {
            "content": result.get("content", ""),
            "model_used": model_name, # Changed key for clarity
            "tokens_used": result.get("usage", {}).get("total_tokens", 0), # Changed key
            "topology_used": execution_config.get("tts_topology", "direct") # Changed key
        }

    def _get_topology_template(self, topology: Optional[str]) -> Optional[str]: # topology can be None
        """Get template for the specified topology."""
        # These would typically come from a templates file or database
        templates = {
            "chain_of_thought": "Think step by step:\n\n{{prompt}}\n\nLet me think through this carefully:\n1. ",
            "recursive_chain_of_thought": "I'll solve this recursively:\n\n{{prompt}}\n\nStarting with the base case:\n",
            "self_reflection": "{{prompt}}\n\nInitial thoughts:\n\nOn reflection:\n",
            "socratic_dialogue": "{{prompt}}\n\nLet me explore this through questions:\n"
        }
        
        return templates.get(topology)

    def _apply_topology_to_prompt(self, prompt: str, template: str) -> str:
        """Apply topology template to the prompt."""
        return template.replace("{{prompt}}", prompt)

    def _execute_cot_prompt(self, prompt: str, system_prompt: str, 
                            model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prompt with Chain of Thought topology."""
        # Implementation of chain of thought execution
        model_name = model_config.get("model_name", "gpt-4o-mini")
        parameters = model_config.get("parameters", {})
        
        # First execution to get the reasoning steps
        reasoning_result = self.generator.get_completion(
            prompt_id=time.time_ns(),
            prompt=prompt,
            system_prompt=system_prompt + "\nThink step by step before providing your final answer.",
            model=model_name,
            temperature=parameters.get("temperature", 0.7),
            max_tokens=parameters.get("max_tokens", 4000),
            return_full_response=True
        )
        
        # For recursive CoT, we might have additional steps based on the complexity
        if model_config.get("tts_topology") == "recursive_chain_of_thought":
            steps = parameters.get("num_recursive_steps", 1)
            current_reasoning = reasoning_result.get("content", "")
            
            for step in range(steps):
                # Add previous reasoning to prompt for refinement
                refine_prompt = f"{prompt}\n\nMy previous reasoning:\n{current_reasoning}\n\nLet me refine this further:"
                
                refine_result = self.generator.get_completion(
                    prompt_id=time.time_ns(),
                    prompt=refine_prompt,
                    system_prompt=system_prompt,
                    model=model_name,
                    temperature=max(0.1, parameters.get("temperature", 0.7) - 0.2),  # Lower temperature for refinement
                    max_tokens=parameters.get("max_tokens", 4000),
                    return_full_response=True
                )
                
                current_reasoning = refine_result.get("content", "")
            
            final_content = current_reasoning
        else:
            final_content = reasoning_result.get("content", "")
        
        return {
            "content": final_content,
            "model": model_name,
            "tokens": reasoning_result.get("usage", {}).get("total_tokens", 0),
            "topology": model_config.get("tts_topology", "chain_of_thought")
        }

    def _execute_self_reflection_prompt(self, prompt: str, system_prompt: str, 
                                        model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prompt with Self Reflection topology."""
        model_name = model_config.get("model_name", "gpt-4o-mini")
        parameters = model_config.get("parameters", {})
        
        # First execution to get initial response
        initial_result = self.generator.get_completion(
            prompt_id=time.time_ns(),
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=parameters.get("temperature", 0.7),
            max_tokens=parameters.get("max_tokens", 4000) // 2,  # Half tokens for initial
            return_full_response=True
        )
        
        initial_response = initial_result.get("content", "")
        
        # Reflection prompt
        reflection_prompt = f"{prompt}\n\nMy initial response:\n{initial_response}\n\nOn reflection, I can improve my response by:"
        
        reflection_result = self.generator.get_completion(
            prompt_id=time.time_ns(),
            prompt=reflection_prompt,
            system_prompt=system_prompt + "\nCritically evaluate your initial response and provide an improved answer.",
            model=model_name,
            temperature=parameters.get("temperature", 0.7),
            max_tokens=parameters.get("max_tokens", 4000) // 2,  # Half tokens for reflection
            return_full_response=True
        )
        
        reflection = reflection_result.get("content", "")
        
        # Combine initial and reflection
        final_content = f"Initial response:\n{initial_response}\n\nReflection and improvements:\n{reflection}"
        
        total_tokens = (
            initial_result.get("usage", {}).get("total_tokens", 0) + 
            reflection_result.get("usage", {}).get("total_tokens", 0)
        )
        
        return {
            "content": final_content,
            "model": model_name,
            "tokens": total_tokens,
            "topology": "self_reflection"
        }

    def _parse_output(self, response: Dict[str, Any], task_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the model output according to task prompt specifications."""
        content = response.get("content", "")
        output_format = task_prompt.get("output_format", "text")
        
        # Parse based on output format
        if output_format == "json":
            try:
                # Try to parse as JSON
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON with a meta prompt
                try:
                    extracted_json = self.meta_generator.get_meta_generation(
                        application="metaworkflow",
                        category="output_parser",
                        action="extract_json",
                        prompt_id=101,
                        task_prompt=content,
                        model="Qwen2.5-1.5B",
                        temperature=0.3
                    )
                    
                    if isinstance(extracted_json, dict):
                        return extracted_json
                    else:
                        return {"raw_content": content, "parsing_error": "Failed to extract JSON"}
                    
                except Exception as e:
                    logger.error(f"JSON extraction failed: {e}")
                    return {"raw_content": content, "parsing_error": str(e)}
        
        elif output_format == "list":
            # Split by newlines and clean up
            items = [line.strip() for line in content.split("\n") if line.strip()]
            return {"items": items}
        
        else:
            # Default text format
            return {"text": content}

    def _evaluate_output(self, parsed_output: Dict[str, Any], task_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the output against expected criteria."""
        evaluation_criteria = task_prompt.get("evaluation_criteria", [])
        
        # If no criteria specified, consider successful
        if not evaluation_criteria:
            return {"success": True}
        
        # Convert parsed output to string for evaluation if needed
        output_str = json.dumps(parsed_output) if isinstance(parsed_output, dict) else str(parsed_output)
        
        # Use evaluator to check criteria
        try:
            evaluation_results = []
            overall_success = True
            
            for criterion in evaluation_criteria:
                criterion_type = criterion.get("type")
                criterion_value = criterion.get("value")
                
                if criterion_type == "contains":
                    # Check if output contains specific text
                    success = criterion_value in output_str
                    result = {"criterion": criterion, "success": success}
                    evaluation_results.append(result)
                    if not success:
                        overall_success = False
                
                elif criterion_type == "matches_regex":
                    # Check if output matches regex pattern
                    import re
                    pattern = re.compile(criterion_value)
                    success = bool(pattern.search(output_str))
                    result = {"criterion": criterion, "success": success}
                    evaluation_results.append(result)
                    if not success:
                        overall_success = False
                
                elif criterion_type == "semantic_similarity":
                    # Check semantic similarity with expected output
                    target_text = criterion.get("target_text", "")
                    threshold = criterion.get("threshold", 0.7)
                    
                    # Get embeddings and calculate similarity
                    output_embedding = self.generator.get_embedding(output_str)
                    target_embedding = self.generator.get_embedding(target_text)
                    
                    from scipy.spatial.distance import cosine
                    similarity = 1 - cosine(output_embedding, target_embedding)
                    
                    success = similarity >= threshold
                    result = {
                        "criterion": criterion,
                        "success": success,
                        "similarity": similarity
                    }
                    evaluation_results.append(result)
                    if not success:
                        overall_success = False
            
            return {
                "success": overall_success,
                "criteria_results": evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Output evaluation failed: {e}")
            logger.debug(f"Evaluation error details: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def _store_result_to_memory(self, result: Dict[str, Any], task_prompt: Dict[str, Any], 
                                iteration_value: Any) -> None:
        """Store result to memory for future reference."""
        memory_entry = {
            "memory_id": str(uuid.uuid4()),
            "entity": task_prompt.get("entity_name", f"result_{task_prompt.get('prompt_id')}"),
            "query": task_prompt.get("prompt_text", ""),
            "context": {
                "iteration_value": iteration_value,
                "result": result.get("parsed_output", {}),
                "evaluation": result.get("evaluation", {}),
                "model_config": result.get("model_config", {})
            },
            "timestamp": time.time(),
            "source": "execution"
        }
        
        # Store in memory DB
        self.memory_retriever.save_memory(memory_entry)
        
        # Add to local tracking
        self.memory_entries.append(memory_entry)


# def main():
#     """Main entry point for the execution service."""
#     parser = argparse.ArgumentParser(description='AutoLM Execution Service')
#     parser.add_argument('--goal', type=str, help='Goal description')
#     parser.add_argument('--config', type=str, help='Path to configuration file')
#     args = parser.parse_args()
    
#     try:
#         # Initialize execution service
#         service = ExecutionService(
#             config_file_path=args.config,
#             goal=args.goal
#         )
        
#         # Execute the plan
#         results = service.execute()
        
#         # Print summary
#         print(f"Execution completed with {results.get('tasks_successful', 0)}/{results.get('tasks_executed', 0)} successful tasks")
        
#         return 0
    
#     except Exception as e:
#         logger.error(f"Execution failed: {e}")
#         return 1


# if __name__ == "__main__":
#     exit(main())