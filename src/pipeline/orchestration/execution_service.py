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
    def __init__(self, config_file_path: Optional[Union[str, Path]] = None):
        """Initialize the ExecutionService with necessary components.
        
        Args:
            config_file_path: Path to configuration file. If None, uses default path.
        """
        logger.info("Initializing ExecutionService")
        start_time = time.time()
        
        try:
            # Set up directory structure
            self.config_dir = Path.cwd() / "config"
            self.db_dir = Path.cwd() / "db"
            self.prompts_dir = self.db_dir / "prompt"
            self.memory_dir = self.db_dir / "memory"
            self.vector_db_dir = self.db_dir / "vector"
            self.graph_db_dir = self.db_dir / "graph"

            # Ensure directories exist
            self.memory_dir.mkdir(exist_ok=True, parents=True)
            self.prompts_dir.mkdir(exist_ok=True, parents=True)
            self.vector_db_dir.mkdir(exist_ok=True, parents=True)
            self.graph_db_dir.mkdir(exist_ok=True, parents=True)

            # Load graph database
            self.vector_db = "v_"
            self.graph_db = self._load_graph_db()

            # Load configuration
            self.config_path = Path(config_file_path) if config_file_path else self.config_dir / "main_config.json"
            self.config = self._load_config()
            
            # Load MLE configuration
            self.mle_config_path = self.config_dir / "mle_config.json"
            self.mle_config = self._load_mle_config()
            
            # Load plan configuration
            self.plan_path = self.config_dir / "plan.json"
            self.plan = self._load_plan()
            
            # Initialize utilities
            self.data_utility = DataUtility()
            self.ai_utility = AIUtility()
            self.stats_utility = StatisticsUtility()
            
            # Initialize components
            self.generator = Generator()
            self.meta_generator = MetaGenerator(generator=self.generator)
            self.evaluator = Evaluator()
            
            # Initialize retrievers (will be configured later when needed)
            self.vector_retriever = VectorRetriever(vector_db=self.vector_db, generator=self.generator)
            self.graph_retriever = GraphRetriever(graph_db=self.graph_db, generator=self.generator, graph_type="standard")
            self.memory_retriever = MemoryRetriever(generator=self.generator)
            
            # Track execution state
            self.current_task_index = 0
            self.execution_results = {}
            self.memory_entries = []
            
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
                    plan = json.load(f)
                logger.debug(f"Loaded plan from {self.plan_path}")
                return plan
            else:
                logger.warning(f"Plan file not found at {self.plan_path}")
                return {"tasks": []}
        except Exception as e:
            logger.error(f"Failed to load plan: {e}")
            return {"tasks": []}

    def _load_task_prompt_library(self) -> Dict[str, Any]:
        """Load task prompt library from file."""
        task_prompt_path = self.prompts_dir / "task_prompt_library.json"
        try:
            if task_prompt_path.exists():
                with open(task_prompt_path, 'r') as f:
                    library = json.load(f)
                logger.debug(f"Loaded task prompt library from {task_prompt_path}")
                return library
            else:
                logger.warning(f"Task prompt library not found at {task_prompt_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load task prompt library: {e}")
            return {}
    
    def execute(self) -> Dict[str, Any]:
        """Execute the chain of task prompts according to the plan.
        
        Returns:
            Dict containing execution results
        """
        logger.info("Starting execution process")
        start_time = time.time()
        
        try:
            # Step 1: Load task prompt library
            task_library = self._load_task_prompt_library()
            if not task_library:
                raise ValueError("Task prompt library is empty or could not be loaded")
            
            # Step 2: Get tasks from plan
            tasks = self.plan.get("tasks", [])
            if not tasks:
                raise ValueError("No tasks found in the plan")
            
            # Step 3: Execute each task in sequence
            for task_index, task in enumerate(tasks):
                self.current_task_index = task_index
                
                # Get task prompt ID
                task_prompt_id = task.get("prompt_id")
                if not task_prompt_id:
                    logger.warning(f"Task at index {task_index} has no prompt_id, skipping")
                    continue
                
                # Get task prompt from library
                task_prompt = self._get_task_prompt(task_prompt_id, task_library)
                if not task_prompt:
                    logger.warning(f"Task prompt with ID {task_prompt_id} not found in library, skipping")
                    continue
                
                # Execute the task prompt
                result = self._execute_task_prompt(task, task_prompt)
                
                # Store the result
                self.execution_results[task_prompt_id] = result
                
                # Check for dependencies and conditions
                if not self._check_task_conditions(task, result):
                    logger.info(f"Task conditions not met for task {task_prompt_id}, stopping execution")
                    break
            
            # Collate final results
            final_results = self._collate_results()
            
            logger.info(f"Execution completed in {time.time() - start_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")
            logger.debug(f"Execution error details: {traceback.format_exc()}")
            return {"error": str(e), "results": self.execution_results}
    
    def _get_task_prompt(self, prompt_id: str, task_library: Dict[str, Any]) -> Dict[str, Any]:
        """Get task prompt from library by ID."""
        # Look for the prompt in the library
        for prompt in task_library.get("prompts", []):
            if prompt.get("prompt_id") == prompt_id:
                return prompt
        return None
    
    def _check_task_conditions(self, task: Dict[str, Any], result: Dict[str, Any]) -> bool:
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
            contexts = self._retrieve_contexts_from_dbs(task, task_prompt)
            
            # Step 3.2: Store contexts to Memory DB
            self._store_contexts_to_memory(contexts, task_prompt)
            
            # Step 3.3: Load relevant contexts from Memory DB
            memory_contexts = self._retrieve_contexts_from_memory(task_prompt)
            
            # Step 3.4: Get iteration parameters if available
            iterations = self._get_iteration_parameters(task, task_prompt)
            
            # Prepare results container
            results = []
            
            # Step 3.5: Execute for each iteration
            for iteration_index, iteration_value in enumerate(iterations):
                # Contextualize the prompt with iteration value
                contextualized_prompt = self._contextualize_prompt(task_prompt, iteration_value, memory_contexts)
                
                # Classify the task type and complexity
                task_type, task_complexity = self._classify_task(contextualized_prompt)
                
                # Get personality traits for the task
                personality_traits = self._get_personality_traits(task_prompt)
                
                # Apply personality traits to system prompt
                system_prompt = self._apply_personality_to_system_prompt(
                    task_prompt.get("system_prompt", ""),
                    personality_traits
                )
                
                # Select model, topology, and parameters based on task classification
                execution_config = self._select_execution_config(task_type, task_complexity)

                # Execute the prompt with selected configuration
                response = self._execute_prompt(
                    contextualized_prompt,
                    system_prompt,
                    execution_config
                )
                
                # Parse and store the output
                parsed_output = self._parse_output(response, task_prompt)
                
                # Evaluate the output
                evaluation = self._evaluate_output(parsed_output, task_prompt)
                
                # Store result for this iteration
                iteration_result = {
                    "iteration_index": iteration_index,
                    "iteration_value": iteration_value,
                    "parsed_output": parsed_output,
                    "evaluation": evaluation,
                    "model_config": model_config
                }
                
                # Store in memory
                self._store_result_to_memory(iteration_result, task_prompt, iteration_value)
                
                # Add to results
                results.append(iteration_result)
            
            # Return combined results
            return {
                "prompt_id": prompt_id,
                "iterations": len(iterations),
                "results": results,
                "successful": all(r.get("evaluation", {}).get("success", False) for r in results)
            }
            
        except Exception as e:
            logger.error(f"Task execution failed for prompt {prompt_id}: {str(e)}")
            logger.debug(f"Task execution error details: {traceback.format_exc()}")
            return {
                "prompt_id": prompt_id,
                "error": str(e),
                "successful": False
            }
    
    def _retrieve_contexts(self, task: Dict[str, Any], task_prompt: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant contexts from Vector and Graph DBs."""
        contexts = []
        query = task_prompt.get("context_query", task_prompt.get("prompt_text", ""))
        
        # Configure vector retriever if needed
        if not self.vector_retriever and "vector_db" in task.get("retrieval_config", {}):
            vector_db_path = task["retrieval_config"]["vector_db"]
            self.vector_retriever = VectorRetriever(
                vector_db=vector_db_path,
                generator=self.generator
            )
        
        # Configure graph retriever if needed
        if not self.graph_retriever and "graph_db" in task.get("retrieval_config", {}):
            graph_db_path = task["retrieval_config"]["graph_db"]
            self.graph_retriever = GraphRetriever(
                graph_db=graph_db_path,
                generator=self.generator
            )
        
        # Retrieve from vector DB
        if self.vector_retriever and "vector_db" in task.get("retrieval_config", {}):
            config = task.get("retrieval_config", {}).get("vector_config", {})
            vector_results = self.vector_retriever.retrieve(query, config)
            for result in vector_results:
                contexts.append({
                    "source": "vector_db",
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0)
                })
        
        # Retrieve from graph DB
        if self.graph_retriever and "graph_db" in task.get("retrieval_config", {}):
            config = task.get("retrieval_config", {}).get("graph_config", {})
            graph_results = self.graph_retriever.retrieve(query, config)
            for result in graph_results:
                contexts.append({
                    "source": "graph_db",
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0)
                })
        
        return contexts

    def _store_contexts_to_memory(self, contexts: List[Dict[str, Any]], task_prompt: Dict[str, Any]) -> None:
        """Store retrieved contexts to Memory DB."""
        for context in contexts:
            memory_entry = {
                "memory_id": str(uuid.uuid4()),
                "entity": task_prompt.get("entity_name", "context"),
                "query": task_prompt.get("prompt_text", ""),
                "context": context,
                "timestamp": time.time(),
                "source": context.get("source")
            }
            
            # Store in memory DB
            self.memory_retriever.save_memory(memory_entry)
            
            # Add to local tracking
            self.memory_entries.append(memory_entry)

    def _load_memory_contexts(self, task_prompt: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load relevant contexts from Memory DB."""
        query = task_prompt.get("prompt_text", "")
        config = {
            "search": {
                "symbolic": True,
                "semantic": True,
                "hybrid_weight": 0.7
            },
            "limits": {
                "top_k": 10
            }
        }
        
        results = self.memory_retriever.retrieve(query, config)
        
        # Extract contexts
        contexts = []
        for result in results:
            if "context" in result:
                contexts.append(result["context"])
        
        return contexts
        
    def _get_iteration_parameters(self, task: Dict[str, Any], task_prompt: Dict[str, Any]) -> List[Any]:
        """Get iteration parameters for the task."""
        # Check if iterations are specified in the task
        if "iterations" in task:
            return task["iterations"]
        
        # Check if iterations are in memory
        query = f"iterations for {task_prompt.get('prompt_id')}"
        config = {
            "search": {"symbolic": True},
            "limits": {"top_k": 1}
        }
        
        results = self.memory_retriever.retrieve(query, config)
        
        if results and "context" in results[0] and "iterations" in results[0]["context"]:
            return results[0]["context"]["iterations"]
        
        # Default to single iteration with None value
        return [None]
        
    def _contextualize_prompt(self, task_prompt: Dict[str, Any], iteration_value: Any, 
                          contexts: List[Dict[str, Any]]) -> str:
        """Contextualize the prompt with iteration value and contexts."""
        prompt_text = task_prompt.get("prompt_text", "")
        
        # Replace iteration placeholder if present
        if iteration_value is not None:
            prompt_text = prompt_text.replace("{{iteration_value}}", str(iteration_value))
        
        # Add contexts if specified in the task prompt
        if task_prompt.get("include_contexts", False) and contexts:
            context_text = "\n\nContexts:\n"
            for i, context in enumerate(contexts, 1):
                context_text += f"{i}. {context.get('content', '')}\n"
            
            prompt_text += context_text
        
        return prompt_text

    def _classify_task(self, contextualized_prompt: str) -> Tuple[str, str]:
        """Classify the task type and complexity using meta prompt."""
        # Use meta generator to classify
        try:
            classification = self.meta_generator.get_meta_generation(
                application="metaworkflow",
                category="task_classifier",
                action="classify",
                prompt_id=100,
                task_prompt=contextualized_prompt,
                model="Qwen2.5-1.5B",
                temperature=0.3
            )
            
            # Parse the classification result
            if isinstance(classification, dict):
                task_type = classification.get("task_type", "general")
                task_complexity = classification.get("task_complexity", "medium")
            else:
                # Default values if parsing fails
                task_type = "general"
                task_complexity = "medium"
                
        except Exception as e:
            logger.error(f"Task classification failed: {e}")
            # Default values if classification fails
            task_type = "general"
            task_complexity = "medium"
        
        return task_type, task_complexity

    def _get_personality_traits(self, task_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Get personality traits from Memory DB for the task."""
        # Configure memory retriever for personality memory
        personality_retriever = MemoryRetriever(
            memory_db=self.memory_retriever.memory_db,
            generator=self.generator,
            memory_type="personality"
        )
        
        # Query for relevant personality traits
        query = task_prompt.get("prompt_text", "")
        config = {
            "search": {
                "symbolic": True,
                "semantic": True,
                "hybrid_weight": 0.5
            },
            "limits": {
                "top_k": 1
            }
        }
        
        results = personality_retriever.retrieve(query, config)
        
        if results:
            return {
                "mbti_type": results[0].get("mbti_type", ""),
                "personality_type": results[0].get("personality_type", ""),
                "cognitive_style": results[0].get("cognitive_style", ""),
                "mode_description": results[0].get("mode_description", "")
            }
        
        # Default empty traits
        return {}

    def _apply_personality_to_system_prompt(self, system_prompt: str, 
                                        personality_traits: Dict[str, Any]) -> str:
        """Apply personality traits to the system prompt."""
        if not personality_traits:
            return system_prompt
        
        # Add personality information to system prompt
        personality_text = "\n\nPersonality Traits:\n"
        
        if "mbti_type" in personality_traits and personality_traits["mbti_type"]:
            personality_text += f"MBTI Type: {personality_traits['mbti_type']}\n"
        
        if "personality_type" in personality_traits and personality_traits["personality_type"]:
            personality_text += f"Personality Type: {personality_traits['personality_type']}\n"
        
        if "cognitive_style" in personality_traits and personality_traits["cognitive_style"]:
            personality_text += f"Cognitive Style: {personality_traits['cognitive_style']}\n"
        
        if "mode_description" in personality_traits and personality_traits["mode_description"]:
            personality_text += f"Mode Description: {personality_traits['mode_description']}\n"
        
        return system_prompt + personality_text
    
    def _execute_prompt(self, prompt: str, system_prompt: str, 
                       model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the prompt with selected model and parameters."""
        provider = model_config.get("model_provider", "azure_openai")
        model_name = model_config.get("model_name", "gpt-4o-mini")
        parameters = model_config.get("parameters", {})
        topology = model_config.get("tts_topology", {})
        
        # Get template for the specified topology if it exists
        topology_template = self._get_topology_template(topology)
        
        # Apply topology template to prompt if available
        final_prompt = self._apply_topology_to_prompt(prompt, topology_template) if topology_template else prompt
        
        if
        else:
            # Default direct execution
            result = self.generator.get_completion(
                prompt_id=time.time_ns(),
                prompt=final_prompt,
                system_prompt=system_prompt,
                model=model_name,
                temperature=parameters.get("temperature", 0.7),
                max_tokens=parameters.get("max_tokens", 4000),
                top_p=parameters.get("top_p", 1.0),
                frequency_penalty=parameters.get("frequency_penalty", 0.0),
                presence_penalty=parameters.get("presence_penalty", 0.0),
                seed=parameters.get("seed", None),
                json_schema=parameters.get("json_schema", None),
                return_full_response=True
            )
            
            return {
                "content": result.get("content", ""),
                "model": model_name,
                "tokens": result.get("usage", {}).get("total_tokens", 0),
                "topology": "direct"
            }

    def _get_topology_template(self, topology: str) -> Optional[str]:
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