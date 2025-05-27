import json
import os
import uuid
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.logging import get_logger
from src.utility import AIUtility, DataUtility
from src.generator import Generator, MetaGenerator
from src.evaluator import Evaluator
from src.contextualiser import ContextManager
from src.topologist import PromptTopology, ScalingTopology


# TO DO: significant replanning, boundary with task_executor.


logger = get_logger(__name__)

class ExecutionEngine:
    """
    Handles task execution with optimization based on mle_config.json.
    """
    
    def __init__(self, config: Dict[str, Any], context_manager: ContextManager):
        """Initialize ExecutionEngine with configuration and context manager."""
        self.config = config
        self.context_manager = context_manager
        
        # Initialize components
        self.generator = Generator()
        self.metagenerator = MetaGenerator(generator=self.generator)
        self.evaluator = Evaluator(generator=self.generator)
        self.aiutility = AIUtility()
        self.datautility = DataUtility()
        
        # Initialize topology and optimization components
        self.prompt_topology = PromptTopology(generator=self.generator)
        self.scaling_topology = ScalingTopology(generator=self.generator)
        
        # Load MLE configuration
        self.mle_config = self._load_mle_config()
        
        logger.debug("ExecutionManager initialized")
    
    def _load_mle_config(self) -> Dict[str, Any]:
        """Load MLE configuration from file."""
        try:
            config_path = Path.cwd() / "config" / "mle_config.json"
            mle_config = self.datautility.text_operation('load', config_path, file_type='json')
            logger.debug(f"Successfully loaded MLE configuration from {config_path}")
            return mle_config
        except Exception as e:
            logger.warning(f"Failed to load MLE configuration: {e}")
            # Return default configuration
            return {
                "defaults": {
                    "model_fallback_order": ["azure_openai", "anthropic", "huggingface"],
                    "topology_fallback_order": ["direct"],
                    "temperature_fallback_order": [0.7]
                },
                "llm_method_selection": {
                    "deduction": {
                        "medium": {
                            "method": {
                                "model_provider": "huggingface",
                                "model_name": "Qwen2.5-1.5B",
                                "tts_topology": "direct",
                                "parameters": {"temperature": 0.7}
                            }
                        }
                    }
                }
            }

    def execute_task(self,
                    task_id: str,
                    task_prompt: Dict[str, Any],
                    task_type: Optional[str] = None,
                    confidence_level: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a task with optimization based on mle_config.json.
        
        Args:
            task_id: Unique identifier for the task
            task_prompt: Task prompt data from prompt_task_chain.json format
            task_type: Optional override for task type (if not specified, will be inferred)
            confidence_level: Optional override for confidence level (if not specified, will be inferred)
            
        Returns:
            Dictionary containing:
                - task_id: Task identifier
                - response: Generated response
                - metadata: Execution metadata including model used, topology, etc.
                - performance_metrics: Timing and quality metrics
        """
        logger.info(f"Starting task execution for task {task_id}")
        start_time = time.time()
        
        try:
            # Step 1: Determine task type and confidence level
           
            # Step 2: Get execution configuration
            execution_config = self._get_execution_config(task_type, confidence_level)
            logger.debug(f"Selected execution config: {execution_config}")
            
            # Step 3: Prepare context from memory
            context_data = self._prepare_context(task_id, task_prompt)
            
            # Step 4: Apply prompt optimization if configured
            optimized_prompt = self._apply_prompt_optimization(task_prompt, execution_config)
            
            # Step 5: Execute with selected topology
            response, execution_metadata = self._execute_with_topology(
                task_id=task_id,
                task_prompt=optimized_prompt,
                execution_config=execution_config,
                context_data=context_data
            )
            
            # Step 6: Evaluate response quality
            quality_metrics = self._evaluate_response(optimized_prompt, response)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Prepare result
            result = {
                'task_id': task_id,
                'response': response,
                'metadata': {
                    'task_type': task_type,
                    'confidence_level': confidence_level,
                    'execution_config': execution_config,
                    'context_used': len(context_data) > 0,
                    **execution_metadata
                },
                'performance_metrics': {
                    'execution_time': execution_time,
                    'quality_metrics': quality_metrics
                }
            }
            
            logger.info(f"Task {task_id} completed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed for task {task_id}: {str(e)}")
            logger.debug(f"Execution error details: {traceback.format_exc()}")
            
            # Attempt fallback execution
            try:
                logger.info(f"Attempting fallback execution for task {task_id}")
                fallback_result = self._handle_execution_fallback(task_id, task_prompt)
                fallback_result['metadata']['fallback_used'] = True
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback execution also failed for task {task_id}: {fallback_error}")
                raise
    



    def _get_execution_config(self, task_type: str, confidence_level: str) -> Dict[str, Any]:
        """
        Get execution configuration based on task type and confidence level.
        
        Args:
            task_type: Type of task
            confidence_level: Confidence level (high/medium/low)
            
        Returns:
            Execution configuration dictionary
        """
        try:
            # Get configuration from mle_config
            llm_selection = self.mle_config.get('llm_method_selection', {})
            
            if task_type in llm_selection and confidence_level in llm_selection[task_type]:
                config = llm_selection[task_type][confidence_level]['method'].copy()
                logger.debug(f"Found specific config for {task_type}/{confidence_level}")
                return config
            
            # Try fallback with medium confidence
            if task_type in llm_selection and 'medium' in llm_selection[task_type]:
                config = llm_selection[task_type]['medium']['method'].copy()
                logger.warning(f"Using medium confidence fallback for {task_type}/{confidence_level}")
                return config
            
            # Try fallback with deduction task type
            if 'deduction' in llm_selection and confidence_level in llm_selection['deduction']:
                config = llm_selection['deduction'][confidence_level]['method'].copy()
                logger.warning(f"Using deduction fallback for {task_type}/{confidence_level}")
                return config
            
            # Ultimate fallback
            defaults = self.mle_config.get('defaults', {})
            config = {
                'model_provider': defaults.get('model_fallback_order', ['huggingface'])[0],
                'model_name': 'Qwen2.5-1.5B',
                'tts_topology': defaults.get('topology_fallback_order', ['direct'])[0],
                'parameters': {'temperature': defaults.get('temperature_fallback_order', [0.7])[0]}
            }
            logger.warning(f"Using ultimate fallback config for {task_type}/{confidence_level}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to get execution config: {e}")
            # Return minimal safe config
            return {
                'model_provider': 'huggingface',
                'model_name': 'Qwen2.5-1.5B',
                'tts_topology': 'direct',
                'parameters': {'temperature': 0.7}
            }
    
    def _prepare_context(self, task_id: str, task_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context data from memory for task execution.
        
        Args:
            task_id: Task identifier
            task_prompt: Task prompt data
            
        Returns:
            Context data dictionary
        """
        try:
            # Assess if current memory is sufficient, retrieve extra context if needed
            is_sufficient, additional_query = self.context_manager.assess_context_sufficiency(task_prompt, task_id)
            if not is_sufficient and additional_query:
                self.context_manager.retrieve_task_specific_context(task_id, additional_query)

            context_data = {}

            # Get task-specific memory
            task_memory = self.context_manager.get_task_memory(task_id)
            if task_memory:
                context_data['task_contexts'] = task_memory
                logger.debug(f"Found {len(task_memory)} task-specific memory entries")

            # Get goal memory (common contexts)
            goal_memory = self.context_manager.get_goal_memory()
            if goal_memory:
                context_data['goal_contexts'] = goal_memory
                logger.debug(f"Found {len(goal_memory)} goal memory entries")

            # Get procedural memory
            procedural_memory = self.context_manager.get_procedural_memory()
            if procedural_memory:
                context_data['procedural_contexts'] = procedural_memory
                logger.debug(f"Found {len(procedural_memory)} procedural memory entries")

            return context_data

        except Exception as e:
            logger.warning(f"Failed to prepare context for task {task_id}: {e}")
            return {}

    def _apply_prompt_optimization(self, task_prompt: Dict[str, Any], execution_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply prompt optimization based on execution configuration.
        
        Args:
            task_prompt: Original task prompt
            execution_config: Execution configuration
            
        Returns:
            Optimized task prompt
        """
        try:
            optimization_method = execution_config.get('prompt_optimization', 'none')
            
            if optimization_method == 'none':
                logger.debug("No prompt optimization configured")
                return task_prompt
            
            logger.debug(f"Applying prompt optimization: {optimization_method}")
            
            if optimization_method == 'genetic_algorithm':
                # Use genetic algorithm topology for prompt optimization
                optimized_prompts, _ = self.prompt_topology.prompt_genetic_algorithm(
                    task_prompt=str(task_prompt),
                    prompt_id=1003,
                    num_variations=3,
                    num_evolution=2,
                    model="Qwen2.5-1.5B",
                    temperature=0.7,
                    return_full_response=True
                )
                # Parse the best optimized prompt back to dictionary format
                if optimized_prompts and len(optimized_prompts) > 0:
                    optimized_prompt = self.aiutility.format_json_response(optimized_prompts[0])
                    return optimized_prompt
            
            elif optimization_method == 'disambiguation':
                # Use disambiguation topology
                _optimized_prompt, _ = self.prompt_topology.prompt_disambiguation(
                    task_prompt=str(task_prompt),
                    prompt_id=1004,
                    model="Qwen2.5-1.5B",
                    temperature=0.5,
                    return_full_response=False
                )
                # Parse back to dictionary format
                optimized_prompt = self.aiutility.format_json_response(_optimized_prompt)
                return optimized_prompt
            
            elif optimization_method in ['chain_of_thought', 'tree_of_thought', 'program_synthesis', 'deep_thought']:
                # Use template adopter for reasoning enhancement
                optimized_prompt = self.template_adopter.get_prompt_transformation(
                    prompt_dict=task_prompt,
                    fix_template=optimization_method
                )
                return optimized_prompt
            
            else:
                logger.warning(f"Unknown optimization method: {optimization_method}")
                return task_prompt
                
        except Exception as e:
            logger.warning(f"Prompt optimization failed: {e}, using original prompt")
            return task_prompt
    
    def _execute_with_topology(self,
                              task_id: str,
                              task_prompt: Dict[str, Any],
                              execution_config: Dict[str, Any],
                              context_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Execute task using the specified topology.
        
        Args:
            task_id: Task identifier
            task_prompt: Task prompt (potentially optimized)
            execution_config: Execution configuration
            context_data: Context data from memory
            
        Returns:
            Tuple of (response, execution_metadata)
        """
        start_time = time.time()
        
        # Extract configuration parameters
        model = execution_config.get('model', 'Qwen2.5-1.5B')
        temperature = execution_config.get('temperature', 0.7)
        topology = execution_config.get('topology', 'default')
        topology_params = execution_config.get('topology_params', {})
        
        logger.info(f"Executing task {task_id} with topology {topology}")
        
        # Prepare contextual prompt by combining task prompt with context
        contextual_prompt = self._prepare_contextual_prompt(task_prompt, context_data)
        
        # Execute with specified topology
        try:
            response = None
            performance_metrics = {}
            execution_details = {}
            
            # Track execution time
            topology_start_time = time.time()
            
            # Choose topology
            if topology == "default":
                # Direct execution without specific topology
                response = self.generator.get_completion(
                    prompt_id=task_id,
                    prompt=contextual_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                execution_details = {"type": "direct_execution"}
                
            elif topology == "prompt_disambiguation":
                # Topology 1: Prompt disambiguation
                result = self.prompt_topology.prompt_disambiguation(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                response = result[1]  # Optimized response
                execution_details = {
                    "type": "prompt_disambiguation",
                    "original_prompt": result[0][0],
                    "optimized_prompt": result[0][1],
                    "original_response": result[1][0],
                    "optimized_response": result[1][1]
                }
                
            elif topology == "genetic_algorithm":
                # Topology 2: Genetic algorithm
                num_variations = topology_params.get('num_variations', 5)
                num_evolution = topology_params.get('num_evolution', 3)
                result = self.prompt_topology.prompt_genetic_algorithm(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_variations=num_variations,
                    num_evolution=num_evolution,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                response = result[1]  # Best response
                execution_details = {
                    "type": "genetic_algorithm",
                    "prompt_variations": result[0],
                    "responses": result[1],
                    "num_variations": num_variations,
                    "num_evolution": num_evolution
                }
                
            elif topology == "best_of_n_synthesis":
                # Topology 3: Best of N synthesis
                num_variations = topology_params.get('num_variations', 3)
                result = self.scaling_topology.best_of_n_synthesis(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_variations=num_variations,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                if isinstance(result, dict):
                    response = result.get('response')
                    execution_details = {
                        "type": "best_of_n_synthesis",
                        "num_variations": num_variations,
                        "variations": result.get('variations')
                    }
                else:
                    response = result
                    execution_details = {
                        "type": "best_of_n_synthesis",
                        "num_variations": num_variations
                    }
            
            elif topology == "best_of_n_selection":
                # Topology 4: Best of N selection
                num_variations = topology_params.get('num_variations', 3)
                selection_method = topology_params.get('selection_method', 'llm')
                response = self.scaling_topology.best_of_n_selection(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_variations=num_variations,
                    selection_method=selection_method,
                    model=model,
                    model_selector=model,  # Using same model for selection
                    temperature=temperature,
                    **topology_params
                )
                execution_details = {
                    "type": "best_of_n_selection",
                    "num_variations": num_variations,
                    "selection_method": selection_method
                }
                
            elif topology == "self_reflection":
                # Topology 5: Self-reflection
                num_iterations = topology_params.get('num_iterations', 1)
                result = self.scaling_topology.self_reflection(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_iterations=num_iterations,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                if isinstance(result, dict):
                    response = result.get('final_response')
                    execution_details = {
                        "type": "self_reflection",
                        "num_iterations": num_iterations,
                        "iterations": result.get('iterations')
                    }
                else:
                    response = result
                    execution_details = {
                        "type": "self_reflection",
                        "num_iterations": num_iterations
                    }
                
            elif topology == "chain_of_thought":
                # Topology 6: Chain-of-thought reasoning
                template = topology_params.get('template', 'chain_of_thought')
                transformed_prompt = self.prompt_topology.prompt_reasoning(
                    task_prompt=contextual_prompt,
                    template=template
                )
                response = self.generator.get_completion(
                    prompt_id=task_id,
                    prompt=transformed_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                execution_details = {
                    "type": "chain_of_thought",
                    "template": template,
                    "transformed_prompt": transformed_prompt
                }
                
            elif topology == "multi_agent_debate":
                # Topology 8: Multi-agent debate
                num_iterations = topology_params.get('num_iterations', 2)
                model_strong = topology_params.get('model_strong', model)
                model_weak = topology_params.get('model_weak', model)
                selection_method = topology_params.get('selection_method', 'llm')
                result = self.scaling_topology.multi_agent_debate(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_iterations=num_iterations,
                    model_strong=model_strong,
                    model_weak=model_weak,
                    selection_method=selection_method,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                if isinstance(result, dict):
                    response = result.get('final_response')
                    execution_details = {
                        "type": "multi_agent_debate",
                        "num_iterations": num_iterations,
                        "model_strong": model_strong,
                        "model_weak": model_weak,
                        "iterations": result.get('iterations'),
                        "selection_method": selection_method
                    }
                else:
                    response = result
                    execution_details = {
                        "type": "multi_agent_debate",
                        "num_iterations": num_iterations,
                        "model_strong": model_strong,
                        "model_weak": model_weak,
                        "selection_method": selection_method
                    }
            
            elif topology == "regenerative_majority_synthesis":
                # Topology: Regenerative Majority Synthesis
                num_initial_responses = topology_params.get('num_initial_responses', 3)
                num_regen_responses = topology_params.get('num_regen_responses', 3)
                cut_off_fraction = topology_params.get('cut_off_fraction', 0.5)
                synthesis_method = topology_params.get('synthesis_method', 'majority_vote')
                
                result = self.scaling_topology.regenerative_majority_synthesis(
                    task_prompt=contextual_prompt,
                    prompt_id=task_id,
                    num_initial_responses=num_initial_responses,
                    num_regen_responses=num_regen_responses,
                    cut_off_fraction=cut_off_fraction,
                    synthesis_method=synthesis_method,
                    model=model,
                    temperature=temperature,
                    return_full_response=True,
                    **topology_params
                )
                
                if isinstance(result, dict):
                    response = result.get('final_response')
                    execution_details = {
                        "type": "regenerative_majority_synthesis",
                        "num_initial_responses": num_initial_responses,
                        "num_regen_responses": num_regen_responses,
                        "cut_off_fraction": cut_off_fraction,
                        "synthesis_method": synthesis_method,
                        "performance_metrics": result.get('performance_metrics')
                    }
                else:
                    response = result
                    execution_details = {
                        "type": "regenerative_majority_synthesis",
                        "num_initial_responses": num_initial_responses,
                        "num_regen_responses": num_regen_responses,
                        "cut_off_fraction": cut_off_fraction,
                        "synthesis_method": synthesis_method
                    }
            
            else:
                # Default case for unknown topology
                logger.warning(f"Unknown topology '{topology}'. Falling back to default execution.")
                response = self.generator.get_completion(
                    prompt_id=task_id,
                    prompt=contextual_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                execution_details = {
                    "type": "default_fallback",
                    "reason": f"Unknown topology '{topology}'"
                }
            
            # Calculate execution time
            topology_time = time.time() - topology_start_time
            total_time = time.time() - start_time
            
            # Prepare performance metrics
            performance_metrics = {
                "topology_execution_time": topology_time,
                "total_execution_time": total_time,
                "model": model,
                "topology": topology
            }
            
            # Prepare execution metadata
            execution_metadata = {
                "task_id": task_id,
                "model": model,
                "temperature": temperature,
                "topology": topology,
                "execution_details": execution_details,
                "performance_metrics": performance_metrics
            }
            
            # Update iteration count in prompt_flow_config.json if needed
            self._update_task_iteration_count(task_id, task_prompt)
            
            logger.info(f"Task {task_id} executed successfully with topology '{topology}' in {total_time:.2f} seconds")
            return response, execution_metadata
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Error executing task {task_id} with topology '{topology}': {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Create error metadata
            error_metadata = {
                "task_id": task_id,
                "model": model,
                "temperature": temperature,
                "topology": topology,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": error_time
            }
            
            # Return None with error metadata
            return None, error_metadata

    def _update_task_iteration_count(self, task_id: str, task_prompt: Dict[str, Any]) -> None:
        """
        Update iteration count for a task in prompt_flow_config.json.
        
        Args:
            task_id: Task identifier
            task_prompt: Task prompt data
        """
        try:
            # Get the prompt flow config path
            config_path = Path.cwd() / "config" / "prompt_flow_config.json"
            
            # Load existing config if it exists
            if config_path.exists():
                prompt_flow_config = self.datautility.text_operation('load', config_path, file_type='json')
            else:
                prompt_flow_config = {"task_iterations": {}}
            
            # Initialize task_iterations if not present
            if "task_iterations" not in prompt_flow_config:
                prompt_flow_config["task_iterations"] = {}
            
            # Update iteration count for the task
            if task_id in prompt_flow_config["task_iterations"]:
                prompt_flow_config["task_iterations"][task_id] += 1
            else:
                prompt_flow_config["task_iterations"][task_id] = 1
            
            # Save updated config
            self.datautility.text_operation('save', config_path, prompt_flow_config, file_type='json')
            
            logger.debug(f"Updated iteration count for task {task_id} to {prompt_flow_config['task_iterations'][task_id]}")
            
        except Exception as e:
            logger.warning(f"Failed to update task iteration count for {task_id}: {e}")
    
    def _prepare_contextual_prompt(self, task_prompt: Dict[str, Any], context_data: Dict[str, Any]) -> str:
        """
        Prepare contextual prompt by combining task prompt with context data.
        
        Args:
            task_prompt: Task prompt data
            context_data: Context data from memory
            
        Returns:
            Contextualized prompt string
        """
        # Extract task prompt text
        prompt_text = task_prompt.get('prompt_text', '')

        # Prepare context sections
        context_sections = []

        # Add goal contexts
        goal_contexts = context_data.get('goal_contexts', {})
        if goal_contexts:
            context_str = "\n\n### Goal Context:\n"
            for entry in goal_contexts:
                context_str += f"\n- {entry}\n"
            context_sections.append(context_str)

        # Add task-specific contexts
        task_contexts = context_data.get('task_contexts', {})
        if task_contexts:
            context_str = "\n\n### Task-Specific Contexts:\n"
            for entry in task_contexts:
                context_str += f"\n- {entry['original_query']}\n"
            context_sections.append(context_str)

        # Add procedural contexts
        procedural_contexts = context_data.get('procedural_contexts', {})
        if procedural_contexts:
            context_str = "\n\n### Procedural Context:\n"
            for entry in procedural_contexts:
                context_str += f"\n- {entry['config_name']}: {entry['config_content']}\n"
            context_sections.append(context_str)

        # Combine context sections with prompt
        if context_sections:
            context_block = "\n".join(context_sections)
            contextual_prompt = f"{prompt_text}\n\n{context_block}"
        else:
            contextual_prompt = prompt_text

        logger.debug(f"Prepared contextual prompt with {len(context_sections)} context sections")
        return contextual_prompt
    
    def _evaluate_response(self, prompt: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate response quality.
        
        Args:
            prompt: Task prompt
            response: Generated response
            
        Returns:
            Quality metrics dictionary
        """
        try:
            # Basic metrics
            metrics = {
                "response_length": len(response),
                "response_word_count": len(response.split())
            }
            
            # Add more sophisticated evaluation if needed
            # For example, using the evaluator component
            
            return metrics
        except Exception as e:
            logger.warning(f"Response evaluation failed: {e}")
            return {"evaluation_error": str(e)}
    # To Check Usaefulness
    def _handle_execution_fallback(self, task_id: str, task_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle fallback execution when primary execution fails.
        
        Args:
            task_id: Task identifier
            task_prompt: Task prompt data
            
        Returns:
            Fallback execution result
        """
        logger.info(f"Using fallback execution for task {task_id}")
        
        try:
            # Use simplest possible execution with most reliable model
            fallback_model = self.mle_config.get('defaults', {}).get('fallback_model', 'Qwen2.5-1.5B')
            fallback_temp = self.mle_config.get('defaults', {}).get('fallback_temperature', 0.3)
            
            # Extract prompt text
            prompt_text = ""
            if isinstance(task_prompt, dict):
                prompt_text = task_prompt.get('prompt_text', str(task_prompt))
            else:
                prompt_text = str(task_prompt)
            
            # Direct execution
            response = self.generator.get_completion(
                prompt_id=f"{task_id}_fallback",
                prompt=prompt_text,
                model=fallback_model,
                temperature=fallback_temp,
                return_full_response=False
            )
            
            # Create minimal result
            result = {
                'task_id': task_id,
                'response': response,
                'metadata': {
                    'task_type': 'unknown',
                    'confidence_level': 'low',
                    'execution_config': {
                        'model': fallback_model,
                        'temperature': fallback_temp,
                        'topology': 'direct'
                    },
                    'context_used': False
                },
                'performance_metrics': {
                    'execution_time': 0,
                    'quality_metrics': {
                        'response_length': len(response),
                        'response_word_count': len(response.split())
                    }
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback execution also failed: {e}")
            raise
    
    def orchestrate_execution_flow(self, plan: List[Dict[str, Any]], overall_goal_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Orchestrates the execution of a sequence of tasks based on a plan.

        Args:
            plan: A list of task definitions. Each task definition is a dictionary
                  expected to contain 'task_id', 'task_prompt', and optionally
                  'task_type', 'confidence_level'.
            overall_goal_context: Optional string describing the overarching goal for context.

        Returns:
            A list of dictionaries, where each dictionary contains the execution
            result for a corresponding task in the plan.
        """
        logger.info(f"Starting orchestration for a plan with {len(plan)} tasks.")
        all_results = []

        for i, task_definition in enumerate(plan):
            task_id = task_definition.get("task_id", f"task_{uuid.uuid4()}")
            task_prompt = task_definition.get("task_prompt")
            task_type = task_definition.get("task_type")
            confidence_level = task_definition.get("confidence_level")

            if not task_prompt:
                logger.warning(f"Skipping task {i+1} (ID: {task_id}) due to missing 'task_prompt'.")
                all_results.append({
                    "task_id": task_id,
                    "status": "skipped",
                    "reason": "Missing task_prompt"
                })
                continue

            logger.info(f"Executing task {i+1}/{len(plan)}: ID {task_id}")
            
            try:
                # Execute the task
                execution_result = self.execute_task(
                    task_id=task_id,
                    task_prompt=task_prompt,
                    task_type=task_type,
                    confidence_level=confidence_level
                )
                all_results.append(execution_result)

                # Store the result in memory
                if execution_result and execution_result.get("response") is not None:
                    self.context_manager.store_goal_memory(
                        entity_name=f"task_result_{task_id}",
                        entity_information=execution_result, # Store the whole result package
                        information_type="task_execution_result",
                        goal_context=overall_goal_context or f"Result of orchestrated task {task_id}"
                    )
                    logger.info(f"Stored result for task {task_id} in goal memory.")
                elif execution_result:
                    logger.warning(f"Task {task_id} executed but produced no response or an error. Result: {execution_result}")
                else:
                    logger.error(f"Task {task_id} execution returned None. Not storing in memory.")

            except Exception as e:
                logger.error(f"Critical error during orchestration of task {task_id}: {e}")
                logger.error(traceback.format_exc())
                all_results.append({
                    "task_id": task_id,
                    "response": None,
                    "metadata": {"error": f"Orchestration failure: {str(e)}"},
                    "performance_metrics": {}
                })
        
        # Update prompt_flow_config.json with completion of this orchestration
        try:
            config_path = Path.cwd() / "config" / "prompt_flow_config.json"
            if config_path.exists():
                prompt_flow_config = self.datautility.text_operation('load', config_path, file_type='json')
            else:
                prompt_flow_config = {}
            
            # Record orchestration completion
            if "orchestrations" not in prompt_flow_config:
                prompt_flow_config["orchestrations"] = []
            
            orchestration_record = {
                "timestamp": time.time(),
                "tasks_count": len(plan),
                "completed_tasks": len([r for r in all_results if r.get("response") is not None]),
                "goal_context": overall_goal_context,
                "task_ids": [task.get("task_id") for task in plan if "task_id" in task]
            }
            
            prompt_flow_config["orchestrations"].append(orchestration_record)
            self.datautility.text_operation('save', config_path, prompt_flow_config, file_type='json')
            
        except Exception as e:
            logger.warning(f"Failed to update prompt_flow_config.json with orchestration record: {e}")
        
        logger.info(f"Orchestration completed. Processed {len(all_results)} tasks.")
        return all_results


