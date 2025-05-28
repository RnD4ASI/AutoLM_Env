import json
import time
import traceback
import uuid # Added for graph_id

from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

from src.pipeline.shared.logging import get_logger
from src.pipeline.shared.utility import AIUtility, DataUtility
from src.pipeline.processing.generator import Generator, MetaGenerator
from src.pipeline.processing.topologist import PromptTopology, ScalingTopology
from src.pipeline.processing.evaluator import Evaluator

logger = get_logger(__name__)

class PlanService:
    """
    Responsible for planning task execution by selecting and sequencing relevant task prompts
    based on a user-provided goal and the task prompt library.
    """
    
    def __init__(self, 
                 config_file_path: Optional[Union[str, Path]] = None,
                 generator: Optional[Generator] = None):
        """
        Initialize the PlanService.
        
        Args:
            config_file_path: Path to the configuration file. If None, uses default path.
            generator: Optional Generator instance. If None, creates a new one.
        """
        logger.debug("PlanService initialization started")
        try:
            start_time = time.time()
            
            # Initialize utilities and components
            self.generator = generator if generator else Generator()
            self.metagenerator = MetaGenerator(generator=self.generator)
            self.aiutility = AIUtility()
            self.datautility = DataUtility()
            self.evaluator = Evaluator(generator=self.generator)
            self.prompt_topologist = PromptTopology(generator=self.generator)
            self.scale_topologist = ScalingTopology(generator=self.generator)
            
            # Planning configuration
            self.default_max_prompts = 50
            self.default_temperature = 0.7
            self.default_model = "Qwen2.5-1.5B"
            self.default_embedding_model = "Jina-embeddings-v3"
            
            logger.debug(f"TaskPlanner initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"TaskPlanner initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def plan_task_sequence(self,
                          goal: str,
                          task_prompt_library_path: Optional[Union[str, Path]] = None,
                          output_path: Optional[Union[str, Path]] = None,
                          max_prompts: Optional[int] = None,
                          reasoning_model: Optional[str] = None,
                          best_of_n: Optional[int] = None) -> None:
        """
        Plan task sequence by selecting and ordering relevant prompts based on the goal.
        Uses reasoning model-based approaches for both selection and sequencing.
        
        Args:
            goal: User-provided goal description
            task_prompt_library_path: Path to task prompt library JSON file
            max_prompts: Maximum number of prompts to select (if None, uses all selected by reasoning model)
            reasoning_model: Model to use for reasoning-based selection and sequencing
            temperature: Temperature for meta-generation calls
            best_of_n: Number of candidates to generate for best-of-n selection topology
            
        Returns:
            Tuple containing:
                - prompt_flow_config: Dictionary in prompt_flow_config format
                - shortlisted_prompts: Dictionary of selected prompts in task_prompt_library format
        """
        logger.info(f"Starting task sequence planning for goal: '{goal[:50]}{'...' if len(goal) > 50 else ''}'")
        start_time = time.time()
        
        # Set defaults
        max_prompts = max_prompts or self.default_max_prompts # not get used
        reasoning_model = reasoning_model or self.default_model
        best_of_n = best_of_n or 3  # Default to 3 candidates for best-of-n selection
        
        try:
            # Step 1: Load task prompt library
            library_path = Path(task_prompt_library_path) if task_prompt_library_path else Path.cwd() / "config" / "task_prompt_library.json"
            raw_task_library = self.datautility.text_operation('load', library_path, file_type='json')
            
            # Adjust task_library if it's nested according to schema
            if isinstance(raw_task_library, dict) and "collections" in raw_task_library and \
               isinstance(raw_task_library["collections"], dict) and "task_prompt_templates" in raw_task_library["collections"]:
                task_library = raw_task_library["collections"]["task_prompt_templates"]
                logger.info(f"Adjusted task library to use 'collections.task_prompt_templates' structure from {library_path}.")
            else:
                task_library = raw_task_library # Assume it's already the direct dictionary of prompts
                logger.warning(f"Task library from {library_path} does not match the full schema structure (collections.task_prompt_templates). Proceeding with loaded data directly.")
            
            if not isinstance(task_library, dict) or not task_library:
                logger.error(f"Task library is empty or not a dictionary after potential adjustment. Path: {library_path}")
                # Depending on desired behavior, could raise error or return empty results
                return None # Or appropriate error response

            logger.debug(f"Successfully processed task library from {library_path}, found {len(task_library)} prompts.")
            
            # Step 2.1: Select relevant prompts based on goal using reasoning model (i.e. build nodes)
            selection_prompt = self.aiutility.apply_meta_prompt(
                application="metaworkflow",
                category="planner",
                action="select",
                goal=goal,
                task_prompt_library = task_library
            )

            selection_prompt_ids = self.scale_topologist.best_of_n_selection(
                    task_prompt=selection_prompt,
                    prompt_id=1,
                    num_variations=best_of_n,
                    selection_method="llm",  # use a LLM to select one from n options of task prompt set
                    model=reasoning_model,
                    model_selector=reasoning_model
            )
            logger.info(f"Selected {len(selection_prompt_ids)} relevant prompts using reasoning model")
            
            # Step 2.2: Extract the selected prompts
            selected_prompts_data = {}
            for prompt_id in selection_prompt_ids:
                if prompt_id in task_library:
                    selected_prompts_data[prompt_id] = task_library[prompt_id]
                else:
                    logger.warning(f"Selected prompt ID '{prompt_id}' not found in task library")

            # Step 3.1: Sequence the selected prompts (i.e. build edges)
            sequencing_prompt = self.aiutility.apply_meta_prompt(
                application="metaworkflow",
                category="planner",
                action="sequence",
                goal=goal,
                selected_prompts=selected_prompts_data
            )
            # TO DO: ensure "sequence" meta prompt reinforce a specific json schema, in line with prompt_flow_config_example.json

            _prompt_flow_config = self.scale_topologist.best_of_n_selection(
                    task_prompt=sequencing_prompt,
                    prompt_id=1,
                    num_variations=best_of_n,
                    selection_method="llm",  # use a LLM to select one from n options of task prompt set
                    model=reasoning_model,
                    model_selector=reasoning_model
            )
            logger.info(f"Sequenced {len(_prompt_flow_config.get('nodes', {}))} prompts using reasoning model")

            # Step 3.2: Optimise the sequence of the selected prompts (i.e. optimise edges)
            # pm.json to come in as input
            # de-prioritise DAG optimisation - see critique_planner_opp1.md
            prompt_flow_config = _prompt_flow_config

            # Step 3.3: Extract the selected, sequenced and optimised prompts
            selected_sequenced_prompts_data = {}
            for prompt_id in selection_prompt_ids:
                if prompt_id in task_library:
                    selected_sequenced_prompts_data[prompt_id] = task_library[prompt_id]
                else:
                    logger.warning(f"Selected prompt ID '{prompt_id}' not found in task library")

            # Step 4.1: Include user feedback on the sequencing and selection
            # de-prioritise user feedback - see critique_planner_opp2.md

            # Step 4.2: Extract the details of the shortlisted prompts
            # selected_sequenced_prompts_data = {}
            # for prompt_id in selection_prompt_ids:
            #     if prompt_id in task_library:
            #         selected_sequenced_prompts_data[prompt_id] = task_library[prompt_id]
            #     else:
            #         logger.warning(f"Selected prompt ID '{prompt_id}' not found in task library")


            planning_time = time.time() - start_time
            logger.info(f"Task sequence planning completed in {planning_time:.2f} seconds")            
            
            planning_time = time.time() - start_time
            logger.info(f"Task sequence planning completed in {planning_time:.2f} seconds")            
            
            # Step 5: Prepare and Save the planning results in schema_plan.json format
            if output_path is None:
                output_dir = Path.cwd() / "config"
            else:
                output_dir = Path(output_path) if isinstance(output_path, str) else output_path
            
            self.datautility.ensure_directory(output_dir)
            
            graph_id = str(uuid.uuid4())
            schema_conformant_plan = {
                "version": "1.0.0", # From schema_plan.json
                "description": f"AutoLM generated plan for goal: {goal}",
                "collections": { # Adding collections structure
                    "graphs": {
                        graph_id: {
                            "graph_id": graph_id, # Explicitly adding graph_id inside the graph object
                            **prompt_flow_config, # This includes 'nodes' and 'edges'
                            "name": f"Plan for: {goal[:50]}...",
                            "description": goal,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "created_by": "PlanService",
                            "tags": ["automated_plan"],
                            "metadata": {
                                "goal": goal,
                                "reasoning_model": reasoning_model,
                                "best_of_n_used": best_of_n
                            }
                        }
                    }
                }
            }
            
            # Save prompt flow config (now schema_conformant_plan)
            flow_path = output_dir / "prompt_flow_config.json"
            self.datautility.text_operation('save', flow_path, schema_conformant_plan, file_type='json', indent=2)
            
            # Save shortlisted prompts
            prompts_path = output_dir / "shortlisted_prompts.json"
            self.datautility.text_operation('save', prompts_path, selected_sequenced_prompts_data, file_type='json', indent=2)
            
            logger.info(f"Saved planning results to {flow_path} and {prompts_path}")

            return None

        except Exception as e:
            logger.error(f"Task sequence planning failed: {str(e)}")
            logger.debug(f"Planning error details: {traceback.format_exc()}")
            raise
     
    
    def optimise_sequence():
        return None
    
    def incorporate_user_feedback():
        return None