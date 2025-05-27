import json
import time
import traceback

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
            library_path = Path(task_prompt_library_path) if library_path else Path.cwd() / "config" / "task_prompt_library.json"
            task_library = self.datautility.text_operation('load', library_path, file_type='json')
            logger.debug(f"Successfully loaded task library from {library_path} with {len(task_library)} prompts")
            
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
            
            # Step 5: Save the planning results
            if output_path is None:
                output_dir = Path.cwd() / "config"
            else:
                output_dir = Path(output_path) if isinstance(output_path, str) else output_path
            
            # Create output directory if it doesn't exist
            self.datautility.ensure_directory(output_dir)
            
            # Save prompt flow config
            flow_path = output_dir / "prompt_flow_config.json"
            self.datautility.text_operation('save', flow_path, prompt_flow_config, file_type='json', indent=2)
            
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