import sys
import os
from pathlib import Path
import json
import logging
import time

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from pipeline.orchestration.plan_service import PlanService

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    logger.info("Starting PlanService test script.")
    start_time = time.time()

    # Define the hypothetical user goal
    hypothetical_goal = (
        "Generate a comprehensive overview of capital adequacy requirements as outlined in APS112 and APS113, "
        "including key definitions, calculation methods, and reporting obligations."
    )
    logger.info(f"Hypothetical Goal: {hypothetical_goal}")

    # Define paths
    base_dir = Path.cwd()
    # PlanService defaults to using config/task_prompt_library.json if no path is given
    # AIUtility (used by PlanService) defaults to config/meta_prompt_library.json
    # These files were copied from db/prompt/example_*.json in a previous step.
    
    # Output files will be saved in config/ by default by PlanService
    output_dir = base_dir / "config"
    plan_output_file = output_dir / "prompt_flow_config.json"
    shortlisted_prompts_file = output_dir / "shortlisted_prompts.json"

    try:
        logger.info("Initializing PlanService...")
        # Not passing config_file_path, PlanService does not strictly require one for its own init
        # It initializes its components like Generator which might have their own defaults or configs.
        plan_service = PlanService()
        logger.info("PlanService initialized successfully.")

        logger.info("Generating plan...")
        # The plan_task_sequence method saves the files itself and does not return the plan directly.
        plan_service.plan_task_sequence(
            goal=hypothetical_goal,
            task_prompt_library_path=None, # Will use default config/task_prompt_library.json
            output_path=output_dir # Explicitly setting output directory
            # Using default for reasoning_model, best_of_n etc.
        )
        logger.info(f"Plan generation process completed by PlanService.")
        logger.info(f"  Generated plan expected at: {plan_output_file}")
        logger.info(f"  Shortlisted prompts expected at: {shortlisted_prompts_file}")

        # Output the generated plan
        if plan_output_file.exists():
            logger.info(f"\n--- Generated Plan (from {plan_output_file}) ---")
            with open(plan_output_file, 'r') as f:
                generated_plan_json = json.load(f)
            logger.info(json.dumps(generated_plan_json, indent=2))
        else:
            logger.error(f"Generated plan file not found at {plan_output_file}")

        if shortlisted_prompts_file.exists():
            logger.info(f"\n--- Shortlisted Prompts (from {shortlisted_prompts_file}) ---")
            with open(shortlisted_prompts_file, 'r') as f:
                shortlisted_prompts_json = json.load(f)
            logger.info(json.dumps(shortlisted_prompts_json, indent=2))
        else:
            logger.warning(f"Shortlisted prompts file not found at {shortlisted_prompts_file}")

    except Exception as e:
        logger.error(f"An error occurred during the PlanService test: {e}", exc_info=True)

    finally:
        end_time = time.time()
        logger.info(f"PlanService test script finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_test()
