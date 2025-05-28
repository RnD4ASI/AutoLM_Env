import sys
import os
from pathlib import Path
import json
import logging
import time

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from pipeline.orchestration.execution_service import ExecutionService

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    logger.info("Starting ExecutionService test script.")
    start_time = time.time()

    # Define paths
    base_dir = Path.cwd()
    plan_file_path = base_dir / "config/prompt_flow_config.json" # Dummy plan created in previous step
    
    # Other paths (for DBs, task library) will use defaults in ExecutionService,
    # and dummy/copied files have been prepared for those defaults:
    # - config/shortlisted_prompts.json (task library)
    # - db/vector/vector_db_for_retriever.json
    # - db/memory/episodic_memory_for_retriever.json
    # - db/memory/personality_memory_for_retriever.json
    # - db/graph/g_complete.pkl (not created, hoping service handles absence)
    # - config/main_config.json (exists)
    # - config/mle_config.json (exists)

    logger.info(f"Using plan file: {plan_file_path}")

    if not plan_file_path.exists():
        logger.error(f"Plan file not found: {plan_file_path}. Please create it first.")
        # Create a minimal dummy if it's missing, though previous steps should handle this
        plan_file_path.parent.mkdir(exist_ok=True)
        with open(plan_file_path, 'w') as f:
            json.dump({"nodes": {}, "edges": [], "metadata": {"name": "Emergency Dummy Plan"}}, f, indent=2)
        logger.info(f"Created emergency dummy plan at {plan_file_path}")


    try:
        logger.info("Initializing ExecutionService...")
        # Using default paths for DBs, task library, etc.
        # plan_json_path will also use its default (config/prompt_flow_config.json)
        # task_library_json_path will use its default (config/shortlisted_prompts.json)
        execution_service = ExecutionService(
            # No need to pass paths if defaults are used and files are in place
        )
        logger.info("ExecutionService initialized successfully.")

        logger.info("Executing plan...")
        execution_results = execution_service.execute_plan()
        
        logger.info(f"\n--- Plan Execution Results ---")
        logger.info(json.dumps(execution_results, indent=2))

    except FileNotFoundError as fnf_error:
        logger.error(f"A required file was not found: {fnf_error}", exc_info=True)
        logger.error("Please ensure all necessary config, prompt, and dummy DB files are in place.")
    except Exception as e:
        logger.error(f"An error occurred during the ExecutionService test: {e}", exc_info=True)

    finally:
        end_time = time.time()
        logger.info(f"ExecutionService test script finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_test()
