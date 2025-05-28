import sys
import os
from pathlib import Path
import logging
import time

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from pipeline.orchestration.knowledge_service import KnowledgeService

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    logger.info("Starting KnowledgeService test script.")
    start_time = time.time()

    # Define file paths
    # Assuming the script is run from the root of the repository
    base_dir = Path.cwd()
    doc1_path = base_dir / "db/raw/au/standard/APS112.md"
    doc2_path = base_dir / "db/raw/au/standard/APS113.md"
    files_to_process = [doc1_path, doc2_path]

    config_file = base_dir / "config/main_config.json"

    # Define output directories
    vector_dir = base_dir / "db/vector"
    graph_dir = base_dir / "db/graph"
    memory_dir = base_dir / "db/memory"
    
    # Ensure input files exist
    for f_path in files_to_process:
        if not f_path.exists():
            logger.error(f"Input document not found: {f_path}")
            logger.error("Please ensure the 'db/raw/au/standard/APS112.md' and 'APS113.md' files exist.")
            logger.error("You might need to create dummy files or point to existing ones for the test to run.")
            # Create dummy files if they don't exist, so the test can proceed with pipeline logic.
            f_path.parent.mkdir(parents=True, exist_ok=True)
            with open(f_path, 'w') as f:
                f.write(f"# Dummy content for {f_path.name}\nThis is a test document.")
            logger.info(f"Created dummy file: {f_path}")

    # Create dummy hierarchy file if it doesn't exist
    hierarchy_csv_path = base_dir / "db/raw/hierarchy/APS_Hierarchy.csv"
    if not hierarchy_csv_path.exists():
        hierarchy_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hierarchy_csv_path, 'w') as f:
            f.write("heading_id,heading_level,heading_text,parent_id,document_id\n")
            f.write("H1,1,Main Topic,,APS112\n")
            f.write("H2,1,Main Topic,,APS113\n")
        logger.info(f"Created dummy hierarchy CSV: {hierarchy_csv_path}")


    try:
        logger.info(f"Initializing KnowledgeService with config: {config_file}")
        ks = KnowledgeService(config_file_path=str(config_file))

        # Override raw_dir for discover_documents if it were used, and for path relativization in process_documents
        ks.raw_dir = base_dir / "db/raw"
        ks.hierarchy_dir = base_dir / "db/raw/hierarchy"


        logger.info(f"Processing documents: {files_to_process}")
        # Setting force_rebuild=True to ensure DBs are built even if they exist from a previous run
        individual_vector_paths, individual_graph_paths, individual_memory_paths = ks.process_documents(
            files=files_to_process,
            output_vector_dir=str(vector_dir),
            output_graph_dir=str(graph_dir),
            output_memory_dir=str(memory_dir),
            force_rebuild=True 
        )
        logger.info(f"Successfully processed documents.")
        logger.info(f"  Individual Vector DBs created: {individual_vector_paths}")
        logger.info(f"  Individual Graph DBs created: {individual_graph_paths}")
        logger.info(f"  Individual Episodic Memory DBs created: {individual_memory_paths}")

        merged_vector_path = None
        merged_graph_path = None
        merged_episodic_memory_path = None
        
        if individual_vector_paths:
            logger.info("Merging vector and graph databases...")
            # Using 'au_standard_test' as a unique name for this test run's merged output
            merged_vector_path, merged_graph_path = ks.merge_databases(
                vector_db_paths=individual_vector_paths,
                graph_db_paths=individual_graph_paths,
                output_name="au_standard_test"
            )
            if merged_vector_path:
                logger.info(f"  Merged Vector DB created: {merged_vector_path}")
            else:
                logger.warning("  Merged Vector DB was not created.")
            if merged_graph_path:
                logger.info(f"  Merged Graph DB created: {merged_graph_path}")
            else:
                logger.warning("  Merged Graph DB was not created.")
        else:
            logger.warning("No individual vector databases were created, skipping merge step.")

        if individual_memory_paths:
            logger.info("Merging episodic memory databases...")
            merged_episodic_memory_path = ks._merge_memory_dbs(
                memory_db_paths=individual_memory_paths,
                output_dir=memory_dir,
                output_filename="episodic_memory_au_standard_test.parquet"
            )
            if merged_episodic_memory_path:
                logger.info(f"  Merged Episodic Memory DB created: {merged_episodic_memory_path}")
            else:
                logger.warning("  Merged Episodic Memory DB was not created.")
        else:
            logger.warning("No individual episodic memory databases to merge.")

        logger.info("Creating personality memory database...")
        personality_memory_path = ks._create_personality_memory_db(output_dir=memory_dir)
        if personality_memory_path:
            logger.info(f"  Personality Memory DB created: {personality_memory_path}")
        else:
            logger.warning("  Personality Memory DB was not created.")

        # Convert Parquet DBs to JSON
        logger.info("Converting Parquet databases to JSON for retriever...")
        final_json_paths = {
            "vector": None,
            "episodic_memory": None,
            "personality_memory": None
        }

        if merged_vector_path and Path(merged_vector_path).exists():
            json_vector_path = vector_dir / "vector_db_au_standard_test_for_retriever.json"
            if ks._convert_parquet_to_json(merged_vector_path, json_vector_path):
                logger.info(f"  Vector DB JSON created: {json_vector_path}")
                final_json_paths["vector"] = str(json_vector_path)
            else:
                logger.warning(f"  Failed to convert merged vector DB to JSON: {merged_vector_path}")
        elif individual_vector_paths: # Fallback to last individual if no merge
            last_vector_db = individual_vector_paths[-1]
            if Path(last_vector_db).exists():
                json_vector_path = Path(last_vector_db).parent / f"{Path(last_vector_db).stem}_for_retriever.json"
                if ks._convert_parquet_to_json(last_vector_db, json_vector_path):
                    logger.info(f"  Fell back to individual Vector DB JSON: {json_vector_path}")
                    final_json_paths["vector"] = str(json_vector_path)
                else:
                    logger.warning(f"  Failed to convert individual vector DB to JSON: {last_vector_db}")


        if merged_episodic_memory_path and Path(merged_episodic_memory_path).exists():
            json_episodic_path = memory_dir / "episodic_memory_au_standard_test_for_retriever.json"
            if ks._convert_parquet_to_json(merged_episodic_memory_path, json_episodic_path):
                logger.info(f"  Episodic Memory DB JSON created: {json_episodic_path}")
                final_json_paths["episodic_memory"] = str(json_episodic_path)
            else:
                logger.warning(f"  Failed to convert merged episodic memory DB to JSON: {merged_episodic_memory_path}")
        elif individual_memory_paths: # Fallback
            last_memory_db = individual_memory_paths[-1]
            if Path(last_memory_db).exists():
                json_episodic_path = Path(last_memory_db).parent / f"{Path(last_memory_db).stem}_for_retriever.json"
                if ks._convert_parquet_to_json(last_memory_db, json_episodic_path):
                    logger.info(f"  Fell back to individual Episodic Memory DB JSON: {json_episodic_path}")
                    final_json_paths["episodic_memory"] = str(json_episodic_path)
                else:
                    logger.warning(f"  Failed to convert individual episodic memory DB to JSON: {last_memory_db}")


        if personality_memory_path and Path(personality_memory_path).exists():
            json_personality_path = memory_dir / "personality_memory_au_standard_test_for_retriever.json"
            if ks._convert_parquet_to_json(personality_memory_path, json_personality_path):
                logger.info(f"  Personality Memory DB JSON created: {json_personality_path}")
                final_json_paths["personality_memory"] = str(json_personality_path)
            else:
                logger.warning(f"  Failed to convert personality memory DB to JSON: {personality_memory_path}")

        logger.info("\n--- KnowledgeService Test Summary ---")
        logger.info(f"Input documents: {files_to_process}")
        logger.info(f"Merged Vector DB (Parquet): {merged_vector_path if merged_vector_path else 'Not created/merged'}")
        logger.info(f"Merged Graph DB (Pickle): {merged_graph_path if merged_graph_path else 'Not created/merged'}")
        logger.info(f"Merged Episodic Memory DB (Parquet): {merged_episodic_memory_path if merged_episodic_memory_path else 'Not created/merged'}")
        logger.info(f"Personality Memory DB (Parquet): {personality_memory_path if personality_memory_path else 'Not created'}")
        
        logger.info("\nRetriever-ready JSON Databases:")
        logger.info(f"  Vector DB (JSON): {final_json_paths['vector'] if final_json_paths['vector'] else 'Not created'}")
        logger.info(f"  Episodic Memory (JSON): {final_json_paths['episodic_memory'] if final_json_paths['episodic_memory'] else 'Not created'}")
        logger.info(f"  Personality Memory (JSON): {final_json_paths['personality_memory'] if final_json_paths['personality_memory'] else 'Not created'}")

        logger.info(f"\nStats from KnowledgeService:")
        for key, value in ks.stats.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"An error occurred during the KnowledgeService test: {e}", exc_info=True)
        # If ks object exists, print its stats
        if 'ks' in locals() and hasattr(ks, 'stats'):
            logger.info(f"\nStats from KnowledgeService (even on error):")
            for key, value in ks.stats.items():
                logger.info(f"  {key}: {value}")

    finally:
        end_time = time.time()
        logger.info(f"KnowledgeService test script finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_test()
