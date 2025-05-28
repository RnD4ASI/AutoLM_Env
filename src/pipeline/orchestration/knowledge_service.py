import os
import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd

# Import all necessary modules
from src.pipeline.shared.logging import get_logger
from src.pipeline.shared.utility import DataUtility
from src.pipeline.processing.generator import Generator
from src.pipeline.processing.dbbuilder import TextParser, TextChunker, VectorBuilder, GraphBuilder, MemoryBuilder

logger = get_logger(__name__)


class KnowledgeService:
    """
    Orchestrates the process of building knowledge bases from raw documents.
    
    This service handles the end-to-end process of:
    1. Discovering documents in the raw directory
    2. Processing documents (converting PDFs to markdown)
    3. Chunking text into manageable segments
    4. Creating vector embeddings for each chunk
    5. Building a graph database with relationships between chunks
    6. Saving both databases in a consistent format
    
    The service is designed to be run as a standalone script and can process
    either the entire raw directory or specific subdirectories.
    """
    
    def __init__(self, config_file_path: Optional[Union[str, Path]] = None):
        """
        Initialize the KnowledgeService with necessary components.
        
        Args:
            config_file_path: Path to configuration file. If None, uses default path.
        """
        logger.info("Initializing KnowledgeService")
        start_time = time.time()
        
        try:
            # Load configuration
            self.config_path = Path(config_file_path) if config_file_path else Path.cwd() / "config" / "main_config.json"
            self.config = self._load_config()
            
            # Initialize utilities
            self.data_utility = DataUtility()
            
            # Initialize processing components
            self.text_parser = TextParser()
            self.text_chunker = TextChunker(config_file_path=self.config_path)
            self.generator = Generator()
            
            # Initialize database builders
            self.vector_builder = VectorBuilder(
                parser=self.text_parser,
                chunker=self.text_chunker,
                generator=self.generator,
                config_file_path=self.config_path
            )
            
            # Graph builder will be initialized after vector DB creation
            self.graph_builder = None
            
            # Set default paths
            self.db_dir = Path.cwd() / "db"
            self.raw_dir = self.db_dir / "raw"
            self.vector_dir = self.db_dir / "vector"
            self.graph_dir = self.db_dir / "graph"
            self.memory_dir = self.db_dir / "memory"
            self.hierarchy_dir = self.db_dir / "raw/hierarchy"
            
            # Ensure directories exist
            self.vector_dir.mkdir(exist_ok=True, parents=True)
            self.graph_dir.mkdir(exist_ok=True, parents=True)
            self.memory_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize MemoryBuilder
            self.memory_builder = MemoryBuilder()

            # Sample personality traits for basic population
            self.sample_personality_traits = [
                {
                    "mode_name": "Analytical Researcher",
                    "personality_type": "conscientiousness",
                    "cognitive_style": "analytical",
                    "mbti_type": "INTJ",
                    "mode_description": "Focuses on deep analysis, data-driven insights, and methodical research.",
                    "activation_contexts": ["research_task", "data_analysis"],
                    "activation_triggers": {"type": "keyword", "value": ["analyze", "research", "data"]}
                },
                {
                    "mode_name": "Creative Brainstormer",
                    "personality_type": "openness",
                    "cognitive_style": "creative",
                    "mbti_type": "ENFP",
                    "mode_description": "Generates novel ideas, explores diverse perspectives, and facilitates brainstorming sessions.",
                    "activation_contexts": ["ideation_task", "creative_problem_solving"],
                    "activation_triggers": {"type": "keyword", "value": ["create", "brainstorm", "innovate"]}
                },
                {
                    "mode_name": "Default Assistant",
                    "personality_type": "agreeableness",
                    "cognitive_style": "collaborative",
                    "mbti_type": "ISFJ",
                    "mode_description": "Provides general assistance, answers queries factually, and maintains a helpful demeanor.",
                    "activation_contexts": ["general_query", "default_interaction"],
                    "activation_triggers": None
                }
            ]
            
            # Track processing statistics
            self.stats = {
                "files_processed": 0,
                "pdfs_converted": 0,
                "chunks_created": 0,
                "vector_db_size": 0,
                "graph_nodes": 0,
                "graph_edges": 0,
                "episodic_memory_entries_created": 0,
                "personality_memory_entries_created": 0,
                "processing_time": 0
            }
            
            logger.info(f"KnowledgeService initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"KnowledgeService initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with fallback to defaults."""
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
                },
                "supported_file_types": ["pdf", "md", "txt"]
            },
            "knowledge_base": {
                "chunking": {
                    "method": "hierarchy",
                    "other_config": {
                        "chunk_size": 1000,
                        "chunk_overlap": 100
                    }
                },
                "vector_store": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "similarity_threshold": 0.7,
                    "top_k": 5
                }
            }
        }
    
    def discover_documents(self, raw_dir: Union[str, Path]) -> List[Path]:
        """
        Discover all PDF and markdown files in the specified directory and its subdirectories.
        
        Args:
            raw_dir: Directory to scan for documents
            
        Returns:
            List of Path objects for discovered documents
        """
        logger.info(f"Discovering documents in {raw_dir}")
        raw_path = Path(raw_dir)
        
        if not raw_path.exists():
            logger.warning(f"Raw directory {raw_path} does not exist")
            return []
        
        # Find all PDF and markdown files
        pdf_files = list(raw_path.glob("**/*.pdf"))
        md_files = list(raw_path.glob("**/*.md"))
        txt_files = list(raw_path.glob("**/*.txt"))
        
        all_files = pdf_files + md_files + txt_files
        logger.info(f"Discovered {len(all_files)} documents: {len(pdf_files)} PDFs, {len(md_files)} Markdown, {len(txt_files)} Text")
        
        return all_files
    
    def create_memory_db(self, 
                        vector_db_paths: List[str],
                        output_dir: Union[str, Path],
                        num_entries: int = 5) -> str:
        """
        Create a memory database from vector database entries.
        
        Args:
            vector_db_paths: List of paths to vector database files
            output_dir: Directory to save the memory database
            num_entries: Number of entries to extract from each vector database
            
        Returns:
            Path to the created memory database file, or None if an error occurs
        """
        logger.info(f"Creating memory database from {len(vector_db_paths)} vector databases")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        memory_db_path = output_dir / "memory.parquet"
        
        # Skip if memory DB already exists
        if memory_db_path.exists():
            logger.info(f"Memory database already exists at {memory_db_path}")
            return str(memory_db_path)
            
        all_memory_entries = []
        total_entries_created = 0
        
        try:
            # Process each vector database
            for vector_db_path in vector_db_paths:
                logger.info(f"Processing vector database: {vector_db_path}")
                
                # Create memory entries from the current vector database
                memory_path = self.memory_builder.create_db(
                    vector_db_file=vector_db_path,
                    num_entries=num_entries
                )
                
                # Load the created memory entries
                memory_df = self.memory_builder.load_db(memory_path)
                if not memory_df.empty:
                    all_memory_entries.append(memory_df)
                    entries_added = len(memory_df)
                    total_entries_created += entries_added
                    logger.info(f"Added {entries_added} memory entries from {vector_db_path}")
            
            # Combine all memory entries into a single DataFrame
            if all_memory_entries:
                combined_memory = pd.concat(all_memory_entries, ignore_index=True)
                
                # Save the combined memory database
                combined_memory.to_parquet(memory_db_path)
                logger.info(f"Created combined memory database with {total_entries_created} entries at {memory_db_path}")
                
                # Update statistics
                self.stats["episodic_memory_entries_created"] = total_entries_created
                
                return str(memory_db_path)
            else:
                logger.warning("No memory entries were created from the provided vector databases")
                return None
                
        except Exception as e:
            logger.error(f"Error creating memory database: {str(e)}")
            logger.debug(f"Memory DB creation error details: {traceback.format_exc()}")
            return None

    def process_documents(self, 
                     files: List[Path], 
                     output_vector_dir: Union[str, Path],
                     output_graph_dir: Union[str, Path],
                     output_memory_dir: Union[str, Path],
                     memory_entries_per_vector: int = 5,
                     chunking_method: str = 'hierarchy',
                     embedding_model: Optional[str] = None,
                     pdf_conversion_method: str = 'pymupdf',
                     force_rebuild: bool = False) -> Tuple[List[str], List[str], List[str]]:
        """
        Process a list of documents to create vector and graph databases.
        
        Args:
            files: List of files to process
            output_vector_dir: Directory to save vector databases
            output_graph_dir: Directory to save graph databases
            output_memory_dir: Directory to save memory databases
            memory_entries_per_vector: Number of memory entries to create per vector database
            embedding_model: Model to use for generating embeddings
            pdf_conversion_method: Method to use for PDF conversion
            chunking_method: Method to use for text chunking
            force_rebuild: Whether to force rebuilding existing databases
            
        Returns:
            Tuple of (vector_db_paths, graph_db_paths, memory_db_paths)
        """
        logger.info(f"Processing {len(files)} documents")
        start_time = time.time()
        
        # Ensure output directories exist
        vector_output_dir = Path(output_vector_dir)
        graph_output_dir = Path(output_graph_dir)
        memory_output_dir = Path(output_memory_dir)
        vector_output_dir.mkdir(exist_ok=True, parents=True)
        graph_output_dir.mkdir(exist_ok=True, parents=True)
        memory_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get embedding model from config if not specified
        if embedding_model is None:
            embedding_model = self.config.get('knowledge_base', {}).get('vector_store', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        
        vector_db_paths = []
        graph_db_paths = []
        memory_db_paths = []
        
        # Process each file
        for file_path in files:
            try:
                file_start_time = time.time()
                logger.info(f"Processing file: {file_path}")
                
                # Generate output paths
                relative_path = file_path.relative_to(self.raw_dir) if file_path.is_relative_to(self.raw_dir) else file_path.name
                base_name = relative_path.stem
                
                vector_db_path = vector_output_dir / f"v_{base_name}.parquet"
                graph_db_path = graph_output_dir / f"g_{base_name}.pkl"
                
                # Skip if databases already exist and force_rebuild is False
                if not force_rebuild and vector_db_path.exists() and graph_db_path.exists():
                    logger.info(f"Skipping {file_path} as databases already exist")
                    vector_db_paths.append(str(vector_db_path))
                    graph_db_paths.append(str(graph_db_path))
                    continue
                
                # Step 1: Process the input file (convert PDF to markdown if needed)
                markdown_path = self.vector_builder.process_input_file(
                    file_path=str(file_path),
                    conversion_method=pdf_conversion_method
                )
                
                if file_path.suffix.lower() == '.pdf':
                    self.stats["pdfs_converted"] += 1
                
                # Step 2: Apply chunking to the processed file
                # For hierarchy-based chunking, we need headings data
                # In a real implementation, this would be extracted from the document
                # For simplicity, we'll use length-based chunking if no headings data is available
                
                # Create empty headings DataFrame for now
                # In a real implementation, this would be populated with actual headings data
                logger.info(f"Loading headings data from CSV - {self.hierarchy_dir}")
                csv_path = Path(self.hierarchy_dir) / "APS_Hierarchy.csv"
                logger.info(f"Loading headings data from CSV - {csv_path}")
                df_headings = pd.read_csv(csv_path)
                logger.info(f"Headings data loaded from CSV - {csv_path}")

                chunks_df = self.vector_builder.apply_chunking(
                    input_file=markdown_path,
                    df_headings=df_headings if chunking_method == 'hierarchy' else None,
                    chunking_method=chunking_method
                )
                logger.info(f"Chunks created - {len(chunks_df)}")

                self.stats["chunks_created"] += len(chunks_df)
                
                # Step 3: Generate embeddings for the chunks
                chunks_df = self.vector_builder.create_embeddings(
                    chunks_df=chunks_df,
                    model=embedding_model
                )
                
                # Step 4: Save the vector database
                vector_db_path_str = self.vector_builder.save_db(
                    chunks_df=chunks_df,
                    input_file=str(file_path)
                )
                
                vector_db_path = Path(vector_db_path_str)
                vector_db_paths.append(vector_db_path_str)
                self.stats["vector_db_size"] += vector_db_path.stat().st_size
                
                # Step 5: Create graph database from vector database
                self.graph_builder = GraphBuilder(vectordb_file=vector_db_path_str)
                graph = self.graph_builder.create_db(graph_type='standard')
                
                # Step 6: Save the graph database
                graph_db_path_str = self.graph_builder.save_db(db_type='standard')
                graph_db_paths.append(graph_db_path_str)
                
                self.stats["graph_nodes"] += graph.number_of_nodes()
                self.stats["graph_edges"] += graph.number_of_edges()
                
                # Step 7: Create memory database
                try:
                    # Create a filename for this file's memory database
                    memory_db_filename = f"m_{base_name}.parquet"
                    memory_db_path = memory_output_dir / memory_db_filename
                    
                    # Only create if it doesn't exist or force_rebuild is True
                    if force_rebuild or not memory_db_path.exists():
                        memory_db_path_str = self.memory_builder.create_db(
                            vector_db_file=vector_db_path_str,
                            num_entries=memory_entries_per_vector
                        )
                        memory_db_paths.append(memory_db_path_str)
                        # Assuming memory_entries_per_vector is the number of entries in this specific parquet file
                        try:
                            temp_df = pd.read_parquet(memory_db_path_str)
                            self.stats["episodic_memory_entries_created"] += len(temp_df)
                        except Exception:
                            self.stats["episodic_memory_entries_created"] += memory_entries_per_vector # Fallback if read fails
                        logger.info(f"Created episodic memory file for {file_path} at {memory_db_path_str}")
                    else:
                        # If it exists and we're not rebuilding, just add it to the list
                        memory_db_paths.append(str(memory_db_path))
                        try:
                            temp_df = pd.read_parquet(memory_db_path)
                            # Potentially add to stats if needed, though usually, existing means already counted
                        except Exception:
                            pass # Already logged
                        logger.info(f"Using existing episodic memory file for {file_path} at {memory_db_path}")
                except Exception as e:
                    logger.error(f"Failed to create episodic memory file for {file_path}: {str(e)}")
                    logger.debug(traceback.format_exc())
                
                self.stats["files_processed"] += 1
                logger.info(f"Processed {file_path} in {time.time() - file_start_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                logger.debug(f"Processing error details: {traceback.format_exc()}")
        
        self.stats["processing_time"] = time.time() - start_time
        logger.info(f"Processed {self.stats['files_processed']} documents in {self.stats['processing_time']:.2f} seconds")
        logger.info(f"Created {len(vector_db_paths)} vector databases and {len(graph_db_paths)} graph databases")
        
        return vector_db_paths, graph_db_paths, memory_db_paths
    
    def merge_databases(self, 
                       vector_db_paths: List[str], 
                       graph_db_paths: List[str],
                       output_name: str = "merged") -> Tuple[str, str]:
        """
        Merge multiple vector and graph databases into consolidated databases.
        
        Args:
            vector_db_paths: List of vector database paths to merge
            graph_db_paths: List of graph database paths to merge
            output_name: Base name for the merged databases
            
        Returns:
            Tuple of (merged_vector_db_path, merged_graph_db_path)
        """
        logger.info(f"Merging {len(vector_db_paths)} vector databases and {len(graph_db_paths)} graph databases")
        
        merged_vector_path_str: Optional[str] = None
        merged_graph_path_str: Optional[str] = None

        if not vector_db_paths:
            logger.warning("No vector databases to merge, skipping vector and graph DB merge.")
        else:
            try:
                # Merge vector databases
                merged_vector_df = self.vector_builder.merge_db(
                    parquet_files=vector_db_paths,
                    output_name=f"v_{output_name}" 
                )
                
                merged_vector_path_str = str(self.vector_dir / f"v_{output_name}.parquet")
                logger.info(f"Merged vector database saved to {merged_vector_path_str}")
                
                # Initialize graph builder with the merged vector database
                self.graph_builder = GraphBuilder(vectordb_file=merged_vector_path_str)
                
                # Merge graph databases
                if graph_db_paths:
                    merged_graph = self.graph_builder.merge_dbs(
                        graph_files=graph_db_paths,
                        output_name=f"g_{output_name}",
                        db_type='standard'
                    )
                    
                    merged_graph_path_str = str(self.graph_dir / f"g_{output_name}.pkl")
                    logger.info(f"Merged graph database saved to {merged_graph_path_str}")
                else:
                    logger.warning("No graph databases to merge")
            
            except Exception as e:
                logger.error(f"Error merging vector/graph databases: {str(e)}")
                logger.debug(f"Merge error details: {traceback.format_exc()}")

        return merged_vector_path_str, merged_graph_path_str

    def _merge_memory_dbs(self, 
                        memory_db_paths: List[str], 
                        output_dir: Path,
                        output_filename: str = "episodic_memory.parquet") -> Optional[str]:
        """
        Merge multiple episodic memory Parquet files into a single file.
        
        Args:
            memory_db_paths: List of episodic memory Parquet file paths to merge.
            output_dir: Directory to save the merged memory database.
            output_filename: Filename for the merged memory database.
            
        Returns:
            Optional[str]: Path to the merged memory database file, or None if an error occurs.
        """
        if not memory_db_paths:
            logger.warning("No episodic memory database paths provided for merging.")
            return None

        logger.info(f"Merging {len(memory_db_paths)} episodic memory databases into {output_dir / output_filename}")
        
        all_memory_dfs = []
        for file_path_str in memory_db_paths:
            file_path = Path(file_path_str)
            if file_path.exists() and file_path.is_file():
                try:
                    df = pd.read_parquet(file_path)
                    all_memory_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading memory db file {file_path}: {e}")
            else:
                logger.warning(f"Memory db file not found or is not a file: {file_path}")

        if not all_memory_dfs:
            logger.warning("No valid episodic memory dataframes to merge.")
            return None
            
        try:
            combined_df = pd.concat(all_memory_dfs, ignore_index=True)
            output_path = output_dir / output_filename
            combined_df.to_parquet(output_path)
            logger.info(f"Merged episodic memory database saved to {output_path} with {len(combined_df)} entries.")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error merging or saving combined episodic memory database: {str(e)}")
            logger.debug(f"Episodic memory merge error details: {traceback.format_exc()}")
            return None

    def _create_personality_memory_db(self, output_dir: Path) -> Optional[str]:
        """
        Creates and saves a personality memory database from sample traits.

        Args:
            output_dir: Directory to save the personality memory database.

        Returns:
            Optional[str]: Path to the saved personality memory database file, or None if an error occurs.
        """
        logger.info("Creating personality memory database.")
        if not self.sample_personality_traits:
            logger.warning("No sample personality traits defined. Skipping personality memory creation.")
            return None

        try:
            df = pd.DataFrame(self.sample_personality_traits)
            
            # Add mode_id using UUIDs
            df['mode_id'] = [self.data_utility.generate_uuid() for _ in range(len(df))]
            
            # Ensure all required columns from schema are present, even if some are optional in sample data
            # Required: mode_id, mode_name, personality_type, cognitive_style, mbti_type
            # Optional: mode_description, activation_contexts, activation_triggers
            schema_required_cols = ["mode_id", "mode_name", "personality_type", "cognitive_style", "mbti_type"]
            schema_optional_cols = ["mode_description", "activation_contexts", "activation_triggers"]

            for col in schema_required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column '{col}' for personality memory. Cannot create.")
                    # Or df[col] = None / default value if appropriate, but for required, it's an issue.
                    return None 
            
            for col in schema_optional_cols:
                if col not in df.columns:
                    df[col] = None # Add optional columns if missing, filling with None or appropriate default

            # Reorder columns to match schema expectations if desired, though Parquet doesn't strictly enforce order
            # df = df[schema_required_cols + schema_optional_cols]

            output_path = output_dir / "personality_memory.parquet"
            df.to_parquet(output_path)
            
            self.stats["personality_memory_entries_created"] = len(df)
            logger.info(f"Personality memory database saved to {output_path} with {len(df)} entries.")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error creating personality memory database: {str(e)}")
            logger.debug(f"Personality memory creation error details: {traceback.format_exc()}")
            return None

    def _convert_parquet_to_json(self, parquet_path: Union[str, Path], json_path: Path) -> bool:
        """
        Converts a Parquet file to a JSON file (list of records).

        Args:
            parquet_path: Path to the input Parquet file.
            json_path: Path to save the output JSON file.

        Returns:
            bool: True if conversion was successful, False otherwise.
        """
        parquet_file = Path(parquet_path)
        if not parquet_file.exists():
            logger.error(f"Parquet file not found for JSON conversion: {parquet_file}")
            return False
        
        try:
            df = pd.read_parquet(parquet_file)
            # Convert all columns to string to handle potential complex types like lists/embeddings for JSON
            # df = df.astype(str) # This might be too aggressive; to_json handles most types.
            # Let's ensure embeddings (typically lists/arrays) are converted to string representation if not already.
            for col in df.columns:
                if isinstance(df[col].iloc[0], (list, pd.Series, pd.DataFrame)): # Check if first element is list-like
                    df[col] = df[col].apply(str) # Convert list-like objects to string
            
            df.to_json(json_path, orient='records', indent=2, force_ascii=False)
            logger.info(f"Successfully converted {parquet_file} to {json_path}")
            return True
        except Exception as e:
            logger.error(f"Error converting Parquet file {parquet_file} to JSON {json_path}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run(self, 
           raw_dir: Optional[Union[str, Path]] = None,
           vector_dir: Optional[Union[str, Path]] = None,
           graph_dir: Optional[Union[str, Path]] = None,
           memory_dir: Optional[Union[str, Path]] = None,
           memory_entries_per_vector: int = 5,
           chunking_method: str = 'hierarchy',
           embedding_model: Optional[str] = None,
           pdf_conversion_method: str = 'pymupdf',
           force_rebuild: bool = False,
           merge_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete knowledge service pipeline.
        
        Args:
            raw_dir: Directory containing raw documents
            vector_dir: Directory to save vector databases
            graph_dir: Directory to save graph databases
            memory_dir: Directory to save memory databases
            create_memory_db: Whether to create memory databases
            memory_entries_per_vector: Number of memory entries to create per vector database
            chunking_method: Method to use for text chunking
            embedding_model: Model to use for generating embeddings
            pdf_conversion_method: Method to use for PDF conversion
            force_rebuild: Whether to force rebuilding existing databases
            merge_results: Whether to merge individual databases
            
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info("Starting knowledge service pipeline")
        overall_start_time = time.time()
        
        # Set directories
        raw_path = Path(raw_dir) if raw_dir else self.raw_dir
        vector_path = Path(vector_dir) if vector_dir else self.vector_dir
        graph_path = Path(graph_dir) if graph_dir else self.graph_dir
        memory_path = Path(memory_dir) if memory_dir else self.memory_dir
        
        # Step 1: Discover documents
        files = self.discover_documents(raw_path)
        
        if not files:
            logger.warning(f"No documents found in {raw_path}")
            return {
                "status": "completed",
                "message": f"No documents found in {raw_path}",
                "vector_db_paths": [],
                "graph_db_paths": [],
                "memory_db_paths": [],
                "stats": self.stats
            }
        
        # Step 2: Process documents
        vector_db_paths, graph_db_paths, memory_db_paths = self.process_documents(
            files=files,
            output_vector_dir=vector_path,
            output_graph_dir=graph_path,
            output_memory_dir=memory_path,
            memory_entries_per_vector=memory_entries_per_vector,
            chunking_method=chunking_method,
            embedding_model=embedding_model,
            pdf_conversion_method=pdf_conversion_method,
            force_rebuild=force_rebuild
        )
        
        # Step 3: Merge databases if requested
        merged_vector_path_str: Optional[str] = None
        merged_graph_path_str: Optional[str] = None
        merged_episodic_memory_path_str: Optional[str] = None
        personality_memory_db_path_str: Optional[str] = None # Already defined but ensure it's captured

        vector_db_json_path: Optional[str] = None
        episodic_memory_json_path: Optional[str] = None
        personality_memory_json_path: Optional[str] = None
        
        if merge_results:
            output_name = raw_path.name if raw_path.name != "raw" else "complete"
            
            if len(vector_db_paths) > 0 : 
                merged_vector_path_str, merged_graph_path_str = self.merge_databases(
                    vector_db_paths=vector_db_paths,
                    graph_db_paths=graph_db_paths, 
                    output_name=output_name
                )
                if merged_vector_path_str:
                    json_vec_path = vector_path / "vector_db_for_retriever.json"
                    if self._convert_parquet_to_json(merged_vector_path_str, json_vec_path):
                        vector_db_json_path = str(json_vec_path)
            else:
                logger.info("Skipping merging of vector and graph databases as no vector databases were created/found.")

            if memory_db_paths:
                merged_episodic_memory_path_str = self._merge_memory_dbs(
                    memory_db_paths=memory_db_paths,
                    output_dir=memory_path, 
                    output_filename="episodic_memory.parquet"
                )
                if merged_episodic_memory_path_str:
                    try:
                        merged_df = pd.read_parquet(merged_episodic_memory_path_str)
                        self.stats["episodic_memory_entries_created"] = len(merged_df)
                        json_episodic_path = memory_path / "episodic_memory_for_retriever.json"
                        if self._convert_parquet_to_json(merged_episodic_memory_path_str, json_episodic_path):
                            episodic_memory_json_path = str(json_episodic_path)
                    except Exception as e:
                        logger.error(f"Could not read or convert merged episodic memory: {e}")
            else:
                logger.info("No individual episodic memory files to merge.")
        else: # Handle case where merging is not done - convert individual files if any
            if vector_db_paths: # Potentially convert the first or last individual vector DB
                logger.info("Merge results set to False. Consider converting individual Parquet files if needed by retrievers.")
                # As an example, let's convert the last processed vector DB if no merge
                # This logic might need refinement based on actual use case for non-merged data
                if not merged_vector_path_str and vector_db_paths:
                    last_vector_db_parquet = vector_db_paths[-1]
                    json_vec_path = Path(last_vector_db_parquet).parent / f"{Path(last_vector_db_parquet).stem}_for_retriever.json"
                    if self._convert_parquet_to_json(last_vector_db_parquet, json_vec_path):
                        vector_db_json_path = str(json_vec_path) # This would be for the last individual file

            if memory_db_paths and not merged_episodic_memory_path_str:
                # Similar logic for episodic memory if not merged
                last_episodic_mem_parquet = memory_db_paths[-1]
                json_episodic_path = Path(last_episodic_mem_parquet).parent / f"{Path(last_episodic_mem_parquet).stem}_for_retriever.json"
                if self._convert_parquet_to_json(last_episodic_mem_parquet, json_episodic_path):
                    episodic_memory_json_path = str(json_episodic_path)


        # Step 4: Create Personality Memory DB (Parquet)
        personality_memory_db_path_str = self._create_personality_memory_db(output_dir=memory_path)
        if personality_memory_db_path_str:
            json_personality_path = memory_path / "personality_memory_for_retriever.json"
            if self._convert_parquet_to_json(personality_memory_db_path_str, json_personality_path):
                personality_memory_json_path = str(json_personality_path)
        
        # Update stats and prepare result
        total_time = time.time() - overall_start_time
        
        result = {
            "status": "completed",
            "message": f"Successfully processed {len(files)} documents",
            "individual_vector_db_paths": vector_db_paths,
            "individual_graph_db_paths": graph_db_paths,
            "individual_episodic_memory_db_paths": memory_db_paths,
            "merged_vector_db_parquet": merged_vector_path_str,
            "merged_graph_db_pickle": merged_graph_path_str, # Keep original key for clarity
            "merged_episodic_memory_db_parquet": merged_episodic_memory_path_str,
            "personality_memory_db_parquet": personality_memory_db_path_str,
            "vector_db_json_path": vector_db_json_path,
            "episodic_memory_json_path": episodic_memory_json_path,
            "personality_memory_json_path": personality_memory_json_path,
            "stats": {
                **self.stats,
                "total_time": total_time
            }
        }
        
        logger.info(f"Knowledge service pipeline completed in {total_time:.2f} seconds")
        return result


def main():
    """Main entry point for the knowledge service."""
    parser = argparse.ArgumentParser(description="AutoLM Knowledge Service")
    
    parser.add_argument("--raw_dir", type=str, default="db/raw",
                        help="Directory containing raw documents")
    parser.add_argument("--vector_dir", type=str, default="db/vector",
                        help="Directory to save vector databases")
    parser.add_argument("--graph_dir", type=str, default="db/graph",
                        help="Directory to save graph databases")
    parser.add_argument("--memory_dir", type=str, default="db/memory",
                        help="Directory to save memory databases")
    parser.add_argument("--config", type=str, default="config/main_config.json",
                        help="Path to configuration file")
    parser.add_argument("--chunking_method", type=str, default="hierarchy",
                        choices=["hierarchy", "length"],
                        help="Method to use for text chunking")
    parser.add_argument("--embedding_model", type=str,
                        help="Model to use for generating embeddings")
    parser.add_argument("--pdf_conversion", type=str, default="pymupdf",
                        choices=["pymupdf", "markitdown", "openleaf", "ocr", "llamaindex", "textract"],
                        help="Method to use for PDF conversion")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuilding existing databases")
    parser.add_argument("--no_merge", action="store_true",
                        help="Do not merge individual databases")
    
    args = parser.parse_args()
    
    try:
        # Initialize the knowledge service
        service = KnowledgeService(config_file_path=args.config)
        
        # Run the pipeline
        result = service.run(
            raw_dir=args.raw_dir,
            vector_dir=args.vector_dir,
            graph_dir=args.graph_dir,
            memory_dir=args.memory_dir,
            chunking_method=args.chunking_method,
            embedding_model=args.embedding_model,
            pdf_conversion_method=args.pdf_conversion,
            force_rebuild=args.force_rebuild,
            merge_results=not args.no_merge
        )
        
        # Print summary
        print("\nKnowledge Service Summary:")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        print(f"\nStatistics:")
        print(f"  Files processed: {result['stats']['files_processed']}")
        print(f"  PDFs converted: {result['stats']['pdfs_converted']}")
        print(f"  Chunks created: {result['stats']['chunks_created']}")
        print(f"  Vector DB size: {result['stats']['vector_db_size'] / (1024*1024):.2f} MB") # Note: This is sum of individual file sizes
        print(f"  Graph nodes: {result['stats']['graph_nodes']}") # Note: This is sum from individual graphs
        print(f"  Graph edges: {result['stats']['graph_edges']}") # Note: This is sum from individual graphs
        print(f"  Episodic Memory Entries: {result['stats']['episodic_memory_entries_created']}")
        print(f"  Personality Memory Entries: {result['stats']['personality_memory_entries_created']}")
        print(f"  Total Processing time: {result['stats']['total_time']:.2f} seconds")
        
        print(f"\nIndividual DB Paths:")
        print(f"  Vector DBs (Parquet): {len(result['individual_vector_db_paths'])} files")
        print(f"  Graph DBs (Pickle): {len(result['individual_graph_db_paths'])} files")
        print(f"  Episodic Memory DBs (Parquet): {len(result['individual_episodic_memory_db_paths'])} files")

        print(f"\nConsolidated Parquet/Pickle Databases:")
        if result['merged_vector_db_parquet']:
            print(f"  Merged Vector DB (Parquet): {result['merged_vector_db_parquet']}")
        if result['merged_graph_db_pickle']:
            print(f"  Merged Graph DB (Pickle): {result['merged_graph_db_pickle']}")
        if result['merged_episodic_memory_db_parquet']:
            print(f"  Merged Episodic Memory DB (Parquet): {result['merged_episodic_memory_db_parquet']}")
        if result['personality_memory_db_parquet']:
            print(f"  Personality Memory DB (Parquet): {result['personality_memory_db_parquet']}")

        print(f"\nRetriever-Ready JSON Databases:")
        if result['vector_db_json_path']:
            print(f"  Vector DB (JSON for Retriever): {result['vector_db_json_path']}")
        if result['episodic_memory_json_path']:
            print(f"  Episodic Memory DB (JSON for Retriever): {result['episodic_memory_json_path']}")
        if result['personality_memory_json_path']:
            print(f"  Personality Memory DB (JSON for Retriever): {result['personality_memory_json_path']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Knowledge service failed: {str(e)}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
