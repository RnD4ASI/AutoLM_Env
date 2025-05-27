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
            
            # Track processing statistics
            self.stats = {
                "files_processed": 0,
                "pdfs_converted": 0,
                "chunks_created": 0,
                "vector_db_size": 0,
                "graph_nodes": 0,
                "graph_edges": 0,
                "memory_entries_created": 0,
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
                self.stats["memory_entries_created"] = total_entries_created
                
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
                        self.stats["memory_entries_created"] += memory_entries_per_vector
                        logger.info(f"Created memory database for {file_path} at {memory_db_path_str}")
                    else:
                        # If it exists and we're not rebuilding, just add it to the list
                        memory_db_paths.append(str(memory_db_path))
                        logger.info(f"Using existing memory database for {file_path} at {memory_db_path}")
                except Exception as e:
                    logger.error(f"Failed to create memory database for {file_path}: {str(e)}")
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
        
        if not vector_db_paths:
            logger.warning("No vector databases to merge")
            return None, None
        
        try:
            # Merge vector databases
            merged_vector_df = self.vector_builder.merge_db(
                parquet_files=vector_db_paths,
                output_name=f"v_{output_name}"
            )
            
            merged_vector_path = str(self.vector_dir / f"v_{output_name}.parquet")
            logger.info(f"Merged vector database saved to {merged_vector_path}")
            
            # Initialize graph builder with the merged vector database
            self.graph_builder = GraphBuilder(vectordb_file=merged_vector_path)
            
            # Merge graph databases
            if graph_db_paths:
                merged_graph = self.graph_builder.merge_dbs(
                    graph_files=graph_db_paths,
                    output_name=f"g_{output_name}",
                    db_type='standard'
                )
                
                merged_graph_path = str(self.graph_dir / f"g_{output_name}.pkl")
                logger.info(f"Merged graph database saved to {merged_graph_path}")
            else:
                logger.warning("No graph databases to merge")
                merged_graph_path = None
            
            return merged_vector_path, merged_graph_path
            
        except Exception as e:
            logger.error(f"Error merging databases: {str(e)}")
            logger.debug(f"Merge error details: {traceback.format_exc()}")
            return None, None
    
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
        merged_vector_path = None
        merged_graph_path = None
        merged_memory_path = None
        
        if merge_results and len(vector_db_paths) > 1:
            # Generate a name based on the raw directory
            output_name = raw_path.name if raw_path.name != "raw" else "complete"
            
            merged_vector_path, merged_graph_path = self.merge_databases(
                vector_db_paths=vector_db_paths,
                graph_db_paths=graph_db_paths,
                output_name=output_name
            )
        
        # Update stats and prepare result
        total_time = time.time() - overall_start_time
        
        result = {
            "status": "completed",
            "message": f"Successfully processed {len(files)} documents",
            "vector_db_paths": vector_db_paths,
            "graph_db_paths": graph_db_paths,
            "memory_db_paths": memory_db_paths if memory_db_paths else [],
            "merged_vector_db": merged_vector_path,
            "merged_graph_db": merged_graph_path,
            "merged_memory_db": merged_memory_path,
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
        print(f"  Vector DB size: {result['stats']['vector_db_size'] / (1024*1024):.2f} MB")
        print(f"  Graph nodes: {result['stats']['graph_nodes']}")
        print(f"  Graph edges: {result['stats']['graph_edges']}")
        print(f"  Processing time: {result['stats']['total_time']:.2f} seconds")
        
        if result['merged_vector_db']:
            print(f"\nMerged databases:")
            print(f"  Vector DB: {result['merged_vector_db']}")
            print(f"  Graph DB: {result['merged_graph_db']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Knowledge service failed: {str(e)}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
