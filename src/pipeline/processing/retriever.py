import pandas as pd
import pandas as pd
import pandas as pd
import pandas as pd
import json
import os
import datetime # Already present, but good to confirm
# Ensure datetime is available if needed by other parts of the file, though not directly by parquet loading.
import datetime
import datetime # Already present, but good to confirm
import datetime
import re
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

# Other existing imports
from community import best_partition
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# Local imports
from src.pipeline.processing.generator import Generator, MetaGenerator
from src.pipeline.shared.utility import DataUtility, AIUtility
from src.pipeline.processing.evaluator import Evaluator, RetrievalEvaluator
from src.pipeline.shared.logging import get_logger

logger = get_logger(__name__)

class QueryProcessor:
    """
    Handles query preprocessing methods like rewriting and decomposition.
    Responsible for transforming raw queries into more effective forms for retrieval.
    """
    
    def __init__(self, generator: Optional[Generator] = None):
        """
        Initialize the QueryProcessor.
        
        Args:
            generator: Generator instance for text generation and completions
        """
        logger.debug("QueryProcessor initialization started")
        try:
            start_time = time.time()
            self.generator = generator if generator else Generator()
            self.metagenerator = MetaGenerator(generator=self.generator)
            logger.debug(f"QueryProcessor initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"QueryProcessor initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def rephrase_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5) -> str:
        """
        Method 1: Rephrase the query to be more effective for retrieval.
        Uses metaprompt if available, otherwise falls back to template.
        
        Args:
            query: Original query string
            model: Model to use for generation
            temperature: Temperature for generation
            
        Returns:
            str: Rewritten query optimized for retrieval
        """
        logger.debug(f"Rephrasing query with model={model}, temperature={temperature}")
        logger.debug(f"Original query: '{query[:100]}{'...' if len(query) > 100 else ''}'")  # Log first 100 chars
        start_time = time.time()
        try:
            rephrased_query = self.metagenerator.get_meta_generation(
                application="metaworkflow",
                category="retriever",
                action="query_rephrase",  # Using rephrase as the action for query rewriting
                prompt_id=1,  # Using a default prompt ID
                task_prompt=query,
                model=model,
                temperature=temperature,
                return_full_response=False
            )
            logger.debug(f"Query successfully rephrased in {time.time() - start_time:.2f} seconds")
            logger.debug(f"Rephrased query: '{rephrased_query[:100]}{'...' if len(rephrased_query) > 100 else ''}'")  # Log first 100 chars
        except Exception as e:
            logger.error(f"Metaprompt query rephrasing failed, falling back to template: {e}")
            logger.debug(f"Rephrasing error details: {traceback.format_exc()}")
            rephrased_query = query  # Fall back to original query
            logger.debug("Using original query as fallback")
        return rephrased_query

    def decompose_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.7) -> List[Dict[str, Any]]:
        """
        Method 2: Decompose complex query into simpler sub-queries.
        Uses metaprompt if available, otherwise falls back to template.
        
        Args:
            query: Complex query string to decompose
            model: Model to use for generation
            temperature: Temperature for generation
            
        Returns:
            List[Dict[str, Any]]: List of sub-queries with their weights
        """
        try:
            decomposed_query = self.metagenerator.get_meta_generation(
                    application="metaworkflow",
                    category="retriever",
                    action="query_decompose",
                    prompt_id=2,  # Using a default prompt ID
                    task_prompt=query,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
        except Exception as e:
            logger.error(f"Query decomposition failed completely: {e}")
        return decomposed_query

    def hypothesize_query(self, query: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5) -> str:
        """
        Method 3: Apply HyDE to generate a hypothetical documentation as initial anchor for answering the task prompt.
        
        Args:
            query: Task prompt string to generate hypothetical query for
            model: Model to use for generation
            temperature: Temperature for generation
            
        Returns:
            str: Hypothetical query generated for the task prompt
        """
        try:
            hypothesized_query = self.metagenerator.get_meta_generation(
                    application="metaworkflow",
                    category="retriever",
                    action="query_hypothesize",
                    prompt_id=3,
                    task_prompt=query,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
        except Exception as e:
            logger.error(f"Hypothetical query generation failed completely: {e}")
        return hypothesized_query

    def predict_query(self, query: str, response: str, model: str = "Qwen2.5-1.5B", temperature: float = 0.5) -> str:
        """
        Method 5: Generate a prediction of the next subquestion to solve, after considering the original task prompt, the prior subquestions, as well as the response to those prior subquestions.

        Args:
            query: Task prompt string of the original task
            response: Response to the subquestions of the original task and the corresponding responses
            model: Model to use for prediction
            temperature: Temperature for prediction
            
        Returns:
            str: Predicted query for the next subquestion
        """
        try:
            predicted_query = self.metagenerator.get_meta_generation(
                    application="metaworkflow",
                    category="retriever",
                    action="query_predict",
                    prompt_id=5,
                    task_prompt=query,
                    response=response,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
        except Exception as e:
            logger.error(f"Predicted query generation failed completely: {e}")
        return predicted_query

class RerankProcessor:
    """
    Handles reranking of retrieval results using various methods.
    Responsible for improving the ordering of retrieved documents.
    """
    
    def __init__(self, generator: Optional[Generator] = None):
        """
        Initialize.
        """
        self.generator = generator if generator else Generator()
        self.rrf_k = 60
        logger.debug("RerankProcessor initialized")
    
    # Method 1 - Reciprocal Rank Fusion
    def rerank_reciprocal_rank_fusion(self, results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to rerank results.
        Sam doesn't recommend this given it butched the unit of the measure.
        
        Args:
            results: List of retrieval results
            top_k: Number of top results to return (if None, returns all results)
            
        Returns:
            List[Dict[str, Any]]: Top k reranked results
        """
        logger.debug(f"Starting reciprocal rank fusion reranking with top_k: {top_k if top_k else 'all'}")
        start_time = time.time()
        
        if not results:
            logger.debug("Skipping reranking: No results to rerank")
            return results
        
        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Calculate RRF scores
        for i, result in enumerate(sorted_results):
            rrf_score = 1 / (self.rrf_k + i + 1)
            result['score'] = rrf_score
        
        # Return top_k results if specified
        if top_k is not None and top_k < len(sorted_results):
            sorted_results = sorted_results[:top_k]
            logger.debug(f"Returning top {top_k} results")
        
        logger.debug(f"Reciprocal rank fusion reranking completed in {time.time() - start_time:.2f} seconds")
        return sorted_results
    
    # Method 2 - Cross-Encoder
    def rerank_crossencoder(self, query: str, model: str, results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank results using a cross-encoder model via generator.get_reranking.
        
        Args:
            query: Original query string (required for cross-encoder reranking)
            results: List of retrieval results
            top_k: Number of top results to return (if None, returns all results)
            
        Returns:
            List[Dict[str, Any]]: Top k reranked results
        """
        logger.debug(f"Starting cross-encoder reranking for query: '{query[:50]}{'...' if len(query) > 50 else ''}' with {len(results)} results")
        start_time = time.time()
        
        if not results:
            logger.debug("Skipping reranking: No results to rerank")
            return results
            
        try:
            # Extract content from results for reranking
            passages = [result['content'] for result in results]
            
            # Use generator.get_reranking to rerank passages
            logger.debug(f"Using generator.get_reranking with model: {model}")
            reranked_pairs = self.generator.get_reranking(
                query=query,
                passages=passages,
                model=model,
                batch_size=32,
                return_scores=True
            )
            
            # Extract reranked passages and scores
            reranked_passages = [pair[0] for pair in reranked_pairs]
            reranked_scores = [pair[1] for pair in reranked_pairs]
            
            logger.debug(f"Received scores range: min={min(reranked_scores) if reranked_scores else 'N/A'}, max={max(reranked_scores) if reranked_scores else 'N/A'}")
            
            # Create a mapping from content to result for reassembly
            content_to_result = {result['content']: result for result in results}
            
            # Reassemble results with new scores and order
            reranked_results = []
            for passage, score in reranked_pairs:
                if passage in content_to_result:
                    result = dict(content_to_result[passage])  # Create a copy
                    result['cross_encoder_score'] = float(score)
                    reranked_results.append(result)
            
            # If any results weren't matched, add them at the end
            original_contents = set(content_to_result.keys())
            reranked_contents = set(reranked_passages)
            missing_contents = original_contents - reranked_contents
            
            for content in missing_contents:
                reranked_results.append(content_to_result[content])
            
            # Log the change in ranking
            if len(results) > 1 and len(reranked_results) > 0:
                original_top_id = results[0].get('id', 'unknown')
                reranked_top_id = reranked_results[0].get('id', 'unknown')
                if original_top_id != reranked_top_id:
                    logger.debug(f"Reranking changed top result from id={original_top_id} to id={reranked_top_id}")
                else:
                    logger.debug("Top result remained the same after reranking")
            
            # Return top_k results if specified
            if top_k is not None and top_k < len(reranked_results):
                reranked_results = reranked_results[:top_k]
                logger.debug(f"Returning top {top_k} results")
            
            logger.debug(f"Cross-encoder reranking completed in {time.time() - start_time:.2f} seconds")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            logger.debug(f"Reranking error details: {traceback.format_exc()}")
            return results
    
    # Method 3 - Bi-Encoder
    def rerank_biencoder(self, query: str, model: str, results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank results using a bi-encoder model (embedding similarity).
        This allows using a different embedding model than what was used during the initial retrieval.
        
        Args:
            query: Original query string
            results: List of retrieval results
            top_k: Number of top results to return (if None, returns all results)
            model: Optional specific bi-encoder model to use (overrides the one set during initialization)
            
        Returns:
            List[Dict[str, Any]]: Top k reranked results
        """
        logger.debug(f"Starting bi-encoder reranking for query: '{query[:50]}{'...' if len(query) > 50 else ''}' with {len(results)} results")
        start_time = time.time()
        
        if not results:
            logger.debug("Skipping reranking: No results to rerank")
            return results
            
        try:
            # Use specified model or default
            logger.debug(f"Using bi-encoder model: {model}")
            
            # Extract content from results for reranking
            passages = [result['content'] for result in results]
            
            # Get embeddings for query and passages
            embeddings = self.generator.get_embedding(
                texts=[query] + passages,
                model=model,
                batch_size=32
            )
            
            if embeddings is None or len(embeddings) < len(passages) + 1:
                logger.error("Failed to get embeddings for query and passages")
                return results
                
            # Extract query and passage embeddings
            query_embedding = embeddings[0]
            passage_embeddings = embeddings[1:]
            
            # Calculate cosine similarities
            similarities = []
            for i, passage_embedding in enumerate(passage_embeddings):
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, passage_embedding) / \
                            (np.linalg.norm(query_embedding) * np.linalg.norm(passage_embedding))
                similarities.append((i, float(similarity)))
            
            # Sort by similarity score in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Create reranked results
            reranked_results = []
            for idx, score in similarities:
                result = dict(results[idx])  # Create a copy
                result['bi_encoder_score'] = score
                reranked_results.append(result)
            
            # Log the change in ranking
            if len(results) > 1 and len(reranked_results) > 0:
                original_top_id = results[0].get('id', 'unknown')
                reranked_top_id = reranked_results[0].get('id', 'unknown')
                if original_top_id != reranked_top_id:
                    logger.debug(f"Bi-encoder reranking changed top result from id={original_top_id} to id={reranked_top_id}")
                else:
                    logger.debug("Top result remained the same after bi-encoder reranking")
            
            # Return top_k results if specified
            if top_k is not None and top_k < len(reranked_results):
                reranked_results = reranked_results[:top_k]
                logger.debug(f"Returning top {top_k} results")
            
            logger.debug(f"Bi-encoder reranking completed in {time.time() - start_time:.2f} seconds")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Bi-encoder reranking failed: {e}")
            logger.debug(f"Reranking error details: {traceback.format_exc()}")
            return results


class VectorRetriever:
    """Enhanced retrieval engine for vector databases with parameterizable topology.
    
    This class works with vector databases and query inputs to enable both semantic 
    and symbolic search capabilities with configurable retrieval pipeline topologies.
    
    Features:
    - Query preprocessing (rephrasing, hypothesizing)
    - Hybrid search (semantic, symbolic)
    - Reranking methods (bi-encoder, cross-encoder, reciprocal rank fusion)
    - Configurable retrieval topology
    """
    
    def __init__(self, vector_db_path: Optional[Union[str, Path, pd.DataFrame]] = None, generator: Optional[Generator] = None, embedding_model: Optional[str] = None):
        """Initialize VectorRetriever with configurable components.
        
        Args:
            vector_db_path: Path to the vector database Parquet file or a pre-loaded DataFrame.
            generator: Generator instance for embeddings, completions, and query processing.
            embedding_model: Default embedding model to use if not specified in search config.
        """
        logger.debug(f"Initializing VectorRetriever with DB path: {vector_db_path}")
        # Core components
        self.vector_df = None
        self.vector_db_path = None
        self.generator = generator if generator else Generator()
        self.embedding_model = embedding_model # Store default embedding model

        if isinstance(vector_db_path, (str, Path)):
            self.vector_db_path = Path(vector_db_path)
            if self.vector_db_path.exists() and self.vector_db_path.suffix == '.parquet':
                logger.info(f"Loading VectorDB from Parquet file: {self.vector_db_path}")
                self.vector_df = pd.read_parquet(self.vector_db_path)
                logger.info(f"VectorDB loaded with {len(self.vector_df)} entries.")
            else:
                logger.error(f"VectorDB Parquet file not found or invalid: {self.vector_db_path}")
                # Initialize an empty DataFrame with expected columns to prevent errors downstream
                # These columns are based on what _get_corpus_from_vector_db expects
                self.vector_df = pd.DataFrame(columns=['chunk_id', 'corpus', 'corpus_vector', 
                                                       'document_id', 'document_name', 'reference', 
                                                       'hierarchy', 'source', 'level', 'embedding_model',
                                                       'heading', 'content_type']) # Added new optional fields
        elif isinstance(vector_db_path, pd.DataFrame):
            logger.info("Loading VectorDB from pre-loaded DataFrame.")
            self.vector_df = vector_db_path
        else:
            logger.warning("No VectorDB path or DataFrame provided. VectorRetriever will operate on an empty dataset.")
            self.vector_df = pd.DataFrame(columns=['chunk_id', 'corpus', 'corpus_vector',
                                                       'document_id', 'document_name', 'reference',
                                                       'hierarchy', 'source', 'level', 'embedding_model',
                                                       'heading', 'content_type'])


        # Initialize component processors
        self.query_processor = QueryProcessor(generator=self.generator)
        self.rerank_processor = RerankProcessor(generator=self.generator)
        
        # Load NLP model for text processing (used in symbolic search)
        try:
            if spacy.util.is_package("en_core_web_sm"):
                self.nlp = spacy.load("en_core_web_sm")
                logger.debug(f"Spacy tokeniser loaded successfully")
            else:
                logger.warning("Downloading spacy model 'en_core_web_sm'...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.debug(f"Spacy tokeniser loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load spacy model: {e}")
            self.nlp = None
        
        # Initialize symbolic search components
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2)
        )
        self.bm25 = None
        logger.debug("VectorRetriever initialized")
    
    def retrieve(self, query, config=None):
        """Main retrieval method with configurable topology.
        
        Args:
            query: Original query string
            config: Retrieval configuration with these optional parameters:
                - query_processing: Dict with settings for query preprocessing
                    - rephrase: bool - Whether to rephrase the query
                    - hypothesize: bool - Whether to use Hypothetical Document Embedding
                    - model: str - Model to use for query processing
                - search: Dict with settings for search methods
                    - semantic: bool - Whether to use semantic search
                    - symbolic: bool - Whether to use symbolic search
                    - hybrid_weight: float - Weight for hybrid search (0=symbolic only, 1=semantic only)
                    - embedding_model: str - Model to use for semantic search
                    - symbolic_method: str - Method for symbolic search ('tfidf' or 'bm25')
                - rerank: Dict with settings for reranking
                    - method: str - Reranking method ('biencoder', 'crossencoder', 'rrf', or None)
                    - model: str - Model to use for reranking
                - limits: Dict with retrieval limits
                    - top_k: int - Number of results to return
                    - top_p: float - Probability threshold for results
            
        Returns:
            List[Dict[str, Any]]: Retrieval results with scores and metadata
        """
        # Default configuration
        default_config = {
            "parallelism": {
                "enabled": False,
                "number_path": 1
            },
            "query_processing": {
                "rephrase": False,
                "hypothesize": False,
                "model": "Qwen2.5-1.5B"
            },
            "search": {
                "semantic": True,
                "symbolic": False,
                "hybrid_weight": 0.7,  # 0.7 semantic, 0.3 symbolic
                "embedding_model": "Jina-embeddings-v3",
                "symbolic_method": "tfidf"
            },
            "rerank": {
                "method": None,  # 'biencoder', 'crossencoder', 'rrf'
                "model": None
            },
            "limits": {
                "top_k": 10,
                "top_p": None
            }
        }
        
        # Merge provided config with defaults
        if config:
            merged_config = default_config.copy()
            for section in config:
                if section in merged_config and isinstance(config[section], dict):
                    merged_config[section].update(config[section])
                else:
                    merged_config[section] = config[section]
            config = merged_config
        else:
            config = default_config
            
        logger.debug(f"Retrieval configuration: {config}")
        
        # Get base corpus from vector database
        corpus = self._get_corpus_from_vector_db() # This now processes self.vector_df
        if not corpus: # Check if the processed list is empty
            logger.warning("Corpus is empty after processing vector_df. No data to retrieve.")
            return []
        
        # ===== 1. Query Preprocessing with Multiple Paths =====
        # Determine number of parallel paths to process
        number_paths = config["parallelism"]["number_path"] if config["parallelism"]["enabled"] else 1
        logger.debug(f"Using {number_paths} parallel query path(s)")
        
        # Initialize list to store processed queries
        processed_queries = []
        
        # For the first path, always use the original query as a starting point
        processed_queries.append(query)
        
        # Generate additional query paths if needed
        if number_paths > 1:
            # First try to decompose the query if possible
            try:
                decomposed_queries = self.query_processor.decompose_query(
                    query,
                    model=config["query_processing"]["model"]
                )
                # Extract sub-queries from decomposition result
                if isinstance(decomposed_queries, list) and len(decomposed_queries) > 0:
                    for i, sub_query in enumerate(decomposed_queries):
                        if isinstance(sub_query, dict) and 'query' in sub_query:
                            processed_queries.append(sub_query['query'])
                        elif isinstance(sub_query, str):
                            processed_queries.append(sub_query)
                        
                        # Limit to requested number of paths
                        if len(processed_queries) >= number_paths:
                            break
                            
                logger.debug(f"Generated {len(processed_queries)-1} additional query paths through decomposition")
            except Exception as e:
                logger.error(f"Query decomposition failed: {e}")
        
        # If we still need more paths, use rephrasing to generate variations
        while len(processed_queries) < number_paths:
            try:
                # Use the original query for rephrasing to ensure diversity
                rephrased_query = self.query_processor.rephrase_query(
                    query,
                    model=config["query_processing"]["model"],
                    temperature=0.7  # Use higher temperature for diversity
                )
                if rephrased_query and rephrased_query not in processed_queries:
                    processed_queries.append(rephrased_query)
                    logger.debug(f"Generated additional query path through rephrasing")
            except Exception as e:
                logger.error(f"Additional query path generation failed: {e}")
                # If we can't generate more, break to avoid infinite loop
                break
        
        # Apply additional processing to each query path
        final_processed_queries = []
        for i, path_query in enumerate(processed_queries):
            current_query = path_query
            
            # Apply rephrasing if configured (only for non-rephrased paths)
            if config["query_processing"]["rephrase"] and (i == 0 or len(processed_queries) <= 1):
                try:
                    rephrased_query = self.query_processor.rephrase_query(
                        current_query, 
                        model=config["query_processing"]["model"]
                    )
                    current_query = rephrased_query
                    logger.debug(f"Path {i+1}: Rephrased query: '{path_query}' -> '{current_query}'")
                except Exception as e:
                    logger.error(f"Path {i+1}: Query rephrasing failed: {e}")
                    # Keep current query
            
            # Apply hypothetical document embedding if configured
            if config["query_processing"]["hypothesize"]:
                try:
                    hypothetical_doc = self.query_processor.hypothesize_query(
                        current_query,
                        model=config["query_processing"]["model"]
                    )
                    current_query = hypothetical_doc
                    logger.debug(f"Path {i+1}: Generated hypothetical document of length {len(hypothetical_doc)}")
                except Exception as e:
                    logger.error(f"Path {i+1}: Hypothetical document generation failed: {e}")
                    # Keep current query
            
            final_processed_queries.append({
                "path_id": i+1,
                "original": path_query,
                "processed": current_query
            })
        
        logger.debug(f"Generated {len(final_processed_queries)} final query paths")
        
        # ===== 2. Search Phase with Multiple Paths =====
        all_path_results = []
        
        # Process each query path
        for path in final_processed_queries:
            path_id = path["path_id"]
            processed_query = path["processed"]
            path_results = []
            
            logger.debug(f"Processing search for path {path_id}")
            
            # Semantic search
            if config["search"]["semantic"]:
                try:
                    semantic_results = self._semantic_search(
                        processed_query,
                        corpus,
                        embedding_model=config["search"]["embedding_model"],
                        top_k=None,  # Don't filter yet
                        top_p=None   # Don't filter yet
                    )
                    
                    # Tag results with path_id for tracking
                    for result in semantic_results:
                        result["path_id"] = path_id
                        
                    path_results.extend(semantic_results)
                    logger.debug(f"Path {path_id}: Semantic search found {len(semantic_results)} results")
                except Exception as e:
                    logger.error(f"Path {path_id}: Semantic search failed: {e}")
            
            # Symbolic search
            if config["search"]["symbolic"]:
                try:
                    symbolic_results = self._symbolic_search(
                        processed_query,
                        corpus,
                        method=config["search"]["symbolic_method"],
                        top_k=None,  # Don't filter yet
                        top_p=None   # Don't filter yet
                    )
                    
                    # Tag results with path_id for tracking
                    for result in symbolic_results:
                        result["path_id"] = path_id
                        
                    path_results.extend(symbolic_results)
                    logger.debug(f"Path {path_id}: Symbolic search found {len(symbolic_results)} results")
                except Exception as e:
                    logger.error(f"Path {path_id}: Symbolic search failed: {e}")
            
            # Combine results if both search methods were used for this path
            if config["search"]["semantic"] and config["search"]["symbolic"] and path_results:
                try:
                    path_results = self._combine_search_results(
                        path_results, 
                        hybrid_weight=config["search"]["hybrid_weight"]
                    )
                    logger.debug(f"Path {path_id}: Combined {len(path_results)} results using hybrid weighting")
                except Exception as e:
                    logger.error(f"Path {path_id}: Failed to combine search results: {e}")
                    # Sort results by score if combining failed
                    path_results = sorted(path_results, key=lambda x: x.get('score', 0), reverse=True)
            
            # Add path results to all results
            all_path_results.extend(path_results)
        
        # Combine results from all paths
        results = all_path_results
        
        # Return early if no results
        if not results:
            logger.warning("No search results found across all paths")
            return []
            
        # ===== 3. Reranking Phase =====
        if config["rerank"]["method"] and (config["parallelism"]["enabled"] or (config["search"]["semantic"] and config["search"]["symbolic"])):
            try:
                if config["rerank"]["method"] == "biencoder":
                    results = self.rerank_processor.rerank_biencoder(
                        query=query,  # Use original query
                        model=config["rerank"]["model"],
                        results=results,
                        top_k=None  # Apply limits after reranking
                    )
                    logger.debug(f"Reranked with biencoder, got {len(results)} results")
                    
                elif config["rerank"]["method"] == "crossencoder":
                    results = self.rerank_processor.rerank_crossencoder(
                        query=query,  # Use original query
                        model=config["rerank"]["model"],
                        results=results,
                        top_k=None  # Apply limits after reranking
                    )
                    logger.debug(f"Reranked with crossencoder, got {len(results)} results")
                    
                elif config["rerank"]["method"] == "rrf":
                    results = self.rerank_processor.rerank_reciprocal_rank_fusion(
                        results=results,
                        top_k=None  # Apply limits after reranking
                    )
                    logger.debug(f"Reranked with reciprocal rank fusion, got {len(results)} results")
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
        
        # ===== 4. Apply Limits =====
        if config["limits"]["top_p"] or config["limits"]["top_k"]:
            try:
                results = self._apply_limits(
                    results,
                    top_k=config["limits"]["top_k"],
                    top_p=config["limits"]["top_p"]
                )
                logger.debug(f"Applied limits, final result count: {len(results)}")
            except Exception as e:
                logger.error(f"Failed to apply limits: {e}")
                # Apply simple top-k limit if the function fails
                if config["limits"]["top_k"] and len(results) > config["limits"]["top_k"]:
                    results = results[:config["limits"]["top_k"]]
        
        return results

    def _get_corpus_from_vector_db(self):
        """Extract corpus data from vector database, ensuring chunk IDs are retained.
        
        Returns:
            List[Dict[str, Any]]: List of documents with content, metadata, and chunk IDs
        """
        if self.vector_df is None or self.vector_df.empty:
            logger.warning("Vector DataFrame is None or empty in _get_corpus_from_vector_db.")
            return []
            
        try:
            corpus_list = []
            for index, row in self.vector_df.iterrows():
                # Ensure 'embedding' uses 'corpus_vector' primarily, or 'embedding' if that's the column name.
                # The schema uses 'corpus_vector'.
                embedding_data = row.get('corpus_vector', row.get('embedding', [])) # Default to empty list if neither found
                
                # Construct metadata: include all columns except the primary ones being mapped.
                metadata = row.to_dict()
                # Remove fields that are explicitly mapped to top-level keys in processed_doc to avoid duplication.
                # Also remove any vector columns from metadata to keep it light.
                # Assuming 'chunk_id', 'corpus', 'corpus_vector', 'hierarchy_vector' are primary or vector columns.
                # Other columns like 'document_id', 'document_name', 'reference', etc., are good metadata.
                primary_fields_to_exclude = {'chunk_id', 'corpus', 'corpus_vector', 'hierarchy_vector', 'embedding'}
                metadata = {k: v for k, v in metadata.items() if k not in primary_fields_to_exclude}

                processed_doc = {
                    'id': row.get('chunk_id', str(index)), 
                    'content': row.get('corpus', ''),      
                    'embedding': embedding_data,      
                    'metadata': metadata # Store all other columns as metadata
                }
                corpus_list.append(processed_doc)
            
            logger.debug(f"Transformed {len(corpus_list)} documents from vector_df to list format.")
            return corpus_list
            
        except Exception as e:
            logger.error(f"Error transforming vector_df to corpus list: {e}")
            logger.debug(traceback.format_exc())
            return []

    def _semantic_search(self, query, corpus, top_k=None, top_p=None, embedding_model=None):
        """Retrieve documents using semantic vector similarity.
        
        Args:
            query: Query string or embedding vector
            corpus: List of documents to search in, each with content and optional embedding
            top_k: Number of top results to return (optional)
            top_p: Probability threshold for results (optional)
            embedding_model: Name of embedding model to use (optional, uses default if None)
            
        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        if not corpus:
            logger.warning("Corpus is empty")
            return []
            
        try:
            # Get query embedding if string was provided
            query_embedding = None
            if isinstance(query, str):
                if embedding_model:
                    query_embedding = self.generator.get_embedding(query, model=embedding_model)
                else:
                    query_embedding = self.generator.get_embedding(query)
            else:
                # Query is already an embedding vector
                query_embedding = query
            
            # Calculate similarities
            results = []
            
            for i, doc in enumerate(corpus):
                # Get document embedding
                doc_embedding = None
                if 'embedding' in doc:
                    doc_embedding = np.array(doc['embedding'])
                else:
                    # Generate embedding if not present
                    doc_embedding = self.generator.get_embedding(doc['content'], model=embedding_model)
                
                # Calculate similarity (cosine similarity)
                similarity = 1 - cosine(query_embedding, doc_embedding)
                
                # Add to results
                results.append({
                    'id': doc.get('id', str(i)),
                    'content': doc.get('content', ''),
                    'metadata': doc.get('metadata', {}),
                    'score': float(similarity),
                    'source': 'semantic'
                })
            
            # Sort by score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Apply limits if specified
            if top_k or top_p:
                results = self._apply_limits(results, top_k, top_p)
                
            return results
                
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

    def _symbolic_search(self, query, corpus, method="tfidf", top_k=None, top_p=None):
        """Retrieve documents using symbolic scoring (TF-IDF or BM25).
        
        Args:
            query: Query string
            corpus: List of documents to search in
            method: Scoring method ('tfidf' or 'bm25')
            top_k: Number of top results to return (optional)
            top_p: Probability threshold for results (optional)
            
        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        if not corpus:
            logger.warning("Corpus is empty")
            return []
        try:
            # Extract content from corpus for vectorization
            corpus_content = [doc.get('content', '') for doc in corpus]
            if method == "tfidf":
                # Fit vectorizer if not already fitted or if corpus changed
                corpus_hash = hash(tuple(corpus_content))
                if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or \
                   getattr(self, '_last_tfidf_corpus_hash', None) != corpus_hash:
                    self.tfidf_vectorizer.fit(corpus_content)
                    self._last_tfidf_corpus_hash = corpus_hash
                query_vec = self.tfidf_vectorizer.transform([query])
                doc_vectors = self.tfidf_vectorizer.transform(corpus_content)
                similarities = (query_vec * doc_vectors.T).toarray()[0]
            elif method == "bm25":
                if self.nlp is None:
                    try:
                        if spacy.util.is_package("en_core_web_sm"):
                            self.nlp = spacy.load("en_core_web_sm")
                        else:
                            spacy.cli.download("en_core_web_sm")
                            self.nlp = spacy.load("en_core_web_sm")
                    except Exception as e:
                        logger.error(f"Failed to load spacy model: {e}")
                        return []
                corpus_hash = hash(tuple(corpus_content))
                if getattr(self, '_last_bm25_corpus_hash', None) != corpus_hash or self.bm25 is None:
                    tokenized_corpus = [[token.text for token in self.nlp(doc)] for doc in corpus_content]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    self._last_bm25_corpus_hash = corpus_hash
                tokenized_query = [token.text for token in self.nlp(query)]
                similarities = self.bm25.get_scores(tokenized_query)
                max_score = max(similarities) if hasattr(similarities, '__iter__') and len(similarities) > 0 else 1
                similarities = similarities / max_score if max_score > 0 else similarities
            else:
                logger.error(f"Unknown symbolic search method: {method}")
                return []
            results = []
            for i, (doc, score) in enumerate(zip(corpus, similarities)):
                results.append({
                    'id': doc.get('id', str(i)),
                    'content': doc.get('content', ''),
                    'metadata': doc.get('metadata', {}),
                    'score': float(score),
                    'source': method
                })
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            if top_k or top_p:
                results = self._apply_limits(results, top_k, top_p)
            return results
        except Exception as e:
            logger.error(f"Symbolic search failed: {e}")
            return []

    def _apply_limits(self, results: List[Dict[str, Any]], top_k: Optional[int] = None, top_p: Optional[float] = None) -> List[Dict[str, Any]]:
        """Apply limits to retrieval results based on top_k and top_p.

        Args:
            results: List of retrieval results (dictionaries with 'score').
            top_k: Number of top results to return.
            top_p: Minimum score threshold for results.

        Returns:
            List[Dict[str, Any]]: Limited retrieval results.
        """
        if not results:
            return []

        # Apply top_p threshold first (filter by minimum score)
        if top_p is not None:
            results = [res for res in results if res.get('score', 0) >= top_p]
            logger.debug(f"Applied top_p={top_p}, {len(results)} results remaining.")

        # Then apply top_k limit
        if top_k is not None and len(results) > top_k:
            results = results[:top_k]
            logger.debug(f"Applied top_k={top_k}, {len(results)} results remaining.")
        
        return results

    def _combine_search_results(self, results: List[Dict[str, Any]], hybrid_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Combine results from semantic and symbolic search methods.
        
        Args:
            results: List of retrieval results from both search methods
            hybrid_weight: Weight for hybrid search (0=symbolic only, 1=semantic only)
            
        Returns:
            List[Dict[str, Any]]: Combined search results
        """
        # Separate results by source
        semantic_results = [res for res in results if res.get('source') == 'semantic']
        symbolic_results = [res for res in results if res.get('source') in ['tfidf', 'bm25']]
        
        # Calculate weights for each source
        semantic_weight = hybrid_weight
        symbolic_weight = 1.0 - hybrid_weight
        
        # Combine results
        combined_results = []
        for res in semantic_results:
            res['score'] *= semantic_weight
            combined_results.append(res)
        for res in symbolic_results:
            res['score'] *= symbolic_weight
            combined_results.append(res)
        
        # Sort by combined score
        combined_results = sorted(combined_results, key=lambda x: x['score'], reverse=True)
        
        return combined_results

class GraphRetriever:
    """Enhanced retrieval engine for graph databases with parameterizable topology.
    
    This class works with graph databases and query inputs to enable advanced
    graph-based search capabilities with configurable retrieval pipeline topologies.
    
    Features:
    - Query preprocessing (rephrasing, hypothesizing)
    - Multiple graph retrieval methods (path-based, semantic, structural)
    - Support for both standard graphs and hypergraphs
    - Reranking methods (bi-encoder, cross-encoder, reciprocal rank fusion)
    - Configurable retrieval topology (sequential or parallel)
    """
    
    def __init__(self, graph_db=None, generator=None, graph_type='standard'):
        """Initialize GraphRetriever with configurable components.
        
        Args:
            graph_db: Graph database instance, GraphBuilder instance, or path to graph file
            generator: Generator instance for embeddings, completions, and query processing
            graph_type: Type of graph to prioritize ('standard' or 'hypergraph')
        """
        logger.debug("GraphRetriever initialization started")
        try:
            start_time = time.time()
            
            # Core components
            self.graph_db = None
            self.graph = None
            self.hypergraph = None
            self.generator = generator if generator else Generator()
            self.nx = __import__('networkx')
            self.graph_type = graph_type
            
            # Set up directory structure (same as GraphBuilder)
            self.db_dir = Path.cwd() / "db"
            self.graph_dir = self.db_dir / "graph"
            self.vector_dir = self.db_dir / "vector"
            
            # Initialize component processors
            self.query_processor = QueryProcessor(generator=self.generator)
            self.rerank_processor = RerankProcessor(generator=self.generator)
            
            # Load graph database
            if graph_db:
                self._load_graph_db(graph_db)
                
            # Validate graph schema if graph is loaded
            if self.graph:
                self._validate_graph_schema()
            
            logger.debug(f"GraphRetriever initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"GraphRetriever initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise

    def _load_graph_db(self, graph_db):
        """Load graph database from different sources.
        
        Args:
            graph_db: Graph database instance, GraphBuilder instance, or path to graph file
        """
        try:
            # Handle different graph DB types
            if isinstance(graph_db, self.nx.Graph) or isinstance(graph_db, self.nx.DiGraph):
                # Direct graph object
                self.graph = graph_db
                self.graph_db = None
                logger.debug("Using provided graph object directly")
            elif hasattr(graph_db, 'graph') and isinstance(graph_db.graph, self.nx.Graph):
                # GraphBuilder or similar object with graph attribute
                self.graph_db = graph_db
                self.graph = graph_db.graph
                if hasattr(graph_db, 'hypergraph'):
                    self.hypergraph = graph_db.hypergraph
                logger.debug(f"Using graph from {type(graph_db).__name__} instance")
            elif isinstance(graph_db, str):
                # File path to graph database
                graph_path = None
                
                # Handle both relative and absolute paths
                if os.path.isabs(graph_db):
                    graph_path = Path(graph_db)
                else:
                    # Try as a relative path to graph_dir
                    graph_path = self.graph_dir / graph_db
                
                # If path doesn't exist but doesn't have .pkl extension, try adding it
                if not os.path.exists(graph_path) and not str(graph_path).endswith('.pkl'):
                    graph_path = Path(str(graph_path) + '.pkl')
                
                # If still doesn't exist, try applying naming convention
                if not os.path.exists(graph_path) and not os.path.basename(graph_db).startswith('g_'):
                    # Try with standard naming convention
                    graph_path = self.graph_dir / f"g_{graph_db}_{self.graph_type}.pkl"
                
                # If still doesn't exist, try to find a matching file
                if not os.path.exists(graph_path):
                    # Look for files matching the pattern g_*_{graph_type}.pkl
                    graph_files = list(self.graph_dir.glob(f'g_*_{self.graph_type}.pkl'))
                    if graph_files:
                        # Use the most recently modified file
                        graph_path = sorted(graph_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
                        logger.debug(f"Using most recent {self.graph_type} graph database file: {graph_path}")
                    else:
                        logger.error(f"No {self.graph_type} graph database files found in {self.graph_dir}")
                        return
                
                logger.debug(f"Loading graph from file: {graph_path}")
                try:
                    # Import GraphBuilder here to avoid circular imports
                    from src.pipeline.processing.dbbuilder import GraphBuilder
                    graph_builder = GraphBuilder()
                    
                    # Load graphs based on priority
                    if self.graph_type == 'standard':
                        # Load standard graph first, then hypergraph if available
                        self.graph = graph_builder.load_db(str(graph_path), db_type='standard')
                        self.graph_db = graph_builder
                        try:
                            # Try to load hypergraph with matching name
                            hypergraph_path = str(graph_path).replace('_standard.pkl', '_hypergraph.pkl')
                            if os.path.exists(hypergraph_path):
                                self.hypergraph = graph_builder.load_db(hypergraph_path, db_type='hypergraph')
                            else:
                                logger.debug("Matching hypergraph file not found")
                        except Exception as hypergraph_error:
                            logger.debug(f"Hypergraph not available: {hypergraph_error}")
                    else:  # hypergraph priority
                        # Load hypergraph first, then standard graph if available
                        hypergraph_path = str(graph_path)
                        if '_standard.pkl' in hypergraph_path:
                            hypergraph_path = hypergraph_path.replace('_standard.pkl', '_hypergraph.pkl')
                        
                        if os.path.exists(hypergraph_path):
                            self.hypergraph = graph_builder.load_db(hypergraph_path, db_type='hypergraph')
                            self.graph_db = graph_builder
                        
                        # Try to load standard graph
                        standard_path = str(graph_path)
                        if '_hypergraph.pkl' in standard_path:
                            standard_path = standard_path.replace('_hypergraph.pkl', '_standard.pkl')
                        elif not '_standard.pkl' in standard_path:
                            standard_path = str(graph_path).replace('.pkl', '_standard.pkl')
                        
                        if os.path.exists(standard_path):
                            self.graph = graph_builder.load_db(standard_path, db_type='standard')
                        else:
                            logger.debug("Matching standard graph file not found")
                except Exception as file_error:
                    logger.error(f"Failed to load graph from file: {file_error}")
            else:
                logger.error(f"Unsupported graph database type: {type(graph_db)}")
                
            if self.graph:
                logger.debug(f"Graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            if self.hypergraph:
                logger.debug(f"Hypergraph loaded with {self.hypergraph.number_of_nodes()} nodes and {self.hypergraph.number_of_edges()} edges")
                
        except ImportError:
            logger.error("NetworkX not found. Please install with: pip install networkx")
            raise
        except Exception as e:
            logger.error(f"Failed to load graph database: {e}")
            logger.debug(f"Load error details: {traceback.format_exc()}")
    
    def _validate_graph_schema(self):
        """Validate that loaded graph conforms to expected schema.
        
        Checks for required node and edge attributes based on GraphBuilder schema.
        Logs warnings if schema doesn't match expectations.
        """
        if not self.graph:
            return
            
        try:
            # Define expected attributes based on GraphBuilder schema
            expected_node_attrs = ['node_type']
            expected_edge_attrs = ['edge_type']
            
            # Sample nodes to check schema (limit to 10 for performance)
            sample_nodes = list(self.graph.nodes)[:min(10, len(self.graph.nodes))]
            for node in sample_nodes:
                node_data = self.graph.nodes[node]
                for attr in expected_node_attrs:
                    if attr not in node_data:
                        logger.warning(f"Node {node} missing expected attribute: {attr}")
            
            # Sample edges to check schema (limit to 10 for performance)
            sample_edges = list(self.graph.edges)[:min(10, len(self.graph.edges))]
            for edge in sample_edges:
                edge_data = self.graph.edges[edge]
                for attr in expected_edge_attrs:
                    if attr not in edge_data:
                        logger.warning(f"Edge {edge} missing expected attribute: {attr}")
                        
            # Check for REFERENCES edges (important for LLM-based cross-reference extraction)
            reference_edges = [e for e in self.graph.edges if self.graph.edges[e].get('edge_type') == 'REFERENCES']
            if not reference_edges:
                logger.info("No REFERENCES edges found in graph. Cross-reference search may be limited.")
            else:
                logger.debug(f"Found {len(reference_edges)} REFERENCES edges for cross-reference search")
                
        except Exception as e:
            logger.warning(f"Graph schema validation failed: {e}")
    
    def retrieve(self, query, config=None):
        """Main retrieval method with configurable topology.
        
        Args:
            query: Original query string
            config: Retrieval configuration with these optional parameters:
                - query_processing: Dict with settings for query preprocessing
                    - rephrase: bool - Whether to rephrase the query
                    - hypothesize: bool - Whether to use Hypothetical Document Embedding
                    - model: str - Model to use for query processing
                - search: Dict with settings for search methods
                    - path_based: bool - Whether to use path-based search
                    - semantic: bool - Whether to use semantic cluster search
                    - structural: bool - Whether to use structural search
                    - community: bool - Whether to use community detection
                    - hypergraph: bool - Whether to use hypergraph search
                    - embedding_model: str - Model to use for semantic search
                - rerank: Dict with settings for reranking
                    - method: str - Reranking method ('biencoder', 'crossencoder', 'rrf', or None)
                    - model: str - Model to use for reranking
                - limits: Dict with retrieval limits
                    - top_k: int - Number of results to return
                    - top_p: float - Probability threshold for results
                - parallelism: Dict with settings for parallel processing
                    - enabled: bool - Whether to enable parallel processing
                    - number_path: int - Number of parallel paths to process
                
        Returns:
            List[Dict[str, Any]]: Retrieval results with scores and metadata
        """
        # Default configuration
        default_config = {
            "parallelism": {
                "enabled": False,
                "number_path": 1
            },
            "query_processing": {
                "rephrase": False,
                "hypothesize": False,
                "model": "Qwen2.5-1.5B"
            },
            "search": {
                "path_based": True,
                "semantic": True,
                "structural": False,
                "community": False,
                "hypergraph": True,
                "embedding_model": "Jina-embeddings-v3"
            },
            "rerank": {
                "method": None,  # 'biencoder', 'crossencoder', 'rrf'
                "model": None
            },
            "limits": {
                "top_k": 10,
                "top_p": None
            }
        }
        
        # Merge provided config with defaults
        if config:
            merged_config = default_config.copy()
            for section in config:
                if section in merged_config and isinstance(config[section], dict):
                    merged_config[section].update(config[section])
                else:
                    merged_config[section] = config[section]
            config = merged_config
        else:
            config = default_config
            
        logger.debug(f"Retrieval configuration: {config}")
        
        # Validate graph database is loaded
        if not self.graph:
            logger.error("No graph database loaded")
            return []
        
        # ===== 1. Query Preprocessing with Multiple Paths =====
        # Determine number of parallel paths to process
        number_paths = config["parallelism"]["number_path"] if config["parallelism"]["enabled"] else 1
        logger.debug(f"Using {number_paths} parallel query path(s)")
        
        # Initialize list to store processed queries
        processed_queries = []
        
        # For the first path, always use the original query as a starting point
        processed_queries.append({"path_id": 1, "original": query, "processed": query})
        
        # Process query with hypothetical embedding if enabled
        if config["query_processing"]["hypothesize"] and len(processed_queries) < number_paths:
            try:
                hypothetical_query = self.query_processor.hypothesize_query(
                    query,
                    model=config["query_processing"]["model"]
                )
                if hypothetical_query and hypothetical_query != query:
                    processed_queries.append({
                        "path_id": len(processed_queries) + 1,
                        "original": query,
                        "processed": hypothetical_query
                    })
                logger.debug(f"Generated hypothetical query: {hypothetical_query[:50]}...")
            except Exception as e:
                logger.error(f"Query hypothesizing failed: {e}")
        
        # Process query with rephrasing if enabled
        if config["query_processing"]["rephrase"] and len(processed_queries) < number_paths:
            try:
                rephrased_query = self.query_processor.rephrase_query(
                    query,
                    model=config["query_processing"]["model"]
                )
                if rephrased_query and rephrased_query != query:
                    processed_queries.append({
                        "path_id": len(processed_queries) + 1,
                        "original": query,
                        "processed": rephrased_query
                    })
                logger.debug(f"Generated rephrased query: {rephrased_query}")
            except Exception as e:
                logger.error(f"Query rephrasing failed: {e}")
        
        # Generate additional query paths if needed through decomposition
        if len(processed_queries) < number_paths:
            try:
                decomposed_queries = self.query_processor.decompose_query(
                    query,
                    model=config["query_processing"]["model"]
                )
                # Extract sub-queries from decomposition result
                if isinstance(decomposed_queries, list) and len(decomposed_queries) > 0:
                    for i, sub_query in enumerate(decomposed_queries):
                        if isinstance(sub_query, dict) and 'query' in sub_query:
                            processed_queries.append({
                                "path_id": len(processed_queries) + 1,
                                "original": query,
                                "processed": sub_query['query']
                            })
                        elif isinstance(sub_query, str):
                            processed_queries.append({
                                "path_id": len(processed_queries) + 1,
                                "original": query,
                                "processed": sub_query
                            })
                        
                        # Limit to requested number of paths
                        if len(processed_queries) >= number_paths:
                            break
                            
                logger.debug(f"Generated {len(processed_queries)-1} additional query paths through decomposition")
            except Exception as e:
                logger.error(f"Query decomposition failed: {e}")
        
        # ===== 2. Search Phase with Multiple Paths =====
        all_path_results = []
        
        # Process each query path
        for path in processed_queries:
            path_id = path["path_id"]
            processed_query = path["processed"]
            path_results = []
            
            logger.debug(f"Processing search for path {path_id}")
            
            # Get query embedding for semantic search methods
            query_embedding = None
            if config["search"]["semantic"] or config["search"]["hypergraph"]:
                try:
                    query_embedding = self.generator.get_embedding(
                        processed_query,
                        model=config["search"]["embedding_model"]
                    )
                except Exception as e:
                    logger.error(f"Failed to generate query embedding: {e}")
            
            # Path-based search
            if config["search"]["path_based"]:
                try:
                    path_based_results = self._path_based_search(
                        processed_query,
                        top_k=None  # Don't filter yet
                    )
                    
                    # Tag results with path_id for tracking
                    for result in path_based_results:
                        result["path_id"] = path_id
                        
                    path_results.extend(path_based_results)
                    logger.debug(f"Path {path_id}: Path-based search found {len(path_based_results)} results")
                except Exception as e:
                    logger.error(f"Path {path_id}: Path-based search failed: {e}")
            
            # Semantic search
            if config["search"]["semantic"] and query_embedding is not None:
                try:
                    semantic_results = self._semantic_search(
                        query_embedding,
                        top_k=None  # Don't filter yet
                    )
                    
                    # Tag results with path_id for tracking
                    for result in semantic_results:
                        result["path_id"] = path_id
                        
                    path_results.extend(semantic_results)
                    logger.debug(f"Path {path_id}: Semantic search found {len(semantic_results)} results")
                except Exception as e:
                    logger.error(f"Path {path_id}: Semantic search failed: {e}")
            
            # Structural search
            if config["search"]["structural"]:
                try:
                    structural_results = self._structural_search(
                        processed_query,
                        top_k=None  # Don't filter yet
                    )
                    
                    # Tag results with path_id for tracking
                    for result in structural_results:
                        result["path_id"] = path_id
                        
                    path_results.extend(structural_results)
                    logger.debug(f"Path {path_id}: Structural search found {len(structural_results)} results")
                except Exception as e:
                    logger.error(f"Path {path_id}: Structural search failed: {e}")
            
            # Community search
            if config["search"]["community"]:
                try:
                    community_results = self._community_search(
                        processed_query,
                        top_k=None  # Don't filter yet
                    )
                    
                    # Tag results with path_id for tracking
                    for result in community_results:
                        result["path_id"] = path_id
                        
                    path_results.extend(community_results)
                    logger.debug(f"Path {path_id}: Community search found {len(community_results)} results")
                except Exception as e:
                    logger.error(f"Path {path_id}: Community search failed: {e}")
            
            # Hypergraph search
            if config["search"]["hypergraph"] and self.hypergraph and query_embedding is not None:
                try:
                    hypergraph_results = self._hypergraph_search(
                        query_embedding,
                        processed_query,
                        top_k=None  # Don't filter yet
                    )
                    
                    # Tag results with path_id for tracking
                    for result in hypergraph_results:
                        result["path_id"] = path_id
                        
                    path_results.extend(hypergraph_results)
                    logger.debug(f"Path {path_id}: Hypergraph search found {len(hypergraph_results)} results")
                except Exception as e:
                    logger.error(f"Path {path_id}: Hypergraph search failed: {e}")
            
            # Add path results to all results
            all_path_results.extend(path_results)
        
        # Combine results from all paths
        results = all_path_results
        
        # Return early if no results
        if not results:
            logger.warning("No search results found across all paths")
            return []
        
        # ===== 3. Reranking Phase =====
        if config["rerank"]["method"] and (config["parallelism"]["enabled"] or len(processed_queries) > 1):
            try:
                if config["rerank"]["method"] == "biencoder":
                    results = self.rerank_processor.rerank_biencoder(
                        query=query,  # Use original query
                        model=config["rerank"]["model"],
                        results=results,
                        top_k=None  # Apply limits after reranking
                    )
                    logger.debug(f"Reranked with biencoder, got {len(results)} results")
                    
                elif config["rerank"]["method"] == "crossencoder":
                    results = self.rerank_processor.rerank_crossencoder(
                        query=query,  # Use original query
                        model=config["rerank"]["model"],
                        results=results,
                        top_k=None  # Apply limits after reranking
                    )
                    logger.debug(f"Reranked with crossencoder, got {len(results)} results")
                    
                elif config["rerank"]["method"] == "rrf":
                    results = self.rerank_processor.rerank_reciprocal_rank_fusion(
                        results=results,
                        top_k=None  # Apply limits after reranking
                    )
                    logger.debug(f"Reranked with reciprocal rank fusion, got {len(results)} results")
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
        
        # ===== 4. Apply Limits =====
        if config["limits"]["top_p"] or config["limits"]["top_k"]:
            try:
                results = self._apply_limits(
                    results,
                    top_k=config["limits"]["top_k"],
                    top_p=config["limits"]["top_p"]
                )
                logger.debug(f"Applied limits, final result count: {len(results)}")
            except Exception as e:
                logger.error(f"Failed to apply limits: {e}")
                # Apply simple top-k limit if the function fails
                if config["limits"]["top_k"] and len(results) > config["limits"]["top_k"]:
                    results = results[:config["limits"]["top_k"]]
        
        return results
    
    def _apply_limits(self, results: List[Dict[str, Any]], top_k: Optional[int] = None, top_p: Optional[float] = None) -> List[Dict[str, Any]]:
        """Apply limits to retrieval results based on top_k and top_p.

        Args:
            results: List of retrieval results (dictionaries with 'score').
            top_k: Number of top results to return.
            top_p: Minimum score threshold for results.

        Returns:
            List[Dict[str, Any]]: Limited retrieval results.
        """
        if not results:
            return []

        # Apply top_p threshold first (filter by minimum score)
        if top_p is not None:
            results = [res for res in results if res.get('score', 0) >= top_p]
            logger.debug(f"Applied top_p={top_p}, {len(results)} results remaining.")

        # Then apply top_k limit
        if top_k is not None and len(results) > top_k:
            results = results[:top_k]
            logger.debug(f"Applied top_k={top_k}, {len(results)} results remaining.")
        
        return results
        
    def _semantic_search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search nodes by semantic similarity to query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return (optional)
            
        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        logger.debug("Starting semantic search")
        
        try:
            results = []
            
            # Find nodes with embeddings
            nodes_with_embeddings = []
            for node, data in self.graph.nodes(data=True):
                if 'embedding' in data:
                    nodes_with_embeddings.append((node, data))
            
            logger.debug(f"Found {len(nodes_with_embeddings)} nodes with embeddings")
            
            # Calculate similarity for each node
            for node, data in nodes_with_embeddings:
                try:
                    # Calculate cosine similarity
                    similarity = 1 - cosine(query_embedding, data['embedding'])
                    
                    # Create result entry
                    result = {
                        'id': node,
                        'score': float(similarity),
                        'content': data.get('content', ''),
                        'metadata': {
                            'node_type': data.get('node_type', 'unknown'),
                            'source': 'semantic'
                        }
                    }
                    
                    # Add additional metadata
                    for key, value in data.items():
                        if key not in ['embedding', 'content'] and not isinstance(value, (list, dict, np.ndarray)):
                            result['metadata'][key] = value
                    
                    results.append(result)
                except Exception as node_error:
                    logger.error(f"Error processing node {node}: {node_error}")
            
            # Sort by similarity score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Apply top_k if specified
            if top_k and len(results) > top_k:
                results = results[:top_k]
                
            logger.debug(f"Semantic search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _path_based_search(self, query: str, max_path_length: int = 3, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find paths in the graph related to the query.
        
        Args:
            query: Query string
            max_path_length: Maximum path length to consider
            top_k: Number of top results to return (optional)
            
        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        logger.debug(f"Starting path-based search with max_path_length={max_path_length}")
        
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            logger.debug(f"Extracted keywords: {keywords}")
            
            # Find relevant nodes based on keywords
            start_nodes = []
            for node, data in self.graph.nodes(data=True):
                node_text = data.get('content', '') or str(data)
                if any(keyword.lower() in node_text.lower() for keyword in keywords):
                    start_nodes.append(node)
            
            logger.debug(f"Found {len(start_nodes)} potential start nodes")
            
            # Find paths between relevant nodes
            results = []
            processed_paths = set()
            
            for i, start_node in enumerate(start_nodes[:10]):  # Limit to 10 start nodes for performance
                for j, end_node in enumerate(start_nodes[i+1:10]):  # Limit to 10 end nodes for performance
                    try:
                        # Find all simple paths up to max_path_length
                        paths = list(self.nx.all_simple_paths(
                            self.graph, 
                            source=start_node, 
                            target=end_node, 
                            cutoff=max_path_length
                        ))
                        
                        for path in paths:
                            path_key = tuple(sorted(path))  # Create a unique key for the path
                            if path_key in processed_paths:
                                continue
                            processed_paths.add(path_key)
                            
                            # Calculate path score based on keyword matches, path length, and REFERENCES edges
                            path_text = ""
                            reference_edge_count = 0
                            
                            # Check for REFERENCES edges in the path
                            for i in range(len(path) - 1):
                                source, target = path[i], path[i+1]
                                if self.graph.has_edge(source, target):
                                    edge_data = self.graph.edges[source, target]
                                    if edge_data.get('edge_type') == 'REFERENCES':
                                        reference_edge_count += 1
                            
                            # Collect text from nodes
                            for node in path:
                                node_data = self.graph.nodes[node]
                                path_text += node_data.get('content', '') or str(node_data)
                            
                            # Count keyword matches in path
                            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in path_text.lower())
                            
                            # Calculate base score (higher score for shorter paths with more matches)
                            base_score = keyword_matches / (len(path) * 0.5)
                            
                            # Boost score for paths with REFERENCES edges (from LLM-based cross-reference extraction)
                            reference_boost = 1.0 + (reference_edge_count * 0.2)  # 20% boost per REFERENCES edge
                            path_score = base_score * reference_boost
                            
                            # Create result entry
                            result = {
                                'id': f"path_{len(results)}",
                                'score': float(path_score),
                                'content': f"Path: {' -> '.join(str(n) for n in path)}",
                                'metadata': {
                                    'path': path,
                                    'path_length': len(path),
                                    'start_node': start_node,
                                    'end_node': end_node,
                                    'source': 'path_based'
                                }
                            }
                            
                            results.append(result)
                    except Exception as path_error:
                        logger.error(f"Error finding paths between {start_node} and {end_node}: {path_error}")
            
            # Sort by score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Apply top_k if specified
            if top_k and len(results) > top_k:
                results = results[:top_k]
                
            logger.debug(f"Path-based search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Path-based search failed: {e}")
            return []
    
    def _structural_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search based on graph structure and node importance.
        
        Args:
            query: Query string
            top_k: Number of top results to return (optional)
            
        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        logger.debug("Starting structural search")
        
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Calculate centrality measures
            try:
                degree_centrality = self.nx.degree_centrality(self.graph)
                betweenness_centrality = self.nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
            except Exception as centrality_error:
                logger.error(f"Error calculating centrality: {centrality_error}")
                degree_centrality = {}
                betweenness_centrality = {}
            
            # Find relevant nodes based on keywords and centrality
            results = []
            for node, data in self.graph.nodes(data=True):
                node_text = data.get('content', '') or str(data)
                
                # Calculate keyword relevance
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in node_text.lower())
                keyword_score = keyword_matches / max(1, len(keywords))
                
                # Get centrality scores
                degree_score = degree_centrality.get(node, 0)
                betweenness_score = betweenness_centrality.get(node, 0)
                
                # Combined score (weighted average)
                combined_score = (0.6 * keyword_score) + (0.2 * degree_score) + (0.2 * betweenness_score)
                
                if combined_score > 0:  # Only include nodes with some relevance
                    # Create result entry
                    result = {
                        'id': node,
                        'score': float(combined_score),
                        'content': node_text[:500],  # Limit content length
                        'metadata': {
                            'node_type': data.get('node_type', 'unknown'),
                            'degree_centrality': degree_score,
                            'betweenness_centrality': betweenness_score,
                            'keyword_score': keyword_score,
                            'source': 'structural'
                        }
                    }
                    
                    # Add additional metadata
                    for key, value in data.items():
                        if key not in ['embedding', 'content'] and not isinstance(value, (list, dict, np.ndarray)):
                            result['metadata'][key] = value
                    
                    results.append(result)
            
            # Sort by score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Apply top_k if specified
            if top_k and len(results) > top_k:
                results = results[:top_k]
                
            logger.debug(f"Structural search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Structural search failed: {e}")
            return []
    
    def _community_search(self, query: str, algorithm: str = "louvain", top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find communities in the graph related to the query.
        
        Args:
            query: Query string
            algorithm: Community detection algorithm ('louvain', 'label_propagation', 'greedy_modularity')
            top_k: Number of top results to return (optional)
            
        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        logger.debug(f"Starting community search with algorithm={algorithm}")
        
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Detect communities
            communities = {}
            
            try:
                if algorithm == "louvain":
                    communities = best_partition(self.graph)
                elif algorithm == "label_propagation":
                    communities = self.nx.algorithms.community.label_propagation_communities(self.graph)
                    communities = {node: i for i, comm in enumerate(communities) for node in comm}
                elif algorithm == "greedy_modularity":
                    communities = self.nx.algorithms.community.greedy_modularity_communities(self.graph)
                    communities = {node: i for i, comm in enumerate(communities) for node in comm}
                else:
                    logger.warning(f"Unknown community detection algorithm: {algorithm}, using louvain")
                    from community import best_partition
                    communities = best_partition(self.graph)
            except Exception as community_error:
                logger.error(f"Error detecting communities: {community_error}")
                # Create dummy communities based on connected components
                components = self.nx.connected_components(self.graph.to_undirected())
                communities = {node: i for i, comp in enumerate(components) for node in comp}
            
            # Group nodes by community
            community_nodes = {}
            for node, community_id in communities.items():
                if community_id not in community_nodes:
                    community_nodes[community_id] = []
                community_nodes[community_id].append(node)
            
            # Calculate community relevance to query
            results = []
            for community_id, nodes in community_nodes.items():
                # Combine text from all nodes in community
                community_text = ""
                for node in nodes:
                    if node in self.graph.nodes:
                        node_data = self.graph.nodes[node]
                        community_text += node_data.get('content', '') or str(node_data)
                
                # Calculate keyword relevance
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in community_text.lower())
                keyword_score = keyword_matches / max(1, len(keywords))
                
                # Adjust score based on community size (prefer medium-sized communities)
                size_factor = 1.0
                community_size = len(nodes)
                if community_size < 3:
                    size_factor = 0.5  # Penalize very small communities
                elif community_size > 50:
                    size_factor = 0.7  # Penalize very large communities
                
                combined_score = keyword_score * size_factor
                
                if combined_score > 0:  # Only include communities with some relevance
                    # Create result entry
                    result = {
                        'id': f"community_{community_id}",
                        'score': float(combined_score),
                        'content': f"Community {community_id} with {len(nodes)} nodes",
                        'metadata': {
                            'community_id': community_id,
                            'community_size': len(nodes),
                            'nodes': nodes[:100],  # Limit number of nodes in metadata
                            'keyword_score': keyword_score,
                            'size_factor': size_factor,
                            'source': 'community'
                        }
                    }
                    
                    results.append(result)
            
            # Sort by score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Apply top_k if specified
            if top_k and len(results) > top_k:
                results = results[:top_k]
                
            logger.debug(f"Community search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Community search failed: {e}")
            return []
    
    def _hypergraph_search(self, query_embedding: np.ndarray, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search using hypergraph relations when available.
        
        Args:
            query_embedding: Query embedding vector
            query: Query string
            top_k: Number of top results to return (optional)
            
        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        logger.debug("Starting hypergraph search")
        
        if not self.hypergraph:
            logger.warning("Hypergraph not available")
            return []
        
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Find hyperedges
            hyperedges = [n for n, d in self.hypergraph.nodes(data=True) 
                         if d.get('type') == 'hyperedge']
            
            logger.debug(f"Found {len(hyperedges)} hyperedges")
            
            # Find relevant nodes through hyperedges
            results = []
            processed_nodes = set()
            
            for edge in hyperedges:
                edge_data = self.hypergraph.nodes[edge]
                edge_type = edge_data.get('edge_type', 'unknown')
                
                # Get nodes connected to this hyperedge
                connected_nodes = list(self.hypergraph.neighbors(edge))
                
                # Skip if no nodes connected
                if not connected_nodes:
                    continue
                
                # Process each connected node
                for node in connected_nodes:
                    # Skip if already processed
                    if node in processed_nodes:
                        continue
                    processed_nodes.add(node)
                    
                    node_data = self.hypergraph.nodes[node]
                    node_text = node_data.get('content', '') or str(node_data)
                    
                    # Calculate score components
                    semantic_score = 0.0
                    keyword_score = 0.0
                    
                    # Semantic similarity if embedding available
                    if 'embedding' in node_data:
                        semantic_score = 1 - cosine(query_embedding, node_data['embedding'])
                    
                    # Keyword matching
                    keyword_matches = sum(1 for keyword in keywords if keyword.lower() in node_text.lower())
                    keyword_score = keyword_matches / max(1, len(keywords))
                    
                    # Combined score with edge type weighting
                    edge_weight = 1.0
                    if edge_type == 'semantic_cluster':
                        edge_weight = 1.2  # Boost semantic clusters
                    elif edge_type == 'reference':
                        edge_weight = 1.1  # Boost references
                    elif edge_type == 'REFERENCES':  # LLM-based cross-reference extraction
                        edge_weight = 1.3  # Higher boost for LLM-extracted references
                    
                    combined_score = ((0.7 * semantic_score) + (0.3 * keyword_score)) * edge_weight
                    
                    if combined_score > 0:  # Only include nodes with some relevance
                        # Create result entry
                        result = {
                            'id': node,
                            'score': float(combined_score),
                            'content': node_text[:500],  # Limit content length
                            'metadata': {
                                'node_type': node_data.get('node_type', 'unknown'),
                                'edge_type': edge_type,
                                'semantic_score': semantic_score,
                                'keyword_score': keyword_score,
                                'edge_weight': edge_weight,
                                'source': 'hypergraph'
                            }
                        }
                        
                        # Add additional metadata
                        for key, value in node_data.items():
                            if key not in ['embedding', 'content'] and not isinstance(value, (list, dict, np.ndarray)):
                                result['metadata'][key] = value
                        
                        results.append(result)
            
            # Sort by score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Apply top_k if specified
            if top_k and len(results) > top_k:
                results = results[:top_k]
                
            logger.debug(f"Hypergraph search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hypergraph search failed: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query string.
        
        Args:
            query: Query string
            
        Returns:
            List[str]: Extracted keywords
        """
        try:
            # Simple keyword extraction by removing stopwords
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            
            # Split query into words and remove stopwords
            words = query.split()
            keywords = [word for word in words if word.lower() not in stop_words and len(word) > 2]
            
            # If no keywords found, use all words
            if not keywords:
                keywords = words
                
            return keywords
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            # Fallback to simple word splitting
            return [word for word in query.split() if len(word) > 2]
    
    def semantic_cluster_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search through semantic clusters in the hypergraph.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Ranked list of relevant sections
        """
        logger.debug(f"Starting semantic_cluster_search with top_k={top_k}")
        start_time = time.time()
        
        try:
            if not hasattr(self.graph_builder, 'semantic_hypergraph'):
                logger.error("Semantic hypergraph not built")
                raise ValueError("Semantic hypergraph not built")
                
            H = self.graph_builder.semantic_hypergraph
            results = []
            
            # Find semantic cluster nodes
            logger.debug("Finding semantic cluster nodes in hypergraph")
            semantic_clusters = [n for n, d in H.nodes(data=True) 
                               if d.get('type') == 'hyperedge' and 
                               d.get('edge_type') == 'semantic_cluster']
            logger.debug(f"Found {len(semantic_clusters)} semantic cluster nodes")
            
            for i, cluster in enumerate(semantic_clusters):
                # Get sections in this cluster
                sections = [n for n in H.neighbors(cluster) if H.nodes[n].get('type') == 'section']
                logger.debug(f"Cluster {i+1}/{len(semantic_clusters)} has {len(sections)} sections")
                
                # Calculate average similarity to query
                similarities = []
                for section in sections:
                    if 'embedding' in H.nodes[section]:
                        sim = 1 - cosine(query_embedding, H.nodes[section]['embedding'])
                        similarities.append((section, sim))
            
                if similarities:
                    # Add top section from each relevant cluster
                    best_section, best_sim = max(similarities, key=lambda x: x[1])
                    
                    # Add to results
                    results.append({
                        'id': best_section,
                        'score': best_sim,
                        'content': H.nodes[best_section].get('content', ''),
                        'metadata': {
                            'source': 'semantic_cluster',
                            'cluster_id': cluster
                        }
                    })
                    
            # Sort results by similarity score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Truncate to top_k
            if top_k and len(results) > top_k:
                results = results[:top_k]
                
            logger.debug(f"Semantic cluster search completed in {time.time() - start_time:.2f} seconds, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic cluster search failed: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            return []

    def concept_hierarchy_search(self, query: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Search through concept hierarchies in the semantic hypergraph.
        
        Args:
            query: Search query
            max_depth: Maximum depth to traverse in concept hierarchy
            
        Returns:
            List[Dict[str, Any]]: Related concepts and sections
        """
        if not hasattr(self.graph_builder, 'semantic_hypergraph'):
            raise ValueError("Semantic hypergraph not built")
            
        H = self.graph_builder.semantic_hypergraph
        results = []
        
        # Find concept hierarchy edges
        concept_edges = [n for n, d in H.nodes(data=True)
                        if d.get('type') == 'hyperedge' and
                        d.get('edge_type') == 'concept_hierarchy']
        
        def traverse_concepts(edge, depth=0):
            if depth >= max_depth:
                return
                
            edge_data = H.nodes[edge]
            parent = edge_data.get('parent_concept', '')
            children = edge_data.get('child_concepts', [])
            
            # Check if query matches any concepts
            if (query.lower() in parent.lower() or 
                any(query.lower() in child.lower() for child in children)):
                
                # Get sections connected to this concept
                sections = [n for n in H.neighbors(edge) if H.nodes[n].get('type') == 'section']
                results.append({
                    'parent_concept': parent,
                    'child_concepts': children,
                    'sections': sections,
                    'depth': depth
                })
                
                # Recursively check related concepts
                for child in children:
                    child_edges = [n for n, d in H.nodes(data=True)
                                 if d.get('type') == 'hyperedge' and
                                 d.get('edge_type') == 'concept_hierarchy' and
                                 d.get('parent_concept') == child]
                    for child_edge in child_edges:
                        traverse_concepts(child_edge, depth + 1)
        
        for edge in concept_edges:
            traverse_concepts(edge)
            
        return results

    def temporal_search(self, start_date: str, end_date: str = None) -> List[Dict[str, Any]]:
        """Search for content within a time period in the multilayer hypergraph.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: Optional end date string (YYYY-MM-DD)
            
        Returns:
            List[Dict[str, Any]]: Time-relevant sections
        """
        if not hasattr(self.graph_builder, 'multilayer_hypergraph'):
            raise ValueError("Multilayer hypergraph not built")
            
        layers = self.graph_builder.multilayer_hypergraph
        metadata_layer = layers.get('metadata')
        if not metadata_layer:
            raise ValueError("Metadata layer not found")
            
        results = []
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else None
        
        # Find temporal nodes in metadata layer
        temporal_nodes = [n for n, d in metadata_layer.nodes(data=True)
                         if d.get('type') == 'metadata' and d.get('field') == 'date']
        
        for node in temporal_nodes:
            node_data = metadata_layer.nodes[node]
            node_date = pd.to_datetime(node_data.get('value'))
            
            if start <= node_date and (not end or node_date <= end):
                # Get sections connected to this date
                sections = [n for n in metadata_layer.neighbors(node)]
                results.append({
                    'date': node_date,
                    'sections': sections
                })
        
        return sorted(results, key=lambda x: x['date'])

    def cross_layer_search(self, query: str, layer_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Search across multiple layers of the multilayer hypergraph.
        
        Args:
            query: Search query
            layer_weights: Optional weights for each layer (default: equal weights)
            
        Returns:
            List[Dict[str, Any]]: Cross-layer search results
        """
        if not hasattr(self.graph_builder, 'multilayer_hypergraph'):
            raise ValueError("Multilayer hypergraph not built")
            
        layers = self.graph_builder.multilayer_hypergraph
        if not layer_weights:
            layer_weights = {name: 1.0 for name in layers.keys()}
            
        results = {}
        
        for layer_name, layer in layers.items():
            weight = layer_weights.get(layer_name, 1.0)
            
            if layer_name == 'content':
                # Use semantic similarity for content layer
                matches = self._content_layer_search(layer, query)
            elif layer_name == 'structure':
                # Use hierarchy traversal for structure layer
                matches = self._structure_layer_search(layer, query)
            elif layer_name == 'reference':
                # Use reference graph traversal
                matches = self._reference_layer_search(layer, query)
            else:
                # Use basic attribute matching for other layers
                matches = self._attribute_layer_search(layer, query)
                
            # Combine scores across layers
            for section_id, score in matches:
                if section_id not in results:
                    results[section_id] = 0
                results[section_id] += score * weight
        
        # Return normalized results
        return [{'reference': k, 'score': v} for k, v in 
                sorted(results.items(), key=lambda x: x[1], reverse=True)]

    def _content_layer_search(self, layer, query: str) -> List[Tuple[str, float]]:
        """Search within content layer using semantic similarity."""
        results = []
        for node, data in layer.nodes(data=True):
            if data.get('type') == 'section' and 'embedding' in data:
                sim = 1 - cosine(query_embedding, data['embedding'])
                results.append((node, sim))
        return results

    def _structure_layer_search(self, layer, query: str) -> List[Tuple[str, float]]:
        """Search within structure layer using hierarchy."""
        results = []
        for node, data in layer.nodes(data=True):
            if data.get('type') == 'section':
                # Score based on hierarchy level match
                hierarchy = data.get('hierarchy', '').lower()
                if query.lower() in hierarchy:
                    level = data.get('level', 0)
                    score = 1.0 / (level + 1)  # Higher score for top levels
                    results.append((node, score))
        return results

    def _reference_layer_search(self, layer, query: str) -> List[Tuple[str, float]]:
        """Search within reference layer using graph traversal."""
        results = []
        # Find nodes matching query
        matches = [n for n in layer.nodes() if query.lower() in n.lower()]
        
        for match in matches:
            # Use PageRank to score connected nodes
            personalization = {match: 1.0}
            scores = self.nx.pagerank(layer, personalization=personalization)
            results.extend(scores.items())
        return results

    def _attribute_layer_search(self, layer, query: str) -> List[Tuple[str, float]]:
        """Search within attribute-based layers (metadata, access)."""
        results = []
        for node, data in layer.nodes(data=True):
            if data.get('type') == 'section':
                # Score based on attribute matches
                matches = sum(1 for v in data.values() 
                            if isinstance(v, str) and query.lower() in v.lower())
                if matches:
                    results.append((node, matches))
        return results
        
    def path_based_search(self, start_node: str, end_node: str, 
                          max_path_length: int = 4) -> List[Dict[str, Any]]:
        """Find all paths between two nodes in the standard graph.
        
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_path_length: Maximum path length to consider
            
        Returns:
            List[Dict[str, Any]]: List of paths with metadata
        """
        if not hasattr(self.graph_builder, 'graph'):
            raise ValueError("Standard graph not built")
            
        G = self.graph_builder.graph
        results = []
        
        # Find all simple paths within length limit
        try:
            all_paths = list(self.nx.all_simple_paths(
                G, source=start_node, target=end_node, cutoff=max_path_length
            ))
            
            for i, path in enumerate(all_paths):
                path_edges = []
                path_weight = 0
                
                # Extract edge metadata
                for j in range(len(path) - 1):
                    src, dst = path[j], path[j+1]
                    edge_data = G.get_edge_data(src, dst)
                    
                    # Handle case of multiple edges between nodes
                    if isinstance(edge_data, dict) and len(edge_data) > 0:
                        # If multiple edges exist, choose the first one
                        if 0 in edge_data:
                            edge_attr = edge_data[0]
                        else:
                            # Get the first edge key and its data
                            first_key = list(edge_data.keys())[0]
                            edge_attr = edge_data[first_key]
                            
                        edge_type = edge_attr.get('type', 'unknown')
                        weight = edge_attr.get('weight', 1.0)
                        
                        path_edges.append({
                            'source': src,
                            'target': dst,
                            'type': edge_type,
                            'weight': weight
                        })
                        
                        path_weight += weight
                
                # Calculate path score (inversely proportional to length)
                path_score = path_weight / len(path)
                
                results.append({
                    'path': path,
                    'path_edges': path_edges,
                    'length': len(path) - 1,
                    'score': path_score
                })
            
            # Sort by path score
            return sorted(results, key=lambda x: x['score'], reverse=True)
            
        except (self.nx.NetworkXNoPath, self.nx.NodeNotFound) as e:
            logger.warning(f"No path found: {e}")
            return []
            
    def connectivity_search(self, nodes: List[str], 
                           connectivity_metric: str = 'shortest_path') -> Dict[str, Any]:
        """Analyze connectivity between a set of nodes.
        
        Args:
            nodes: List of node IDs to analyze
            connectivity_metric: One of 'shortest_path', 'clustering', 'centrality'
            
        Returns:
            Dict[str, Any]: Connectivity analysis results
        """
        if not hasattr(self.graph_builder, 'graph'):
            raise ValueError("Standard graph not built")
            
        G = self.graph_builder.graph
        valid_nodes = [n for n in nodes if n in G]
        
        if len(valid_nodes) < 2:
            return {'error': 'Need at least two valid nodes for connectivity analysis'}
            
        results = {
            'nodes': valid_nodes,
            'valid_node_count': len(valid_nodes),
            'analysis_type': connectivity_metric
        }
        
        if connectivity_metric == 'shortest_path':
            # Find shortest paths between all pairs
            paths = {}
            for i in range(len(valid_nodes)):
                for j in range(i+1, len(valid_nodes)):
                    src, dst = valid_nodes[i], valid_nodes[j]
                    try:
                        path = self.nx.shortest_path(G, source=src, target=dst)
                        length = len(path) - 1
                        paths[f"{src}_to_{dst}"] = {
                            'path': path,
                            'length': length
                        }
                    except (self.nx.NetworkXNoPath, self.nx.NodeNotFound):
                        paths[f"{src}_to_{dst}"] = {'path': [], 'length': float('inf')}
            
            # Calculate average path length
            finite_lengths = [p['length'] for p in paths.values() if p['length'] < float('inf')]
            avg_length = sum(finite_lengths) / len(finite_lengths) if finite_lengths else float('inf')
            
            results['paths'] = paths
            results['average_path_length'] = avg_length
            results['connectivity_score'] = 1.0 / avg_length if avg_length > 0 else 0
            
        elif connectivity_metric == 'clustering':
            # Extract subgraph
            subgraph = G.subgraph(valid_nodes)
            
            # Get clustering coefficient
            clustering = self.nx.clustering(subgraph)
            avg_clustering = sum(clustering.values()) / len(clustering) if clustering else 0
            
            results['node_clustering'] = clustering
            results['average_clustering'] = avg_clustering
            results['connectivity_score'] = avg_clustering
            
        elif connectivity_metric == 'centrality':
            # Extract subgraph
            subgraph = G.subgraph(valid_nodes)
            
            # Calculate various centrality measures
            degree_centrality = self.nx.degree_centrality(subgraph)
            betweenness_centrality = self.nx.betweenness_centrality(subgraph)
            
            results['degree_centrality'] = degree_centrality
            results['betweenness_centrality'] = betweenness_centrality
            results['connectivity_score'] = sum(degree_centrality.values()) / len(degree_centrality)
            
        else:
            results['error'] = f"Unknown connectivity metric: {connectivity_metric}"
            
        return results
        
    def community_detection(self, algorithm: str = 'louvain', 
                           resolution: float = 1.0) -> Dict[str, Any]:
        """Detect communities in the graph using various algorithms.
        
        Args:
            algorithm: One of 'louvain', 'label_propagation', 'greedy_modularity'
            resolution: Resolution parameter for community detection (for louvain)
            
        Returns:
            Dict[str, Any]: Community detection results
        """
        if not hasattr(self.graph_builder, 'graph'):
            raise ValueError("Standard graph not built")
            
        G = self.graph_builder.graph
        
        # Convert MultiDiGraph to simple undirected graph for community detection
        simple_G = self.nx.Graph()
        for u, v, data in G.edges(data=True):
            # If edge already exists, we don't add it again or increment
            if not simple_G.has_edge(u, v):
                simple_G.add_edge(u, v, weight=data.get('weight', 1.0))
                
        results = {
            'algorithm': algorithm,
            'node_count': simple_G.number_of_nodes()
        }
        
        try:
            communities = {}
            
            if algorithm == 'louvain':
                # Import community detection algorithm
                from community import best_partition
                
                partition = best_partition(simple_G, resolution=resolution)
                # Reformat partition to communities
                communities_dict = {}
                for node, community_id in partition.items():
                    if community_id not in communities_dict:
                        communities_dict[community_id] = []
                    communities_dict[community_id].append(node)
                
                communities = list(communities_dict.values())
                
            elif algorithm == 'label_propagation':
                communities = list(self.nx.algorithms.community.label_propagation_communities(simple_G))
                
            elif algorithm == 'greedy_modularity':
                communities = list(self.nx.algorithms.community.greedy_modularity_communities(simple_G))
                
            else:
                return {'error': f"Unknown community detection algorithm: {algorithm}"}
            
            # Calculate modularity
            modularity = self.nx.algorithms.community.modularity(simple_G, communities)
            
            results['communities'] = [list(c) for c in communities]
            results['community_count'] = len(communities)
            results['modularity'] = modularity
            
            # Calculate additional statistics
            community_sizes = [len(c) for c in communities]
            results['avg_community_size'] = sum(community_sizes) / len(community_sizes)
            results['max_community_size'] = max(community_sizes)
            results['min_community_size'] = min(community_sizes)
            
            return results
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {'error': f"Community detection failed: {str(e)}"}
            
    def hypergraph_query(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """Query hypergraph structures with various specialized algorithms.
        
        Args:
            query_type: One of 'entity_influence', 'knowledge_gap', 'concept_diffusion'
            **kwargs: Additional arguments for specific query types
            
        Returns:
            List[Dict[str, Any]]: Hypergraph query results
        """
        if not hasattr(self.graph_builder, 'hypergraph'):
            raise ValueError("Hypergraph not built")
            
        H = self.graph_builder.hypergraph
        results = []
        
        if query_type == 'entity_influence':
            # Calculate influence of entities by hyperedge membership analysis
            min_hyperedges = kwargs.get('min_hyperedges', 3)
            
            # Count hyperedge membership for each section node
            membership_count = {}
            for node in H.nodes():
                if H.nodes[node].get('type') == 'section':
                    # Get all hyperedges this node belongs to
                    hyperedges = [n for n in H.neighbors(node) 
                                if H.nodes[n].get('type') == 'hyperedge']
                    membership_count[node] = len(hyperedges)
            
            # Filter by minimum hyperedge count
            influential_nodes = {node: count for node, count in membership_count.items() 
                               if count >= min_hyperedges}
            
            # Calculate influence score based on hyperedge diversity
            node_scores = {}
            for node, count in influential_nodes.items():
                hyperedges = [n for n in H.neighbors(node) if H.nodes[n].get('type') == 'hyperedge']
                
                # Measure diversity by counting different edge types
                edge_types = set(H.nodes[he].get('edge_type', 'unknown') for he in hyperedges)
                
                # Influence = membership count * edge type diversity
                node_scores[node] = count * len(edge_types)
            
            # Return sorted by influence score
            for node, score in sorted(node_scores.items(), key=lambda x: x[1], reverse=True):
                results.append({
                    'node': node,
                    'influence_score': score,
                    'hyperedge_count': influential_nodes[node]
                })
                
        elif query_type == 'knowledge_gap':
            # Identify potential gaps in the knowledge base
            min_cluster_size = kwargs.get('min_cluster_size', 3)
            max_connections = kwargs.get('max_connections', 2)
            
            # Find document clusters (hyperedges with document_group type)
            doc_clusters = [n for n, d in H.nodes(data=True)
                          if d.get('type') == 'hyperedge' and 
                          d.get('edge_type') == 'document_group']
            
            for cluster in doc_clusters:
                # Get sections in this document cluster
                sections = [n for n in H.neighbors(cluster)]
                
                if len(sections) >= min_cluster_size:
                    # Look for sections with few connections to other document clusters
                    for section in sections:
                        # Get all hyperedges this section belongs to
                        section_hyperedges = [n for n in H.neighbors(section)
                                           if H.nodes[n].get('type') == 'hyperedge']
                        
                        # Count connections to other document clusters
                        other_doc_connections = sum(1 for he in section_hyperedges
                                                if H.nodes[he].get('edge_type') == 'document_group'
                                                and he != cluster)
                        
                        if other_doc_connections <= max_connections:
                            results.append({
                                'section': section,
                                'document_cluster': cluster,
                                'cluster_size': len(sections),
                                'other_connections': other_doc_connections,
                                'gap_score': 1.0 / (other_connections + 1)
                            })
            
            # Sort by gap score
            results = sorted(results, key=lambda x: x['gap_score'], reverse=True)
                
        elif query_type == 'concept_diffusion':
            # Track how concepts spread through the hypergraph
            concept_name = kwargs.get('concept', '')
            max_distance = kwargs.get('max_distance', 3)
            
            if not concept_name:
                return [{'error': 'Concept name required for concept diffusion query'}]
                
            # Find hyperedges related to this concept
            concept_edges = [n for n, d in H.nodes(data=True)
                          if d.get('type') == 'hyperedge' and 
                          ((d.get('edge_type') == 'key_concept' and 
                           d.get('concept', '').lower() == concept_name.lower()) or
                           (d.get('edge_type') == 'concept_hierarchy' and 
                           (d.get('parent_concept', '').lower() == concept_name.lower() or
                            concept_name.lower() in [c.lower() for c in d.get('child_concepts', [])])))]
            
            if not concept_edges:
                return [{'error': f"Concept '{concept_name}' not found in the hypergraph"}]
                
            # Track diffusion from these concept edges
            diffusion_map = {}
            visited = set()
            
            def traverse_diffusion(node, distance=0):
                if distance > max_distance or node in visited:
                    return
                    
                visited.add(node)
                
                # Store node in diffusion map
                if distance not in diffusion_map:
                    diffusion_map[distance] = []
                    
                diffusion_map[distance].append(node)
                
                # Traverse neighbors
                for neighbor in H.neighbors(node):
                    traverse_diffusion(neighbor, distance + 1)
            
            # Start traversal from each concept edge
            for edge in concept_edges:
                traverse_diffusion(edge)
            
            # Format results
            for distance, nodes in sorted(diffusion_map.items()):
                hyperedges = [n for n in nodes if H.nodes[n].get('type') == 'hyperedge']
                sections = [n for n in nodes if H.nodes[n].get('type') == 'section']
                
                results.append({
                    'distance': distance,
                    'hyperedge_count': len(hyperedges),
                    'section_count': len(sections),
                    'hyperedges': hyperedges[:10],  # Limit to avoid huge results
                    'sections': sections[:10]       # Limit to avoid huge results
                })
                
        elif query_type == 'structural_patterns':
            # Detect recurring structural patterns in the hypergraph
            min_pattern_size = kwargs.get('min_pattern_size', 3)
            max_patterns = kwargs.get('max_patterns', 10)
            
            # Find patterns of interconnected hyperedges
            patterns = []
            
            # Group hyperedges by type
            hyperedge_types = {}
            for node, data in H.nodes(data=True):
                if data.get('type') == 'hyperedge':
                    edge_type = data.get('edge_type', 'unknown')
                    if edge_type not in hyperedge_types:
                        hyperedge_types[edge_type] = []
                    hyperedge_types[edge_type].append(node)
            
            # Compare patterns within each edge type
            for edge_type, edges in hyperedge_types.items():
                for i in range(len(edges)):
                    edge1 = edges[i]
                    neighbors1 = set(H.neighbors(edge1))
                    
                    if len(neighbors1) < min_pattern_size:
                        continue
                        
                    # Find similar hyperedges (with similar section connections)
                    for j in range(i+1, len(edges)):
                        edge2 = edges[j]
                        neighbors2 = set(H.neighbors(edge2))
                        
                        if len(neighbors2) < min_pattern_size:
                            continue
                            
                        # Calculate Jaccard similarity
                        intersection = len(neighbors1.intersection(neighbors2))
                        union = len(neighbors1.union(neighbors2))
                        
                        if union > 0:
                            similarity = intersection / union
                            
                            if similarity > 0.5:  # Significant overlap
                                common_nodes = neighbors1.intersection(neighbors2)
                                
                                if len(common_nodes) >= min_pattern_size:
                                    patterns.append({
                                        'edge_type': edge_type,
                                        'hyperedges': [edge1, edge2],
                                        'common_nodes': list(common_nodes),
                                        'similarity': similarity,
                                        'pattern_size': len(common_nodes)
                                    })
            
            # Sort by pattern size and similarity
            patterns.sort(key=lambda x: (x['pattern_size'], x['similarity']), reverse=True)
            results = patterns[:max_patterns]  # Limit number of patterns
            
        else:
            results = [{'error': f"Unknown hypergraph query type: {query_type}"}]
            
        return results
        
    def multilayer_hypergraph_fusion(self, query: str, 
                                    layer_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Perform fusion search across all layers in multilayer hypergraph.
        
        This advanced search combines results from different hypergraph layers,
        using an ensemble approach with customizable weights.
        
        Args:
            query: Search query
            layer_weights: Optional weights for each layer (default: equal weights)
            
        Returns:
            List[Dict[str, Any]]: Fusion search results
        """
        if not hasattr(self.graph_builder, 'multilayer_hypergraph'):
            raise ValueError("Multilayer hypergraph not built")
            
        layers = self.graph_builder.multilayer_hypergraph
        
        # Default layer weights (content prioritized)
        if layer_weights is None:
            layer_weights = {
                'content': 1.0,
                'structure': 0.7,
                'reference': 0.6,
                'metadata': 0.4,
                'access': 0.3
            }
        
        # Unified results dictionary (node_id -> score mapping)
        fusion_scores = {}
        layer_results = {}
        
        # Split query into semantic and keyword components
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        keywords = [word.lower() for word in query.split() if word.lower() not in stop_words]
        
        # Process each layer
        for layer_name, layer in layers.items():
            if layer_name not in layer_weights:
                continue
                
            # Layer-specific search strategy
            if layer_name == 'content':
                # Semantic search in content layer
                section_scores = {}
                
                # For each section node with embedding
                for node, data in layer.nodes(data=True):
                    if 'embedding' in data:
                        # Calculate similarity to query
                        text = data.get('text', '')
                        # Direct keyword matching (basic TF-IDF like scoring)
                        keyword_matches = 0
                        for keyword in keywords:
                            if keyword in text.lower():
                                keyword_matches += 1
                                
                        # Normalize by number of keywords
                        score = keyword_matches / len(keywords) if keywords else 0
                        section_scores[node] = score
                        
                layer_results[layer_name] = section_scores
                
            elif layer_name == 'structure':
                # Hierarchical search
                section_scores = {}
                
                # Search through hierarchy nodes
                for node, data in layer.nodes(data=True):
                    hierarchy = data.get('hierarchy', '')
                    level = data.get('level', 0)
                    
                    # Match query against hierarchy
                    if any(keyword in hierarchy.lower() for keyword in keywords):
                        # Score inversely proportional to level depth
                        score = 1.0 / (level + 1)
                        section_scores[node] = score
                        
                layer_results[layer_name] = section_scores
                
            elif layer_name == 'reference':
                # Reference graph analysis
                section_scores = {}
                
                # Find initial matches
                matched_nodes = [node for node in layer.nodes() 
                               if any(keyword in str(node).lower() for keyword in keywords)]
                
                # Use PageRank with personalization to extend from matches
                if matched_nodes:
                    personalization = {node: 1.0 for node in matched_nodes}
                    try:
                        scores = self.nx.pagerank(layer, personalization=personalization)
                        section_scores.update(scores)
                    except:
                        # Fallback if PageRank fails
                        for node in matched_nodes:
                            section_scores[node] = 1.0
                            # Add immediate neighbors with reduced scores
                            for neighbor in layer.neighbors(node):
                                section_scores[neighbor] = 0.5
                                
                layer_results[layer_name] = section_scores
                
            elif layer_name in ('metadata', 'access'):
                # Attribute matching for metadata/access layers
                section_scores = {}
                
                # Search through attributes
                for node, data in layer.nodes(data=True):
                    # Check all string attributes for keyword matches
                    matches = sum(1 for k, v in data.items() 
                                if isinstance(v, str) and 
                                any(keyword in v.lower() for keyword in keywords))
                    
                    if matches > 0:
                        section_scores[node] = matches
                        
                layer_results[layer_name] = section_scores
        
        # Combine layer results with weights
        for layer_name, scores in layer_results.items():
            weight = layer_weights.get(layer_name, 1.0)
            
            for node, score in scores.items():
                if node not in fusion_scores:
                    fusion_scores[node] = 0
                
                fusion_scores[node] += score * weight
        
        # Normalize and format results
        max_score = max(fusion_scores.values()) if fusion_scores else 1.0
        
        # Return normalized results
        ranked_results = [
            {
                'node': node,
                'score': score / max_score,
                'layer_scores': {
                    layer: layer_results[layer].get(node, 0) * layer_weights.get(layer, 0)
                    for layer in layer_results if node in layer_results[layer]
                }
            }
            for node, score in fusion_scores.items()
        ]
        
        return sorted(ranked_results, key=lambda x: x['score'], reverse=True)


class MemoryRetriever:
    """Enhanced retrieval engine for memory databases with parameterizable topology.
    
    This class works with memory databases and query inputs to enable advanced
    memory-based search capabilities with configurable retrieval pipeline topologies.
    
    Features:
    - Query preprocessing (rephrasing, hypothesizing)
    - Symbolic and semantic search with hybrid capabilities
    - Reranking methods (bi-encoder, cross-encoder, reciprocal rank fusion)
    - Configurable retrieval topology
    """
    
    def __init__(self, memory_db_path: Optional[Union[str, Path, pd.DataFrame]] = None, generator: Optional[Generator] = None, memory_type: str = "episodic"):
        """Initialize MemoryRetriever with configurable components.
        
        Args:
            memory_db_path: Path to the memory Parquet file or a pre-loaded DataFrame.
            generator: Generator instance for embeddings, completions, and query processing.
            memory_type: Type of memory to work with ("episodic" or "personality").
        """
        logger.debug(f"Initializing MemoryRetriever for '{memory_type}' memory from: {memory_db_path}")
        try:
            start_time = time.time()
            
            self.memory_df = None
            self.memory_db_path = None
            self.generator = generator if generator else Generator()
            self.memory_type = memory_type
            
            self.db_dir = Path.cwd() / "db" # Base DB directory
            self.memory_dir = self.db_dir / "memory" # Specific memory directory
            
            self.query_processor = QueryProcessor(generator=self.generator)
            self.rerank_processor = RerankProcessor(generator=self.generator)
            
            if not self.memory_dir.exists():
                self.memory_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created memory directory at {self.memory_dir}")

            self._load_memory_db(memory_db_path) # Call internal load method
            
            logger.debug(f"MemoryRetriever initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"MemoryRetriever initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def _load_memory_db(self, memory_db_input: Optional[Union[str, Path, pd.DataFrame]]):
        """Load memory database from path or use pre-loaded DataFrame.
        
        Args:
            memory_db_input: Path to memory Parquet file or a pre-loaded DataFrame.
        """
        default_columns = {
            "episodic": ['memory_id', 'query', 'entity', 'context', 'transformed_query', 'prompt_id'],
            "personality": ['mode_id', 'mode_name', 'personality_type', 'cognitive_style', 'mbti_type', 'mode_description', 'activation_contexts', 'activation_triggers']
        }
        
        if isinstance(memory_db_input, (str, Path)):
            self.memory_db_path = Path(memory_db_input)
            if not self.memory_db_path.is_absolute():
                 # Assume relative to standard memory directory if not absolute
                 self.memory_db_path = self.memory_dir / self.memory_db_path

            if self.memory_db_path.exists() and self.memory_db_path.suffix == '.parquet':
                logger.info(f"Loading MemoryDB from Parquet file: {self.memory_db_path}")
                self.memory_df = pd.read_parquet(self.memory_db_path)
                logger.info(f"MemoryDB loaded with {len(self.memory_df)} entries.")
            else:
                logger.error(f"MemoryDB Parquet file not found or invalid: {self.memory_db_path}")
                self.memory_df = pd.DataFrame(columns=default_columns.get(self.memory_type, []))
        elif isinstance(memory_db_input, pd.DataFrame):
            logger.info("Loading MemoryDB from pre-loaded DataFrame.")
            self.memory_df = memory_db_input
        else:
            logger.warning(f"No MemoryDB path or DataFrame provided for {self.memory_type} memory. Retriever will operate on an empty dataset.")
            self.memory_df = pd.DataFrame(columns=default_columns.get(self.memory_type, []))

    def retrieve(self, query, config=None):
        """Main retrieval method with configurable topology.
        
        Args:
            query: Original query string
            config: Retrieval configuration with these optional parameters:
                - query_processing: Dict with settings for query preprocessing
                    - rephrase: bool - Whether to rephrase the query
                    - hypothesize: bool - Whether to use Hypothetical Document Embedding
                    - model: str - Model to use for query processing
                - search: Dict with settings for search methods
                    - symbolic: bool - Whether to use symbolic search
                    - semantic: bool - Whether to use semantic similarity
                    - hybrid_weight: float - Weight for hybrid search (0=symbolic only, 1=semantic only)
                    - embedding_model: str - Model to use for semantic search
                - rerank: Dict with settings for reranking
                    - method: str - Reranking method ('biencoder', 'crossencoder', 'rrf', or None)
                    - model: str - Model to use for reranking
                - limits: Dict with retrieval limits
                    - top_k: int - Number of results to return
                    - top_p: float - Probability threshold for results
            
        Returns:
            List[Dict[str, Any]]: Retrieval results with scores and metadata
        """
        # Default configuration
        default_config = {
            "query_processing": {
                "rephrase": False,
                "hypothesize": False,
                "model": "Qwen2.5-1.5B"
            },
            "search": {
                "symbolic": True,
                "semantic": True,
                "hybrid_weight": 0.7,  # 0.7 semantic, 0.3 symbolic
                "embedding_model": "Jina-embeddings-v3"
            },
            "rerank": {
                "method": None,  # 'biencoder', 'crossencoder', 'rrf'
                "model": None
            },
            "limits": {
                "top_k": 10,
                "top_p": None
            }
        }
        
        # Merge provided config with defaults
        if config:
            merged_config = default_config.copy()
            for section in config:
                if section in merged_config:
                    if isinstance(config[section], dict) and isinstance(merged_config[section], dict):
                        merged_config[section].update(config[section])
                    else:
                        merged_config[section] = config[section]
                else:
                    merged_config[section] = config[section]
            config = merged_config
        else:
            config = default_config
        
        logger.debug(f"Memory retrieval started with config: {config}")
        
        try:
            start_time = time.time()
            
            # Check if memory_df is loaded and not empty
            if self.memory_df is None or self.memory_df.empty:
                logger.warning(f"Memory DataFrame for '{self.memory_type}' is None or empty. Cannot retrieve.")
                return []
            
            # Process query if enabled
            processed_query = query
            model = config["query_processing"]["model"]
            
            # Apply query processing in sequence if enabled
            if config["query_processing"]["rephrase"]:
                processed_query = self.query_processor.rephrase_query(
                    query=processed_query, # Use current state of processed_query
                    model=model,
                    temperature=0.5
                )
                logger.debug(f"Rephrased query: {processed_query}")
            
            if config["query_processing"]["hypothesize"]:
                processed_query = self.query_processor.hypothesize_query(
                    query=processed_query, # Use current state of processed_query
                    model=model,
                    temperature=0.5
                )
                logger.debug(f"Hypothesized query: {processed_query}")
                
            if processed_query != query: # Log if query was actually changed
                logger.debug(f"Final processed query for retrieval: {processed_query}")
            
            # Execute search based on configuration
            symbolic_results = []
            semantic_results = []
            
            # active_collection is now self.memory_df
            if self.memory_df.empty:
                logger.warning(f"No {self.memory_type} memories found in DataFrame.")
                return []
            
            # Symbolic search
            if config["search"]["symbolic"]:
                symbolic_results = self._symbolic_search(processed_query, self.memory_df)
                logger.debug(f"Symbolic search returned {len(symbolic_results)} results")
            
            # Semantic search
            if config["search"]["semantic"]:
                semantic_results = self._semantic_search(
                    processed_query,
                    self.memory_df, # Pass the DataFrame
                    embedding_model=config["search"]["embedding_model"]
                )
                logger.debug(f"Semantic search returned {len(semantic_results)} results")
            
            # Combine results using hybrid search
            results = self._hybrid_search(
                symbolic_results,
                semantic_results,
                config["search"]["hybrid_weight"]
            )
            
            # Rerank results if a method is specified
            if config["rerank"]["method"] and results:
                if config["rerank"]["method"] == "biencoder":
                    results = self.rerank_processor.rerank_biencoder(
                        processed_query,
                        config["rerank"]["model"],
                        results,
                        top_k=config["limits"]["top_k"]
                    )
                elif config["rerank"]["method"] == "crossencoder":
                    results = self.rerank_processor.rerank_crossencoder(
                        processed_query,
                        config["rerank"]["model"],
                        results,
                        top_k=config["limits"]["top_k"]
                    )
                elif config["rerank"]["method"] == "rrf":
                    results = self.rerank_processor.rerank_reciprocal_rank_fusion(
                        results,
                        top_k=config["limits"]["top_k"]
                    )
                logger.debug(f"Reranking with {config['rerank']['method']} returned {len(results)} results")
            
            # Apply limits
            results = self._apply_limits(results, config["limits"]["top_k"], config["limits"]["top_p"])
            
            logger.debug(f"Memory retrieval completed in {time.time() - start_time:.2f} seconds, {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {str(e)}")
            logger.debug(f"Retrieval error details: {traceback.format_exc()}")
            return []
    
    def _symbolic_search(self, query, collection):
        """Perform symbolic search on the provided collection.
        
        Args:
            query: Query string
            collection: Collection to search in
            
        Returns:
            List[Dict[str, Any]]: Search results with scores
        """
        results = []
        
        if self.memory_type == "episodic":
            # Search episodic memory by entity and context
            for index, memory_row in collection.iterrows():
                score = 0.0
                
                # Entity matching - direct match gets a higher score
                if "entity" in memory_row and pd.notna(memory_row["entity"]):
                    if memory_row["entity"].lower() in query.lower():
                        score += 0.5  # Higher score for entity match
                    elif query.lower() in memory_row["entity"].lower():
                        score += 0.4  # Lower score for partial match
                
                # Context matching - simplified JSON check
                if "context" in memory_row and pd.notna(memory_row["context"]):
                    # Assuming context could be a dict stored as a string, or already a dict
                    context_data = memory_row["context"]
                    if isinstance(context_data, str):
                        try:
                            context_data = json.loads(context_data) # Parse if string
                        except json.JSONDecodeError:
                            context_data = {} # Or handle error appropriately
                    
                    if isinstance(context_data, dict):
                        context_text = json.dumps(context_data).lower()
                        query_terms = query.lower().split()
                        matched_terms = sum(1 for term in query_terms if term in context_text)
                        if matched_terms > 0:
                            score += 0.3 * (matched_terms / len(query_terms))
                
                # Original query matching
                if "query" in memory_row and pd.notna(memory_row["query"]):
                    # Simple text matching (word overlap)
                    query_words = set(query.lower().split())
                    memory_words = set(memory_row["query"].lower().split())
                    overlap = len(query_words.intersection(memory_words))
                    if overlap > 0:
                        score += 0.4 * (overlap / max(len(query_words), len(memory_words)))
                
                # Only include results with a positive score
                if score > 0.0:
                    results.append({
                        "memory_id": memory_row.get("memory_id", ""),
                        "query": memory_row.get("query", ""),
                        "entity": memory_row.get("entity", ""),
                        "context": memory_row.get("context", {}), # Ensure it's a dict if possible
                        "score": score,
                        "memory_type": "episodic",
                        "search_method": "symbolic"
                    })
                    
        elif self.memory_type == "personality":
            # Search personality memory by personality_type and mbti_type
            for index, memory_row in collection.iterrows():
                score = 0.0
                
                # Personality type matching
                if "personality_type" in memory_row and pd.notna(memory_row["personality_type"]):
                    if memory_row["personality_type"].lower() in query.lower():
                        score += 0.5
                    elif query.lower() in memory_row["personality_type"].lower():
                        score += 0.4
                
                if "mbti_type" in memory_row and pd.notna(memory_row["mbti_type"]):
                    if memory_row["mbti_type"].lower() in query.lower():
                        score += 0.5
                
                if "cognitive_style" in memory_row and pd.notna(memory_row["cognitive_style"]):
                    if memory_row["cognitive_style"].lower() in query.lower():
                        score += 0.4
                
                # Activation contexts (assuming it's a list of strings or string representation of list)
                if "activation_contexts" in memory_row and pd.notna(memory_row["activation_contexts"]):
                    activation_contexts = memory_row["activation_contexts"]
                    # Handle if activation_contexts is stored as a string representation of a list
                    if isinstance(activation_contexts, str):
                        try:
                            activation_contexts = json.loads(activation_contexts.replace("'", "\"")) # Basic string to list
                        except json.JSONDecodeError:
                            activation_contexts = [activation_contexts] # Treat as single item list if parse fails
                    
                    if isinstance(activation_contexts, list):
                        for context_item in activation_contexts:
                            if isinstance(context_item, str) and context_item.lower() in query.lower():
                                score += 0.5
                                break # Match found in contexts
                
                if score > 0.0:
                    results.append({
                        "mode_id": memory_row.get("mode_id", ""),
                        "mode_name": memory_row.get("mode_name", ""),
                        "personality_type": memory_row.get("personality_type", ""),
                        "cognitive_style": memory_row.get("cognitive_style", ""),
                        "mbti_type": memory_row.get("mbti_type", ""),
                        "mode_description": memory_row.get("mode_description", ""),
                        "score": score,
                        "memory_type": "personality",
                        "search_method": "symbolic"
                    })
        
        # Sort results by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def _semantic_search(self, query, collection, embedding_model=None):
        """Perform semantic search on the provided collection.
        
        Args:
            query: Query string
            collection: Collection to search in
            embedding_model: Model to use for semantic search
            
        Returns:
            List[Dict[str, Any]]: Search results with scores
        """
        results = []
        query_embedding = self.generator.get_embedding(query, model=embedding_model)
        
        if self.memory_type == "episodic":
            # For episodic memories, compare against query and entity
            for index, memory_row in collection.iterrows():
                comparison_text = ""
                if pd.notna(memory_row.get("query")): comparison_text += str(memory_row["query"]) + " "
                if pd.notna(memory_row.get("entity")): comparison_text += str(memory_row["entity"]) + " "
                if pd.notna(memory_row.get("transformed_query")): comparison_text += str(memory_row["transformed_query"]) + " "
                
                if not comparison_text.strip(): continue
                
                memory_embedding = self.generator.get_embedding(comparison_text.strip(), model=embedding_model)
                if query_embedding is not None and memory_embedding is not None:
                    score = 1.0 - cosine(query_embedding, memory_embedding)
                    if score > 0.2:
                        results.append({
                            "memory_id": memory_row.get("memory_id", ""),
                            "query": memory_row.get("query", ""),
                            "entity": memory_row.get("entity", ""),
                            "context": memory_row.get("context", {}),
                            "score": score,
                            "memory_type": "episodic",
                            "search_method": "semantic"
                        })
                    
        elif self.memory_type == "personality":
            for index, memory_row in collection.iterrows():
                comparison_text = ""
                if pd.notna(memory_row.get("mode_name")): comparison_text += str(memory_row["mode_name"]) + " "
                if pd.notna(memory_row.get("personality_type")): comparison_text += str(memory_row["personality_type"]) + " "
                if pd.notna(memory_row.get("cognitive_style")): comparison_text += str(memory_row["cognitive_style"]) + " "
                if pd.notna(memory_row.get("mbti_type")): comparison_text += str(memory_row["mbti_type"]) + " "
                if pd.notna(memory_row.get("mode_description")): comparison_text += str(memory_row["mode_description"]) + " "
                
                if not comparison_text.strip(): continue

                memory_embedding = self.generator.get_embedding(comparison_text.strip(), model=embedding_model)
                if query_embedding is not None and memory_embedding is not None:
                    score = 1.0 - cosine(query_embedding, memory_embedding)
                    if score > 0.2: 
                        results.append({
                            "mode_id": memory_row.get("mode_id", ""),
                            "mode_name": memory_row.get("mode_name", ""),
                            "personality_type": memory_row.get("personality_type", ""),
                            "cognitive_style": memory_row.get("cognitive_style", ""),
                            "mbti_type": memory_row.get("mbti_type", ""),
                            "mode_description": memory_row.get("mode_description", ""),
                            "score": score,
                            "memory_type": "personality",
                            "search_method": "semantic"
                        })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def _hybrid_search(self, symbolic_results, semantic_results, hybrid_weight=0.7):
        """Combine results from symbolic and semantic search.
        
        Args:
            symbolic_results: Results from symbolic search
            semantic_results: Results from semantic search
            hybrid_weight: Weight for hybrid search (0=symbolic only, 1=semantic only)
            
        Returns:
            List[Dict[str, Any]]: Combined search results
        """
        # Early return if one method has no results
        if not symbolic_results:
            return semantic_results
        if not semantic_results:
            return symbolic_results
        
        # Map for combining scores
        results_map = {}
        
        # Process symbolic results
        for result in symbolic_results:
            id_field = "memory_id" if self.memory_type == "episodic" else "mode_id"
            key = result.get(id_field, "")
            if key:
                results_map[key] = {
                    "item": result,
                    "symbolic_score": result["score"],
                    "semantic_score": 0.0
                }
        
        # Process semantic results
        for result in semantic_results:
            id_field = "memory_id" if self.memory_type == "episodic" else "mode_id"
            key = result.get(id_field, "")
            if key:
                if key in results_map:
                    # Update existing entry
                    results_map[key]["semantic_score"] = result["score"]
                else:
                    # Add new entry
                    results_map[key] = {
                        "item": result,
                        "symbolic_score": 0.0,
                        "semantic_score": result["score"]
                    }
        
        # Calculate combined scores
        combined_results = []
        for key, data in results_map.items():
            item = data["item"].copy()
            # Calculate weighted score
            semantic_contribution = data["semantic_score"] * hybrid_weight
            symbolic_contribution = data["symbolic_score"] * (1.0 - hybrid_weight)
            combined_score = semantic_contribution + symbolic_contribution
            
            item["score"] = combined_score
            item["original_semantic_score"] = data["semantic_score"]
            item["original_symbolic_score"] = data["symbolic_score"]
            item["search_method"] = "hybrid"
            
            combined_results.append(item)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results
    
    def _apply_limits(self, results: List[Dict[str, Any]], top_k: Optional[int] = None, top_p: Optional[float] = None):
        """Apply limits to retrieval results based on top_k and top_p.
        
        Args:
            results: List of retrieval results (dictionaries with 'score').
            top_k: Number of top results to return.
            top_p: Minimum score threshold for results.
            
        Returns:
            List[Dict[str, Any]]: Limited retrieval results.
        """
        # Sort results by score in descending order if not already sorted
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top_p threshold if specified
        if top_p is not None and 0 <= top_p <= 1.0:
            results = [r for r in results if r["score"] >= top_p]
        
        # Apply top_k limit if specified
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return results
    
    def save_memory(self, new_memory):
        """Save a new memory to the database.
        
        Args:
            new_memory: Memory data to save
            
        Returns:
            bool: Success status
        """
        try:
            collection_name = f"{self.memory_type}_memory"
            
            if self.memory_type == "episodic":
                # Ensure required fields are present
                if "memory_id" not in new_memory:
                    new_memory["memory_id"] = str(uuid.uuid4())
                if "query" not in new_memory:
                    raise ValueError("Query field is required for episodic memory")
                if "entity" not in new_memory:
                    raise ValueError("Entity field is required for episodic memory")
                if "context" not in new_memory:
                    raise ValueError("Context field is required for episodic memory")
                
            elif self.memory_type == "personality":
                # Ensure required fields are present
                if "mode_id" not in new_memory:
                    new_memory["mode_id"] = str(uuid.uuid4())
                if "mode_name" not in new_memory:
                    raise ValueError("Mode name is required for personality memory")
                if "personality_type" not in new_memory:
                    raise ValueError("Personality type is required for personality memory")
                if "cognitive_style" not in new_memory:
                    raise ValueError("Cognitive style is required for personality memory")
                if "mbti_type" not in new_memory:
                    raise ValueError("MBTI type is required for personality memory")
            
            # Add to database
            if collection_name not in self.memory_db:
                self.memory_db[collection_name] = []
            
            self.memory_db[collection_name].append(new_memory)
            
            id_field = "memory_id" if self.memory_type == "episodic" else "mode_id"
            logger.debug(f"Added {self.memory_type} memory with ID: {new_memory[id_field]}")
            
            # Save to file if memory_df was loaded from a file path
            if self.memory_db_path and isinstance(self.memory_df, pd.DataFrame):
                # The collection (e.g., episodic_memory) is now the entire DataFrame
                # So, we save self.memory_df directly to self.memory_db_path
                self.memory_df.to_parquet(self.memory_db_path)
                logger.debug(f"Saved memory database to {self.memory_db_path}")
            elif isinstance(self.memory_df, pd.DataFrame): # Loaded from DataFrame, no path to save to by default
                 logger.debug("Memory updated in DataFrame, but no file path associated for saving.")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memory: {str(e)}")
            logger.debug(f"Memory save error details: {traceback.format_exc()}")
            return False