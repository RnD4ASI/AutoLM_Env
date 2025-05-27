from src.pipeline.shared.logging import get_logger
from typing import List, Dict, Any, Tuple, Optional, Union
import math
import numpy as np
import pandas as pd
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame
import spacy
from src.pipeline.processing.generator import Generator
from src.pipeline.shared.utility import DataUtility, StatisticsUtility, AIUtility

logger = get_logger(__name__)

class Evaluator:
    """
    Evaluates generated text against reference text using various NLP metrics 
    such as BLEU, ROUGE, METEOR, and BERTScore.
    """

    def __init__(self, generator = None):
        """Initialize Evaluator with required models and utilities."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.rouge = Rouge()
            self.generator = generator if generator else Generator()
            self.datautility = DataUtility()
            self.statsutility = StatisticsUtility()
            self.aiutility = AIUtility()
            self.ngram_weights = {
                1: (1.0,),
                2: (0.5, 0.5),
                4: (0.25, 0.25, 0.25, 0.25)
            }
            logger.debug("Evaluator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Evaluator: {e}")
            raise

    def get_tokenisation(self, text: str, granularity: str = 'word') -> List[str]:
        """Tokenizes the input text into either sentences or words.

        Parameters:
            text (str): Text to tokenize.
            granularity (str): Level of tokenization ('word' or 'sentence').

        Returns:
            List[str]: List of tokens in lowercase.
        """
        try:
            doc = self.nlp(text)
            if granularity == 'sentence':
                return [str(sent).lower() for sent in doc.sents]
            return [str(token).lower() for token in doc if not token.is_space]
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise

    def _generate_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Generate n-grams from a list of tokens.

        Parameters:
            tokens (List[str]): List of tokens.
            n (int): Size of n-grams.

        Returns:
            List[Tuple[str, ...]]: List of n-grams.
        """
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def get_bleu(self, reference_text: str, generated_text: str, n: int = 4, 
                 mode: str = 'calculation') -> Union[float, Dict[str, Any]]:
        """Compute BLEU score for n-gram precision evaluation.

        Parameters:
            reference_text (str): Reference text to compare against.
            generated_text (str): Generated text to evaluate.
            n (int): Maximum n-gram size.
            mode (str): 'calculation' returns float score, 'reporting' returns detailed metrics.

        Returns:
            Union[float, Dict[str, Any]]: BLEU score or detailed metrics dictionary.
        """
        try:
            # Tokenize texts
            gen_tokens = self.get_tokenisation(generated_text)
            ref_tokens = self.get_tokenisation(reference_text)

            # Calculate n-gram precisions
            precisions = []
            precision_details = {}
            for i in range(1, n + 1):
                gen_ngrams = self._generate_ngrams(gen_tokens, i)
                ref_ngrams = self._generate_ngrams(ref_tokens, i)
                
                if not gen_ngrams:
                    precisions.append(0.0)
                    precision_details[f'{i}-gram'] = 0.0
                    continue
                
                # Calculate clipped precision
                matches = sum(1 for gram in gen_ngrams if gram in ref_ngrams)
                precision = matches / len(gen_ngrams)
                precisions.append(precision)
                precision_details[f'{i}-gram'] = precision

            # Calculate brevity penalty
            bp = 1.0
            if len(gen_tokens) < len(ref_tokens):
                bp = math.exp(1 - len(ref_tokens) / len(gen_tokens))

            # Calculate final BLEU score
            if all(p == 0 for p in precisions):
                score = 0.0
            else:
                weights = self.ngram_weights.get(n, tuple(1/n for _ in range(n)))
                score = bp * math.exp(sum(w * math.log(p) if p > 0 else float('-inf')
                                        for w, p in zip(weights, precisions)))

            if mode == 'calculation':
                return score
            else:
                return {
                    'score': score,
                    'brevity_penalty': bp,
                    'precisions': precision_details,
                    'reference_length': len(ref_tokens),
                    'generated_length': len(gen_tokens)
                }

        except Exception as e:
            logger.error(f"BLEU calculation error: {e}")
            raise

    def get_rouge(self, reference_text: str, generated_text: str, 
                 rouge_types: Optional[List[str]] = None,
                 mode: str = 'calculation') -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Calculate ROUGE scores for n-gram overlap evaluation.

        Parameters:
            reference_text (str): Reference text to compare against.
            generated_text (str): Generated text to evaluate.
            rouge_types (Optional[List[str]]): ROUGE types to calculate.
            mode (str): 'calculation' returns scores only, 'reporting' returns detailed metrics.

        Returns:
            Union[Dict[str, float], Dict[str, Dict[str, float]]]: ROUGE scores.
        """
        try:
            if rouge_types is None:
                rouge_types = ['rouge-1', 'rouge-2', 'rouge-l']

            scores = self.rouge.get_scores(generated_text, reference_text)[0]
            
            if mode == 'calculation':
                return {type_: scores[type_]['f'] for type_ in rouge_types if type_ in scores}
            else:
                return {type_: scores[type_] for type_ in rouge_types if type_ in scores}
                
        except Exception as e:
            logger.error(f"ROUGE calculation error: {e}")
            raise

    def get_bertscore(self, reference_text: str, generated_text: str, 
                     granularity: str = 'document',
                     model: str = "text-embedding-ada-002", 
                     mode: str = 'calculation') -> Union[float, Dict[str, Any]]:
        """Calculate BERTScore between generated and reference texts.

        Parameters:
            reference_text (str): Reference text to compare against.
            generated_text (str): Generated text to evaluate.
            granularity (str): Evaluation granularity ('document' or 'sentence').
            mode (str): 'calculation' returns F1 score, 'reporting' returns detailed metrics.

        Returns:
            Union[float, Dict[str, Any]]: BERTScore metric(s).
            For calculation mode: returns F1 score
            For reporting mode: returns dict with precision, recall, F1, granularity, and sentence count
        """
        try:
            if granularity == 'sentence':
                # Split into sentences
                gen_sents = self.get_tokenisation(generated_text, 'sentence')
                ref_sents = self.get_tokenisation(reference_text, 'sentence')
                
                # Get embeddings for each sentence
                gen_embs = [self.generator.get_embeddings(sent, model) for sent in gen_sents]
                ref_embs = [self.generator.get_embeddings(sent, model) for sent in ref_sents]
                
                # Create similarity matrix
                sim_matrix = np.zeros((len(gen_sents), len(ref_sents)))
                for i, gen_emb in enumerate(gen_embs):
                    for j, ref_emb in enumerate(ref_embs):
                        sim_matrix[i, j] = cosine_similarity(
                            np.array(gen_emb).reshape(1, -1),
                            np.array(ref_emb).reshape(1, -1)
                        )[0][0]
                
                # Calculate precision (max similarity for each generated sentence)
                precision = np.mean(np.max(sim_matrix, axis=1))
                
                # Calculate recall (max similarity for each reference sentence)
                recall = np.mean(np.max(sim_matrix, axis=0))
                
                # Calculate F1
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                num_sentences = len(gen_sents)
            else:  # document level
                # Get document embeddings
                gen_emb = self.generator.get_embeddings(generated_text, model)
                ref_emb = self.generator.get_embeddings(reference_text, model)
                
                # Calculate similarity
                similarity = cosine_similarity(
                    np.array(gen_emb).reshape(1, -1),
                    np.array(ref_emb).reshape(1, -1)
                )[0][0]
                
                precision = similarity
                recall = similarity
                f1 = similarity
                num_sentences = len(self.get_tokenisation(generated_text, 'sentence'))

            if mode == 'calculation':
                return f1
            else:
                return {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'granularity': granularity,
                    'num_sentences': num_sentences
                }

        except Exception as e:
            logger.error(f"BERTScore calculation error: {e}")
            raise

    def get_meteor(self, reference_text: str, generated_text: str,
                  mode: str = 'calculation') -> Union[float, Dict[str, float]]:
        """Compute METEOR score for text evaluation.

        Parameters:
            reference_text (str): Reference text to compare against.
            generated_text (str): Generated text to evaluate.
            mode (str): 'calculation' returns score only, 'reporting' returns detailed metrics.

        Returns:
            Union[float, Dict[str, float]]: METEOR score or detailed metrics.
        """
        try:
            # Tokenize texts
            gen_tokens = self.get_tokenisation(generated_text)
            ref_tokens = self.get_tokenisation(reference_text)

            # Find exact matches
            matches = set(gen_tokens) & set(ref_tokens)
            
            if not matches:
                return 0.0 if mode == 'calculation' else {
                    'score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f_mean': 0.0,
                    'penalty': 0.0
                }

            # Calculate precision and recall
            precision = len(matches) / len(gen_tokens)
            recall = len(matches) / len(ref_tokens)

            # Calculate F-mean (harmonic mean of precision and recall)
            if precision + recall == 0:
                return 0.0 if mode == 'calculation' else {
                    'score': 0.0,
                    'precision': precision,
                    'recall': recall,
                    'f_mean': 0.0,
                    'penalty': 0.0
                }
                
            fmean = 2 * (precision * recall) / (precision + recall)

            # Calculate penalty
            chunks = len(self._get_chunks(gen_tokens, ref_tokens))
            penalty = 0.5 * (chunks / len(matches)) ** 3

            score = fmean * (1 - penalty)

            if mode == 'calculation':
                return score
            else:
                return {
                    'score': score,
                    'precision': precision,
                    'recall': recall,
                    'f_mean': fmean,
                    'penalty': penalty
                }

        except Exception as e:
            logger.error(f"METEOR calculation error: {e}")
            raise

    def _get_chunks(self, gen_tokens: List[str], ref_tokens: List[str]) -> List[List[str]]:
        """Helper method to find contiguous sequences of matches for METEOR."""
        chunks = []
        current_chunk = []
        
        for i, token in enumerate(gen_tokens):
            if token in ref_tokens:
                if not current_chunk:
                    current_chunk = [token]
                else:
                    current_chunk.append(token)
            elif current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def evaluate_all(self, reference_text: str, generated_text: str,
                    model = "Jina-embeddings-v3",
                    metrics: Optional[List[str]] = None) -> DataFrame:
        """Run all evaluation metrics and return combined results in a DataFrame.

        Parameters:
            reference_text (str): Reference text to compare against.
            generated_text (str): Generated text to evaluate.
            metrics (Optional[List[str]]): List of metrics to compute. If None, computes all.
                Available metrics: ['bleu', 'rouge', 'bertscore', 'meteor']

        Returns:
            DataFrame: Combined evaluation metrics with the following columns:
                - metric: Name of the metric (e.g., 'bleu', 'rouge-1', 'bertscore')
                - type: Type of score (e.g., 'precision', 'recall', 'f1')
                - value: Numerical score value
                - details: Additional metric-specific information
        """
        try:
            # Input validation
            if not reference_text or not generated_text:
                raise ValueError("Reference and generated texts cannot be empty")

            available_metrics = {'bleu', 'rouge', 'bertscore', 'meteor'}
            metrics = metrics or list(available_metrics)
            invalid_metrics = set(metrics) - available_metrics
            if invalid_metrics:
                raise ValueError(f"Invalid metrics specified: {invalid_metrics}")

            results = []

            # Compute BLEU if requested
            if 'bleu' in metrics:
                logger.debug("Computing BLEU metric...")
                bleu_result = self.get_bleu(
                    reference_text, generated_text,
                    n=4,  # Use default n=4 for consistency
                    mode='reporting'
                )
                # Add n-gram precisions
                for n, precision in bleu_result['precisions'].items():
                    results.append({
                        'metric': 'bleu',
                        'type': f'precision-{n}',
                        'value': precision,
                        'details': None
                    })
                # Add overall BLEU score
                results.append({
                    'metric': 'bleu',
                    'type': 'score',
                    'value': bleu_result['score'],
                    'details': f"bp={bleu_result['brevity_penalty']:.3f}"
                })

            # Compute ROUGE if requested
            if 'rouge' in metrics:
                logger.debug("Computing ROUGE metric...")
                # Define all ROUGE types to compute
                rouge_types = ['rouge-1', 'rouge-2', 'rouge-l']
                rouge_result = self.get_rouge(
                    reference_text, generated_text,
                    rouge_types=rouge_types,
                    mode='reporting'
                )
                
                # Add results for each ROUGE type and score type
                for rouge_type, scores in rouge_result.items():
                    # Get n-gram info for ROUGE-N metrics
                    n_gram = rouge_type.split('-')[1] if rouge_type != 'rouge-l' else 'L'
                    
                    for score_type, value in scores.items():
                        details = None
                        if score_type == 'f':
                            score_type = 'f1'  # Rename for consistency
                        if n_gram.isdigit():
                            details = f"{n_gram}-gram overlap"
                        elif n_gram == 'L':
                            details = "Longest common subsequence"
                            
                        results.append({
                            'metric': rouge_type,
                            'type': score_type,
                            'value': value,
                            'details': details
                        })

            # Compute BERTScore if requested
            if 'bertscore' in metrics:
                logger.debug("Computing BERTScore metric...")
                bertscore_result = self.get_bertscore(
                    reference_text, generated_text, model=model,
                    mode='reporting'
                )
                for score_type in ['precision', 'recall', 'f1']:
                    results.append({
                        'metric': 'bertscore',
                        'type': score_type,
                        'value': bertscore_result[score_type],
                        'details': f"granularity={bertscore_result['granularity']}"
                    })

            # Compute METEOR if requested
            if 'meteor' in metrics:
                logger.debug("Computing METEOR metric...")
                meteor_result = self.get_meteor(
                    reference_text, generated_text,
                    mode='reporting'
                )
                for score_type, value in meteor_result.items():
                    results.append({
                        'metric': 'meteor',
                        'type': score_type,
                        'value': value,
                        'details': None
                    })

            # Create DataFrame and set column types
            df = pd.DataFrame(results)
            df['value'] = pd.to_numeric(df['value'])
            
            # Sort the DataFrame for better readability
            df = df.sort_values(['metric', 'type']).reset_index(drop=True)
            
            # Add metadata as DataFrame attributes
            df.attrs['metadata'] = {
                'text_lengths': {
                    'reference': len(self.get_tokenisation(reference_text)),
                    'generated': len(self.get_tokenisation(generated_text))
                },
                'sentence_counts': {
                    'reference': len(self.get_tokenisation(reference_text, 'sentence')),
                    'generated': len(self.get_tokenisation(generated_text, 'sentence'))
                },
                'metrics_computed': {
                    'rouge_types': rouge_types if 'rouge' in metrics else [],
                    'all_metrics': sorted(list(set(df['metric'])))
                }
            }

            logger.info(f"Successfully computed {len(df['metric'].unique())} evaluation metrics")
            return df
            
        except Exception as e:
            logger.error(f"Error in evaluate_all: {e}")
            raise

# TO DO:
class PlanEvaluator:
    def __init__(self, generator: Generator = None, evaluator: Evaluator = None):
        self.generator = generator
        self.evaluator = evaluator
    
    def evaluate_plan(self, plan: str) -> Dict[str, float]:
        return {}


# TO DO:
class RetrievalEvaluator:
    """
    Evaluates retrieval performance using metrics from evaluator.py.
    Responsible for assessing the quality of retrieved documents.
    """
    
    def __init__(self, generator: Generator = None, evaluator: Evaluator = None):
        """
        Initialize the EvaluationProcessor.
        
        Args:
            generator: Optional Generator instance for text generation and embeddings
        """
        # Initialize evaluator components
        self.evaluator = Evaluator(generator)
        self.retrieval_evaluator = RetrievalEvaluator(self.evaluator)
        
        # Evaluation configuration
        self.metrics = {
            'precision': True,
            'recall': True,
            'f1_score': True,
            'ndcg': True,
            'map': True
        }
        
        logger.debug("EvaluationProcessor initialized")
    
    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                                  relevant_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the quality of retrieved documents against known relevant documents.
        
        Args:
            query: Original query string
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Known relevant documents (ground truth)
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        try:
            # Extract content from retrieved and relevant documents
            retrieved_content = "\n\n".join([doc['content'] for doc in retrieved_docs])
            relevant_content = "\n\n".join([doc['content'] for doc in relevant_docs])
            
            # Use the RetrievalEvaluator to calculate metrics
            metrics = self.retrieval_evaluator.evaluate_retrieval(relevant_content, retrieved_content)
            
            # Add additional metrics specific to retrieval evaluation
            metrics.update(self._calculate_rank_metrics(retrieved_docs, relevant_docs))
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating retrieval quality: {e}")
            return {}
    
    def evaluate_query_performance(self, original_query: str, processed_query: str, 
                                 retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate how query processing affected retrieval performance.
        
        Args:
            original_query: Original user query
            processed_query: Processed query after rewriting/expansion
            retrieved_docs: Documents retrieved using the processed query
            
        Returns:
            Dict[str, Any]: Performance metrics and analysis
        """
        try:
            # Calculate query transformation metrics
            query_similarity = self.evaluator.get_bertscore(original_query, processed_query)
            
            # Analyze term overlap between query and retrieved documents
            term_overlap = self._analyze_term_overlap(processed_query, retrieved_docs)
            
            return {
                'query_similarity': query_similarity,
                'term_overlap': term_overlap,
                'query_effectiveness': self._estimate_query_effectiveness(term_overlap)
            }
        except Exception as e:
            logger.error(f"Error evaluating query performance: {e}")
            return {}
    
    def _calculate_rank_metrics(self, retrieved_docs: List[Dict[str, Any]], 
                              relevant_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate ranking-based metrics like NDCG and MAP.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Known relevant documents (ground truth)
            
        Returns:
            Dict[str, float]: Dictionary of ranking metrics
        """
        # Extract IDs of relevant documents for comparison
        relevant_ids = set(doc['id'] for doc in relevant_docs)
        
        # Create relevance judgments (1 for relevant, 0 for non-relevant)
        relevance = [1 if doc['id'] in relevant_ids else 0 for doc in retrieved_docs]
        
        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        ndcg = self._calculate_ndcg(relevance)
        
        # Calculate MAP (Mean Average Precision)
        map_score = self._calculate_map(relevance)
        
        return {
            'ndcg': ndcg,
            'map': map_score
        }
    
    def _calculate_ndcg(self, relevance: List[int], k: int = 10) -> float:
        """
        Calculate NDCG (Normalized Discounted Cumulative Gain).
        
        Args:
            relevance: List of relevance judgments (1 for relevant, 0 for non-relevant)
            k: Cutoff for calculation
            
        Returns:
            float: NDCG score
        """
        # Limit to top k results
        rel = relevance[:k]
        
        # Calculate DCG
        dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
        
        # Calculate ideal DCG (IDCG)
        ideal_rel = sorted(rel, reverse=True)
        idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rel))
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_map(self, relevance: List[int]) -> float:
        """
        Calculate MAP (Mean Average Precision).
        
        Args:
            relevance: List of relevance judgments (1 for relevant, 0 for non-relevant)
            
        Returns:
            float: MAP score
        """
        # Calculate precision at each relevant position
        precisions = []
        relevant_count = 0
        
        for i, rel in enumerate(relevance):
            if rel == 1:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        # Return MAP
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def _analyze_term_overlap(self, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze term overlap between query and retrieved documents.
        
        Args:
            query: Query string
            docs: Retrieved documents
            
        Returns:
            Dict[str, Any]: Term overlap analysis
        """
        # Tokenize query
        query_tokens = set(self.evaluator.get_tokenisation(query))
        
        # Calculate overlap for each document
        overlaps = []
        for doc in docs:
            doc_tokens = set(self.evaluator.get_tokenisation(doc['content']))
            overlap = len(query_tokens & doc_tokens) / len(query_tokens) if query_tokens else 0
            overlaps.append(overlap)
        
        return {
            'mean_overlap': sum(overlaps) / len(overlaps) if overlaps else 0,
            'max_overlap': max(overlaps) if overlaps else 0,
            'min_overlap': min(overlaps) if overlaps else 0
        }
    
    def _estimate_query_effectiveness(self, term_overlap: Dict[str, float]) -> float:
        """
        Estimate query effectiveness based on term overlap metrics.
        
        Args:
            term_overlap: Term overlap analysis dictionary
            
        Returns:
            float: Estimated query effectiveness score (0-1)
        """
        # Simple weighted combination of overlap metrics
        return 0.5 * term_overlap['mean_overlap'] + 0.3 * term_overlap['max_overlap'] + 0.2 * term_overlap['min_overlap']
