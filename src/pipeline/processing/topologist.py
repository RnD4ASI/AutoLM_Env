"""
Module for various topologies (heuristic) using generator and meta-generator.
"""

import json
from typing import List, Dict, Any, Optional, Union, Tuple, Literal
import numpy as np
import random
from src.pipeline.shared.logging import get_logger
from src.pipeline.processing.generator import MetaGenerator, Generator
from src.pipeline.processing.evaluator import Evaluator
from src.pipeline.shared.utility import AIUtility

logger = get_logger(__name__)

class PromptTopology:
    """
    Implements different patterns for executing prompts using meta-prompts.
    """
    
    def __init__(self, generator = None):
        """Initialize with required components."""
        self.generator = generator if generator else Generator()
        self.metagenerator = MetaGenerator(generator = self.generator)
        self.evaluator = Evaluator()
        self.aiutility = AIUtility()
        # self.template_adopter = TemplateAdopter()
        logger.debug("PromptTopology initialized")

    def prompt_disambiguation(self,
        task_prompt: str,
        prompt_id: int,
        model: Optional[str] = "Qwen2.5-1.5B",
        temperature: Optional[float] = 0.7,
        return_full_response: Optional[bool] = False,
        **kwargs
        ) -> Union[Tuple[List[str], List[str]], Tuple[str, str]]:
        """
        Topology 1: Disambiguate prompt to improve clarity.
        
        Args:
            task_prompt: Task prompt
            prompt_id: Prompt ID for tracking
            model: Model to use for generation
            temperature: Temperature for generation
            return_full_response: If True, returns a dictionary with response and metrics
            **kwargs: Additional arguments for get_meta_generation
            
        Returns:
            If return_full_response is False:
                Final response as a string
            If return_full_response is True:
                Dictionary containing response and additional metrics
        """
        # Step 1: Generate disambiguated prompt
        try:
            disambiguation = self.metagenerator.get_meta_generation(
                application="metaprompt",
                category="evaluation",
                action="disambiguate",
                prompt_id=prompt_id,
                task_prompt=task_prompt,
                model=model,
                temperature=0.25,  # Lower temperature for analysis
                return_full_response=False
            )
            logger.debug(f"Disambiguation result: {disambiguation}")
        except Exception as e:
            logger.error(f"Error disambiguating prompt: {e}")
            return None

        # Step 2: Rewrite based on disambiguation
        try:
            rewritten_prompt = self.metagenerator.get_meta_generation(
                application="metaprompt",
                category="manipulation",
                action="rewrite",
                prompt_id=prompt_id,
                task_prompt=task_prompt,
                feedback=disambiguation,
                model=model,
                temperature=0.5,  # Moderate temperature for rewriting
                return_full_response=False
            )
            logger.debug(f"Rewritten prompt: {rewritten_prompt}")
        except Exception as e:
            logger.error(f"Error rewriting prompt: {e}")
            return None
        
        # Step 3: Execute rewritten prompt
        try:
            response_pre = self.generator.get_completion(
                        prompt_id=prompt_id,
                        prompt=task_prompt,
                        model=model,
                        temperature=temperature,
                        return_full_response=False)
            logger.debug(f"Response before disambiguation: {response_pre}")
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return None
        
        try:
            response_post = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=rewritten_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False)
            logger.debug(f"Response after disambiguation: {response_post}")
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return None

        if return_full_response:
            return ([task_prompt,rewritten_prompt], [response_pre, response_post])
        else:
            return (rewritten_prompt, response_post)


    def prompt_genetic_algorithm(self,
        task_prompt: str,
        prompt_id: int,
        num_variations: int = 5,
        num_evolution: int = 3,
        model: Optional[str] = "Qwen2.5-1.5B",
        temperature: Optional[float] = 0.75,
        variation_temperature: Optional[float] = 1.0,
        return_full_response: Optional[bool] = False,
        **kwargs
        ) -> Union[Tuple[List[str], List[str]], Tuple[str, str]]:
        """
        Topology 2: Evolve prompt through variations, crossover, mutation, and ranking.
        
        Args:
            task_prompt: Task prompt
            prompt_id: Prompt ID for tracking
            num_variations: Number of initial variations to generate (default: 5)
            num_evolution: Number of evolution iterations (default: 3)
            model: Model to use for generation
            temperature: Temperature for final response generation
            variation_temperature: Temperature for variation generation
            return_full_response: If True, returns a dictionary with response and metrics
            **kwargs: Additional arguments for get_completion
            
        Returns:
            If return_full_response is False:
                Final response as a string
            If return_full_response is True:
                Dictionary containing response and additional metrics
        """
        # Step 1: Generate initial variations
        try:
            prompt_variations = []
            for i in range(num_variations - 1):  # Generate n-1 variations
                variation = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="manipulation",
                    action="rephrase",
                    prompt_id=prompt_id,
                    task_prompt=task_prompt,
                    feedback="not available",
                    model=model,
                    temperature=variation_temperature,  # High temperature for diversity
                    return_full_response=False
                )
                prompt_variations.append(variation)
                logger.debug(f"Generated variation {i+1}: {variation}")
            
            # Include original task prompt
            prompt_variations.append(task_prompt)
            logger.debug(f"Generated {len(prompt_variations)} initial variations")
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return task_prompt
        
        # Step 2: Execute initial variations to get responses
        responses = []
        for i, variation in enumerate(prompt_variations):
            try:
                response = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=variation,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                responses.append(response)
                logger.debug(f"Generated response for variation {i+1}")
            except Exception as e:
                logger.error(f"Error generating response for variation {i+1}: {e}")
                responses.append("Error generating response")
        
        # Store the mapping between variations and responses
        variation_pairs = list(zip(prompt_variations, responses))
        
        # Evolve the prompt through multiple iterations
        for iteration in range(num_evolution):

            # Step 3: Rank responses
            # Convert responses from a list to a stringed dictionary
            tmp = []
            for index, content in enumerate(responses):
                tmp.append({"response_index": index, "response": content})
            responses_dict = {"responses": tmp}
            
            try:       
                ranking_prompt = self.metagenerator.get_meta_generation(
                    application="metaresponse",
                    category="evaluation",
                    action="rank",
                    prompt_id=prompt_id,
                    task_prompt=task_prompt,
                    responses=responses_dict,
                    model=model,
                    temperature=0.3,  # Low temperature for consistent ranking
                    return_full_response=False
                )
                
                # Parse the ranking result
                try:
                    ranking_result = self.aiutility.format_json_response(ranking_prompt)
                    # Expect format like: {"rankings": [{"response_index": 0, "rank": 3}, ...]}
                    ranked_indices = sorted(
                        range(len(ranking_result["rankings"])), 
                        key=lambda i: ranking_result["rankings"][i]["rank"]
                    )
                    logger.debug(f"Iteration {iteration+1} ranking: {ranking_result}")
                except Exception as e:
                    logger.error(f"Error parsing ranking result: {e}")
                    # If parsing fails, use random ranking
                    ranked_indices = list(range(len(responses)))
                    random.shuffle(ranked_indices)
                
                # Step 4: Select top 3 variations based on ranking
                top_variations = [variation_pairs[i][0] for i in ranked_indices[:3]]
                logger.debug(f"Selected top {len(top_variations)} variations for crossover")
                
                # Step 5: Apply crossover on top variations
                crossovered = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="manipulation",
                    action="crossover",
                    prompt_id=prompt_id,
                    task_prompt=top_variations,
                    model=model,
                    temperature=0.5,  # Moderate temperature for diversity
                    return_full_response=False
                )
                logger.debug(f"Iteration {iteration+1} crossovered prompt: {crossovered}")
                
                # Step 6: Apply mutation to crossovered prompt
                mutated = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="manipulation",
                    action="mutate",
                    prompt_id=prompt_id,
                    task_prompt=crossovered,
                    model=model,
                    temperature=0.75,
                    return_full_response=False
                )
                logger.debug(f"Iteration {iteration+1} mutated prompt: {mutated}")
                
                # Step 7: Execute the mutated prompt
                mutated_response = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=mutated,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                logger.debug(f"Generated response for mutated prompt in iteration {iteration+1}")
                
                # Step 8: Update variations and responses for next iteration
                # Keep top 4 variations and add the new mutated variation
                top_4_pairs = [variation_pairs[i] for i in ranked_indices[:4]]
                variation_pairs = top_4_pairs + [(mutated, mutated_response)]
                
                # Update variations and responses lists for next iteration
                prompt_variations = [pair[0] for pair in variation_pairs]
                responses = [pair[1] for pair in variation_pairs]
                
            except Exception as e:
                logger.error(f"Error in evolution iteration {iteration+1}: {e}")
                break
        
        # After all iterations, find the best response
        try:
            # Final ranking
            tmp = []
            for index, content in enumerate(responses):
                tmp.append({"response_index": index, "response": content})
            responses_dict = {"responses": tmp}
            
            final_ranking_prompt = self.metagenerator.get_meta_generation(
                application="metaresponse",
                category="evaluation",
                action="rank",
                prompt_id=prompt_id,
                task_prompt=task_prompt,
                responses=responses_dict,
                model=model,
                temperature=0.3,
                return_full_response=False
            )
            
            # Parse the final ranking
            try:
                final_ranking = self.aiutility.format_json_response(final_ranking_prompt)
                ranked_indices = sorted(
                    range(len(final_ranking["rankings"])), 
                    key=lambda i: final_ranking["rankings"][i]["rank"]
                )
                # Get the best prompt and response
                best_prompt = prompt_variations[ranked_indices[0]]
                best_response = responses[ranked_indices[0]]
                logger.debug(f"Final best prompt: {best_prompt}; Final best response: {best_response}")
            except Exception as e:
                logger.error(f"Error parsing final ranking: {e}")
                # If parsing fails, use the last mutated response
                best_prompt = prompt_variations[-1]
                best_response = responses[-1]
        except Exception as e:
            logger.error(f"Error in final ranking: {e}")
            # If there's an error, use the last prompt and response
            best_prompt = prompt_variations[-1]
            best_response = responses[-1]
        
        # Return the best task prompt variatioin
        if return_full_response:
            return (prompt_variations, responses)
        else:
            return (best_prompt, best_response)

    #SC - placeholder
    def prompt_differential():
        return None

    #SC - placeholder
    def prompt_breeder():
        return None

    #SC - placeholder
    def prompt_phrase_evolution():
        return None

    #SC - placeholder
    def prompt_persona_search():
        return None

    def prompt_examplar():
        return None

    def prompt_reasoning(self,
        task_prompt: str,
        template: Literal["chain_of_thought", "tree_of_thought", "logic_of_thought", "program_synthesis", "deep_thought"],
        ) -> Dict[str, Any]:
        """
        Topology 6: Execute with chain-of-thought or tree-of-thought.
        
        Args:
            task_prompt: Task prompt
            template: One of (chain-of-thought), (tree-of-thought), (program synthesis), or (deep thought)
            
        Returns:
            Transformed prompt in json format
        """
        try:                   
            return self.template_adopter.get_prompt_transformation(
                prompt_dict=task_prompt,
                fix_template=template
            )
        except Exception as e:
            logger.error(f"Error in prompt transformation: {e}")            
            return None            

class SequentialTopology:
    """
    Implements different patterns for executing prompts using meta-prompts.
    """
    def __init__(self, generator = None):
        self.generator = generator if generator else Generator()
        self.metagenerator = MetaGenerator(generator = self.generator)
        self.evaluator = Evaluator()
        self.aiutility = AIUtility()
        self.template_adopter = TemplateAdopter()
        logger.debug("SequentialTopology initialized")

class ParallelTopology:
    """
    Implements different patterns for executing prompts using meta-prompts.
    """
    def __init__(self, generator = None):
        self.generator = generator if generator else Generator()
        self.metagenerator = MetaGenerator(generator = self.generator)
        self.evaluator = Evaluator()
        self.aiutility = AIUtility()
        self.template_adopter = TemplateAdopter()
        logger.debug("ParallelTopology initialized")

class HybridTopology:
    """
    Implements different patterns for executing prompts using meta-prompts.
    """
    def __init__(self, generator = None):
        self.generator = generator if generator else Generator()
        self.metagenerator = MetaGenerator(generator = self.generator)
        self.evaluator = Evaluator()
        self.aiutility = AIUtility()
        self.template_adopter = TemplateAdopter()
        logger.debug("HybridTopology initialized")

class ScalingTopology:
    """
    Implements different patterns for executing prompts using meta-prompts.
    """
    def __init__(self, generator = None):
        self.generator = generator if generator else Generator()
        self.metagenerator = MetaGenerator(generator = self.generator)
        self.evaluator = Evaluator()
        self.aiutility = AIUtility()
        logger.debug("ScalingTopology initialized")
        
    def best_of_n_synthesis(self,
        task_prompt: str,
        prompt_id: int,
        num_variations: int = 3,
        model: Optional[str] = "Qwen2.5-1.5B",
        temperature: Optional[float] = 0.7,
        return_full_response: Optional[bool] = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Topology 3: Generate multiple responses and synthesize.
        
        Args:
            task_prompt: Task prompt
            prompt_id: Prompt ID for tracking
            num_variations: Number of responses to generate
            model: Model to use for generation
            temperature: Temperature for generation
            **kwargs: Additional arguments for get_meta_generation
            
        Returns:
            Synthesized response
        """
        # Step 1: Generate multiple responses
        try:
            responses = []
            for i in range(num_variations):
                response = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=task_prompt,
                    model=model,
                    temperature=0.75,
                    return_full_response=False
                )
                responses.append(response)
                logger.debug(f"Generated response {i+1}: {response}")
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            return None

        # Step 2: Synthesize responses
        return self.metagenerator.get_meta_generation(
            application="metaresponse",
            category="manipulation",
            action="synthesize",
            prompt_id=prompt_id,
            system_prompt=None,
            responses=responses,
            model=model,
            temperature=temperature,  # Lower temperature for synthesis
            return_full_response=return_full_response
        )
        
    def best_of_n_selection(self,
        task_prompt: str,
        prompt_id: int,
        num_variations: int = 3,
        selection_method: str = "llm",
        model: Optional[str] = "Qwen2.5-1.5B",
        model_selector: Optional[str] = "Qwen2.5-1.5B",
        temperature: Optional[float] = 0.7,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Topology 4: Generate multiple responses and select best using different methods.
        
        Args:
            task_prompt: Task prompt
            prompt_id: Prompt ID for tracking
            num_variations: Number of responses to generate
            selection_method: One of ["perplexity", "similarity", "llm"]
            model: Model to use for generation
            temperature: Temperature for generation
            return_full_response: If True, returns a dictionary with response and metrics
            **kwargs: Additional arguments for get_meta_generation
            
        Returns:
            If return_full_response is False:
                Selected best response as a string
            If return_full_response is True:
                Dictionary containing response and additional metrics
        """
        # Step 1: Generate multiple responses with their perplexity scores
        try:
            responses = []
            perplexities = []
            for i in range(num_variations):
                result = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=task_prompt,
                    model=model,
                    temperature=kwargs.get('temperature', 0.75),
                    logprobs=True,  # Enable for perplexity calculation
                    return_full_response=True,  # Always get full response for internal processing
                )
                
                # Extract response and perplexity
                if isinstance(result, dict) and 'response' in result and 'perplexity' in result:
                    responses.append(result['response'])
                    perplexities.append(result['perplexity'])
                else:
                    # Fallback if we somehow didn't get a dictionary
                    responses.append(str(result))
                    perplexities.append(float('inf'))
                
                logger.debug(f"Generated response {i+1}: {responses[-1]}")
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            return None

        # Step 2: Select best response based on method
        try:
            if selection_method == "perplexity":
                # Return response with lowest perplexity
                best_idx = np.argmin(perplexities)
                return responses[best_idx]
                
            elif selection_method == "similarity":
                # Calculate pairwise similarities between responses
                similarities = np.zeros((len(responses), len(responses)))
                for i, resp1 in enumerate(responses):
                    for j, resp2 in enumerate(responses):
                        if i != j:
                            similarities[i,j] = self.evaluator.get_bertscore(
                                resp1, resp2, model="Jina-embeddings-v3"
                            )
                
                # Return response with highest average similarity to others
                avg_similarities = similarities.mean(axis=1)
                best_idx = np.argmax(avg_similarities)
                return responses[best_idx]
                
            else:  # llm selection
                return self.metagenerator.get_meta_generation(
                    application="metaresponse",
                    category="evaluation",
                    action="select",
                    prompt_id=prompt_id,
                    system_prompt=None,
                    task_prompt=task_prompt,
                    responses=responses,
                    model=model_selector,
                    temperature=0.1,  # Low temperature for selection
                    return_full_response=False
                )
        except Exception as e:
            logger.error(f"Error selecting response: {e}")
            return None

    def self_reflection(self,
        task_prompt: str,
        prompt_id: int,
        num_iterations: int = 1,
        model: Optional[str] = "Qwen2.5-1.5B",
        temperature: Optional[float] = 0.7,
        return_full_response: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Topology 5: Generate response, self-reflect, and improve through multiple iterations.
        
        Args:
            task_prompt: Task prompt
            prompt_id: Prompt ID for tracking
            num_iterations: Number of reflection and refinement iterations
            model: Model to use for generation
            temperature: Temperature for generation
            return_full_response: If True, returns dict with all intermediate responses and reflections
            
            
        Returns:
            If return_full_response is False:
                Final improved response after all iterations as a string
            If return_full_response is True:
                Dict containing:
                - 'final_response': Final improved response
                - 'iterations': List of dicts, each containing:
                    - 'response': Response for this iteration
                    - 'reflection': Reflection for this iteration
        """
        # Track intermediate results if requested
        iterations = []
        
        # Step 1: Generate initial response
        current_response = self.generator.get_completion(
            prompt_id=prompt_id,
            prompt=task_prompt,
            model=model,
            temperature=temperature,
            return_full_response=False,  # Always get full response for internal processing
        )
        
        # Store initial response
        iterations.append({
            'response': current_response,
            'reflection': None  # No reflection for initial response
        })
        
        # Perform requested number of reflection and refinement iterations
        for i in range(num_iterations):
            # Generate reflection on current response
            reflection = self.metagenerator.get_meta_generation(
                application="metaresponse",
                category="evaluation",
                action="reflect",
                prompt_id=prompt_id,
                system_prompt=None,
                task_prompt=task_prompt,
                response=current_response,
                model=model,
                temperature=0.5,
                return_full_response=False
            )
            logger.debug(f"Reflection for iteration {i+1}/{num_iterations}: {reflection}")
            
            # Generate improved response based on reflection
            current_response = self.metagenerator.get_meta_generation(
                application="metaresponse",
                category="manipulation",
                action="refine",
                prompt_id=prompt_id,
                system_prompt=None,
                task_prompt=task_prompt,
                response=current_response,
                feedback=reflection,
                model=model,
                temperature=temperature,
                return_full_response=False
            )
            logger.debug(f"Improved response for iteration {i+1}/{num_iterations}: {current_response}")
            
            # Store this iteration's results
            iterations.append({
                'response': current_response,
                'reflection': reflection
            })
    
        if return_full_response:
            return {
                'final_response': current_response,
                'iterations': iterations
            }
        else:
            return current_response

    def atom_of_thought(self,
                        task_prompt: str,
                        prompt_id: int,
                        model: str = "Qwen2.5-1.5B",
                        temperature: Optional[float] = 0.7,
                        max_iterations: int = 3,
                        return_full_response: Optional[bool] = False,
                        **kwargs
                        ) -> Union[str, Dict[str, Any]]:
        """
        Topology 6: Atom of Thoughts (AOT) - Decompose complex problems into atomic subquestions,
        create dependency graphs, contract to simpler states, and iterate until solution.
        
        Process:
        1. Decompose the original problem into atomic subquestions
        2. Construct a dependency graph (DAG) to identify relationships
        3. Contract independent subquestions into known conditions
        4. Validate the contracted question for Markov property and logical equivalence
        5. Determine if termination is possible or continue iteration
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            model: Model to use for generation
            temperature: Temperature for generation
            max_iterations: Maximum number of AOT iterations
            return_full_response: If True, returns a dictionary with all intermediate steps
            **kwargs: Additional arguments for generation
            
        Returns:
            If return_full_response is False:
                Final solution after all iterations as a string
            If return_full_response is True:
                Dict containing:
                - 'final_solution': Final solution
                - 'iterations': List of dicts, each containing:
                    - 'decomposition': Subquestions from decomposition
                    - 'dag': Dependency graph
                    - 'contraction': Contracted atomic question
                    - 'validation': Validation results
                    - 'termination': Termination decision and reason
        """
        # Track iterations if requested
        iterations = []
        current_problem = task_prompt
        
        # Execute AOT iterations
        for i in range(max_iterations):
            logger.info(f"Starting AOT iteration {i+1}/{max_iterations}")
            iteration_results = {}
            
            # Step 1: Problem Decomposition
            try:
                decomposition = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="decompose",
                    prompt_id=prompt_id,
                    task_prompt=current_problem,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                iteration_results['decomposition'] = decomposition
                logger.debug(f"Problem decomposition: {decomposition}")
            except Exception as e:
                logger.error(f"Error in problem decomposition: {e}")
                if return_full_response:
                    return {
                        'final_solution': "Error in AOT execution: Problem decomposition failed",
                        'iterations': iterations
                    }
                return "Error in AOT execution: Problem decomposition failed"
            
            # Step 2: Dependency Graph (DAG) Construction
            try:
                dag = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="dagify",
                    prompt_id=prompt_id,
                    task_prompt=current_problem,
                    decomposition=decomposition,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                iteration_results['dag'] = dag
                logger.debug(f"DAG construction: {dag}")
            except Exception as e:
                logger.error(f"Error in DAG construction: {e}")
                if return_full_response:
                    return {
                        'final_solution': "Error in AOT execution: DAG construction failed",
                        'iterations': iterations
                    }
                return "Error in AOT execution: DAG construction failed"
            
            # Step 3: Contraction (Atomic State Simplification)
            try:
                contraction = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="contract",
                    prompt_id=prompt_id,
                    task_prompt=current_problem,
                    decomposition=decomposition,
                    dag=dag,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                iteration_results['contraction'] = contraction
                logger.debug(f"Contraction: {contraction}")
            except Exception as e:
                logger.error(f"Error in contraction: {e}")
                if return_full_response:
                    return {
                        'final_solution': "Error in AOT execution: Contraction failed",
                        'iterations': iterations
                    }
                return "Error in AOT execution: Contraction failed"
            
            # Step 4: Iterative Validation (Markov Property & Logical Equivalence)
            try:
                validation = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="validate",
                    prompt_id=prompt_id,
                    task_prompt=current_problem,
                    contraction=contraction,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                iteration_results['validation'] = validation
                logger.debug(f"Validation: {validation}")
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                if return_full_response:
                    return {
                        'final_solution': "Error in AOT execution: Validation failed",
                        'iterations': iterations
                    }
                return "Error in AOT execution: Validation failed"
            
            # Step 5: Termination Decision & Solution Generation
            try:
                termination = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="terminate",
                    prompt_id=prompt_id,
                    task_prompt=current_problem,
                    contraction=contraction,
                    validation=validation,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                iteration_results['termination'] = termination
                logger.debug(f"Termination decision: {termination}")
                
                # Parse termination decision
                try:
                    termination_data = json.loads(termination)
                    decision = termination_data.get('decision', 'continue')
                    final_solution = termination_data.get('final_answer', None)
                    reason = termination_data.get('reason', 'No reason provided')
                    
                    # Store iteration results
                    iterations.append(iteration_results)
                    
                    # Check if we should terminate
                    if decision.lower() == 'terminate' and final_solution:
                        logger.info(f"AOT terminated after {i+1} iterations. Reason: {reason}")
                        if return_full_response:
                            return {
                                'final_solution': final_solution,
                                'iterations': iterations
                            }
                        return final_solution
                    
                    # Update current problem for next iteration
                    current_problem = contraction
                    
                except json.JSONDecodeError:
                    logger.error("Error parsing termination decision JSON")
                    # Continue with the process using the contraction as the new problem
                    current_problem = contraction
                
            except Exception as e:
                logger.error(f"Error in termination decision: {e}")
                if return_full_response:
                    return {
                        'final_solution': "Error in AOT execution: Termination decision failed",
                        'iterations': iterations
                    }
                return "Error in AOT execution: Termination decision failed"
        
        # If we've reached max iterations without termination
        logger.info(f"AOT reached maximum iterations ({max_iterations}) without termination")
        
        # Generate final solution from the last contraction
        try:
            final_solution = self.generator.get_completion(
                prompt_id=prompt_id,
                prompt=f"Based on the following simplified question, provide a final answer:\n\n{current_problem}",
                model=model,
                temperature=temperature,
                return_full_response=False
            )
            
            if return_full_response:
                return {
                    'final_solution': final_solution,
                    'iterations': iterations,
                    'note': f"Reached maximum iterations ({max_iterations}) without termination"
                }
            return final_solution
            
        except Exception as e:
            logger.error(f"Error generating final solution: {e}")
            if return_full_response:
                return {
                    'final_solution': "Error in AOT execution: Final solution generation failed",
                    'iterations': iterations
                }
            return "Error in AOT execution: Final solution generation failed"



    def multimodel_debate_solo(self,
                    task_prompt: str,
                    prompt_id: int,
                    num_iterations: int = 1,
                    defender_model: str = "gpt-4",
                    critic_model: str = "Qwen2.5-1.5B",
                    temperature: Optional[float] = 0.7,
                    return_full_response: Optional[bool] = False,
                    **kwargs
        ) -> Union[str, Dict[str, Any]]:
        """
        Topology 7: Debate between two models to refine response.
        
        Process for each round:
        1. Defender model generates initial response
        2. Critic model critiques the response
        3. Defender model defends and accepts valid critiques
        4. Defender model refines response based on accepted feedback
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            num_iterations: Number of debate iterations
            defender_model: Model to use for initial response, defense, and refinement
            critic_model: Model to use for critique
            temperature: Temperature for generation
            return_full_response: If True, returns a dictionary with response and metrics
            **kwargs: Additional arguments for generation
            
        Returns:
            If return_full_response is False:
                Final refined response after all debate iterations as a string
            If return_full_response is True:
                Dictionary containing response and additional metrics
        """
        # Track debate iterations if requested
        iterations = []
        
        # Get initial response from defender
        current_response = self.generator.get_completion(
            prompt_id=prompt_id,
            prompt=task_prompt,
            model=defender_model,
            temperature=1,  # High temperature for initial response
            return_full_response=False
        )
        
        # Store initial response if tracking
        iterations.append({
            'initial_response': current_response,
            'critique': None,
            'defense': None,
            'refined_response': None
        })
        
        # Conduct specified number of debate iterations
        for i in range(num_iterations):
            # Critic model analyzes defender's response
            critique = self.metagenerator.get_meta_generation(
                application="metaresponse",
                category="evaluation",
                action="critic",
                prompt_id=prompt_id,
                system_prompt=None,
                task_prompt=task_prompt,
                response=current_response,
                model=critic_model,
                temperature=0.3,  # Lower temperature for focused critique
                return_full_response=False
            )
            logger.debug(f"Critic to challenge in iteration {i} is: {critique}")

            # Defender model responds to critique
            defense = self.metagenerator.get_meta_generation(
                application="metaresponse",
                category="manipulation",
                action="defend",
                prompt_id=prompt_id,
                system_prompt=None,
                task_prompt=task_prompt,
                response=current_response,
                feedback=critique,
                model=defender_model,
                temperature=0.3,
                return_full_response=False
            )
            logger.debug(f"Defender to defend in iteration {i} is: {defense}")
            
            # Defender model refines response based on accepted feedback
            current_response = self.metagenerator.get_meta_generation(
                application="metaresponse",
                category="manipulation",
                action="refine",
                prompt_id=prompt_id,
                system_prompt=None,
                task_prompt=task_prompt,
                response=current_response,
                feedback=defense,
                model=defender_model,
                temperature=temperature,
                return_full_response=False
            )
            logger.debug(f"Defender to refine in iteration {i} is: {current_response}")
            
            # Store this round's results if tracking
            iterations.append({
                'initial_response': current_response,
                'critique': critique,
                'defense': defense,
                'refined_response': current_response
            })
        
        # Execute
        # Return based on whether we're storing intermediates
        if return_full_response:
            return {
                'final_response': current_response,
                'iterations': iterations
            }
        else:
            return current_response


    def multimodel_debate_dual(self,
        task_prompt: str,
        prompt_id: int,
        num_iterations: int = 1,
        model_strong: str = "gpt-4",
        model_weak: str = "Qwen2.5-1.5B",
        selector_model: Optional[str] = None,
        selection_method: Optional[str] = "llm",
        temperature: Optional[float] = 0.7,
        return_full_response: bool = False,
        **kwargs
        ) -> Union[str, Dict[str, Any]]:
        """
        Topology 8: Two models debate each other's responses with optional third model selection.
        
        Process:
        First iteration:
        1. Both models generate initial responses
        2. Each model challenges the other's response
        3. Each model defends and refines their response
        
        Subsequent iterations:
        1. Each model challenges other's refined response
        2. Each model defends and refines their response
        
        Final step:
        - If selector_model provided: Third model selects best response
        - Else: Use perplexity/similarity-based selection
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            num_iterations: Number of debate iterations
            model_strong: First model for debate
            model_weak: Second model for debate
            selector_model: Optional third model for final selection
            temperature: Temperature for generation
            return_full_response    : If True, returns a dictionary with response and metrics
            **kwargs: Additional arguments for generation
            
        Returns:
            If store_intermediate is False and return_full_response is False:
                Final refined response after all debate iterations as a string
            If store_intermediate is True:
                Dict containing:
                - 'final_response': Final refined response
                - 'iterations': List of dicts, each containing:
                    - 'model_strong_response': Model A's response
                    - 'model_weak_response': Model B's response
                    - 'model_strong_critique': Model A's critique of B
                    - 'model_weak_critique': Model B's critique of A
                    - 'model_strong_defense': Model A's defense
                    - 'model_weak_defense': Model B's defense
                    - 'model_strong_refined': Model A's refined response
                    - 'model_weak_refined': Model B's refined response
        """
        # Track debate iterations if requested
        iterations = []
        
        try:
            # Initial responses from both models
            response_strong = self.generator.get_completion(
                prompt_id=prompt_id,
                prompt=task_prompt,
                model=model_strong,
                temperature=temperature,
                return_full_response=False
            )
            
            response_weak = self.generator.get_completion(
                prompt_id=prompt_id,
                prompt=task_prompt,
                model=model_weak,
                temperature=temperature,
                return_full_response=False
            )
            logger.debug(f"Model Strong responds: {response_strong}")
            logger.debug(f"Model Weak responds: {response_weak}")

            # Store initial responses if tracking
            iterations.append({
                'model_strong_response': response_strong,
                'model_weak_response': response_weak,
                'model_strong_critique': None,
                'model_weak_critique': None,
                'model_strong_defense': None,
                'model_weak_defense': None,
                'model_strong_refined': None,
                'model_weak_refined': None
            })
            
            # Conduct debate iterations
            for i in range(num_iterations):
                # Model A critiques Model B's response
                critique_strong = self.metagenerator.get_meta_generation(
                    application="metaresponse",
                    category="evaluation",
                    action="critic",
                    prompt_id=prompt_id,
                    system_prompt=None,
                    task_prompt=task_prompt,
                    response=response_weak,
                    model=model_strong,
                    temperature=0.3,
                    return_full_response=False
                )
                logger.debug(f"Critiquer to critique in iteration {i} is: {critique_strong}")
                
                # Model B critiques Model A's response
                critique_weak = self.metagenerator.get_meta_generation(
                    application="metaresponse",
                    category="evaluation",
                    action="critic",
                    prompt_id=prompt_id,
                    system_prompt=None,
                    task_prompt=task_prompt,
                    response=response_strong,
                    model=model_weak,
                    temperature=0.3,
                    return_full_response=False
                )
                logger.debug(f"Critiquer to critique in iteration {i} is: {critique_weak}")
                
                # Model A defends and refines
                defense_strong = self.metagenerator.get_meta_generation(
                    application="metaresponse",
                    category="manipulation",
                    action="defend",
                    prompt_id=prompt_id,
                    system_prompt=None,
                    task_prompt=task_prompt,
                    response=response_strong,
                    feedback=critique_weak,
                    model=model_strong,
                    temperature=0.3,
                    return_full_response=False
                )
                logger.debug(f"Defender to defend in iteration {i} is: {defense_strong}")
                
                response_strong = self.metagenerator.get_meta_generation(
                    application="response",
                    category="manipulation",
                    action="refine",
                    prompt_id=prompt_id,
                    system_prompt=None,
                    task_prompt=task_prompt,
                    response=response_strong,
                    feedback=json.loads(defense_strong),
                    model=model_strong,
                    temperature=temperature,
                    return_full_response=False
                )
                logger.debug(f"Defender to refine in iteration {i} is: {defense_strong}")

                # Model Weak defends and refines
                defense_weak = self.metagenerator.get_meta_generation(
                    application="response",
                    category="manipulation",
                    action="defend",
                    prompt_id=prompt_id,
                    system_prompt=None,
                    task_prompt=task_prompt,
                    response=response_weak,
                    feedback=critique_strong,
                    model=model_weak,
                    temperature=0.3,
                    return_full_response=False
                )
                logger.debug(f"Defender to defend in iteration {i} is: {defense_weak}")

                response_weak = self.metagenerator.get_meta_generation(
                    application="response",
                    category="manipulation",
                    action="refine",
                    prompt_id=prompt_id,
                    system_prompt=None,
                    task_prompt=task_prompt,
                    response=response_weak,
                    feedback=json.loads(defense_weak),
                    model=model_weak,
                    temperature=temperature,
                    return_full_response=False
                )
                logger.debug(f"Defender to refine in iteration {i} is: {defense_weak}")
                
                # Store this round's results if tracking
                iterations.append({
                    'model_strong_response': response_strong,
                    'model_weak_response': response_weak,
                    'model_strong_critique': critique_strong,
                    'model_weak_critique': critique_weak,
                    'model_strong_defense': defense_strong,
                    'model_weak_defense': defense_weak,
                    'model_strong_refined': response_strong,
                    'model_weak_refined': response_weak
                })
            
            # Final selection of best response
            responses = [response_strong, response_weak]
            
            # Use third model for selection
            final_response = self.metagenerator.get_meta_generation(
                application="metaresponse",
                category="evaluation",
                action="select",
                task_prompt=task_prompt,
                prompt_id=prompt_id,
                system_prompt=None,
                responses=responses,
                selection_method=selection_method if selection_method else "",
                model=selector_model if selection_method == "llm" else None,
                temperature=0.3,
                return_full_response=False
            )
            
            if return_full_response:
                return {
                    'final_response': final_response,
                    'iterations': iterations
                }
            else:
                return final_response
            
        except Exception as e:
            logger.error(f"Error in dual model debate: {e}")
            return None

    def multipath_disambiguation_selection(self,
                    task_prompt: str,
                    prompt_id: int,
                    num_variations: int = 3,
                    responses_per_prompt: int = 3,
                    model: Optional[str] = "Qwen2.5-1.5B",
                    temperature: Optional[float] = 0.7,
                    selection_method: str = "llm",
                    selection_model: Optional[str] = None,
                    return_full_response: bool = False,
                    **kwargs
        ) -> Union[str, Dict[str, Any]]:
        """
        Topology 9: Multi-path prompt disambiguation with staged response selection.
        
        Process:
        1. Generate multiple prompt variations through disambiguation
        2. For each prompt variation:
           - Generate multiple responses
           - Select best response using specified method
        3. Final selection from shortlisted responses
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            num_variations: Number of prompt variations to generate
            responses_per_prompt: Number of responses to generate per prompt
            model: Model to use for generation
            temperature: Temperature for generation
            selection_method: Method for response selection ("llm", "perplexity", "similarity")
            return_full_response: If True, returns dict with all intermediate steps
            
        Returns:
            If return_full_response is False:
                Final selected response
            If return_full_response is True:
                Dict containing:
                - 'final_response': Final selected response
                - 'prompt_variations': List of rewritten prompts
                - 'shortlisted_responses': Dict mapping prompt variations to their best responses
                - 'selection_metrics': Metrics used for final selection
        """
        try:
            # Track intermediate results if requested
            prompt_variations = []
            shortlisted_responses = {}
            
            # Step 1: Generate prompt variations through disambiguation
            for i in range(num_variations):
                variation = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="manipulation",
                    action="disambiguate",
                    prompt_id=prompt_id,
                    system_prompt=None,
                    task_prompt=task_prompt,
                    model=model,
                    temperature=0.5,  # Moderate temperature for diverse variations
                    return_full_response=False
                )
                prompt_variations.append(variation)
                logger.debug(f"Generated prompt variation {i+1}: {variation}")

            # Step 2: Generate and select responses for each prompt variation
            for i, prompt_variation in enumerate(prompt_variations):
                # Generate multiple responses for this prompt variation
                responses = []
                for j in range(responses_per_prompt):
                    response = self.generator.get_completion(
                        prompt_id=prompt_id,
                        prompt=prompt_variation,
                        model=model,
                        temperature=temperature,
                        return_full_response=False
                    )
                    responses.append(response)
                    logger.debug(f"Generated response {j+1} for prompt variation {i+1}: {response}")

                # Select best response for this prompt variation
                best_response = self.best_of_n_selection(
                    task_prompt=prompt_variation,
                    prompt_id=prompt_id,
                    responses=responses,
                    selection_method=selection_method,
                    model=model,
                    temperature=0.3  # Lower temperature for selection
                )
                logger.debug(f"Selected best response for prompt variation {i+1}: {best_response}")

                shortlisted_responses[prompt_variation] = best_response
            
            # Step 3: Final selection from shortlisted responses
            final_candidates = list(shortlisted_responses.values())
            logger.debug(f"Final candidates of {num_variations} variations: {final_candidates}")

            final_response = self.best_of_n_selection(
                task_prompt=task_prompt,  # Use original prompt for final selection
                prompt_id=prompt_id,
                responses=final_candidates,
                num_generations=len(final_candidates),
                selection_method=selection_method,
                model=selection_model if selection_method == "llm" else None,
                temperature=0.3,
                return_full_response=False
            )
            
            if return_full_response:
                return {
                    'final_response': final_response,
                    'prompt_variations': prompt_variations,
                    'shortlisted_responses': shortlisted_responses,
                    'selection_metrics': {
                        'method': selection_method,
                        'num_variations': num_variations,
                        'responses_per_prompt': responses_per_prompt
                    }
                }
            else:
                return final_response
            
        except Exception as e:
            logger.error(f"Error in multi-path disambiguation and selection: {e}")
            raise

    def socratic_dialogue(self,
                    task_prompt: str,
                    prompt_id: int,
                    num_questions: int = 3,
                    model: Optional[str] = "Qwen2.5-1.5B",
                    temperature: Optional[float] = 0.7,
                    return_full_response: bool = False,
                    **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Topology 10: Socratic Dialogue - Uses meta-prompts to generate probing questions,
        leading to deeper understanding and more comprehensive responses.
        
        Process:
        1. Generate key questions about the task
        2. Answer each question systematically
        3. Synthesize answers into final response
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            num_questions: Number of Socratic questions to generate
            model: Model to use for generation
            temperature: Temperature for generation
            return_full_response: If True, returns a dictionary with intermediate steps
            **kwargs: Additional arguments for generation
            
        Returns:
            If return_full_response is False:
                Final synthesized response
            If return_full_response is True:
                Dict containing:
                - questions: Generated Socratic questions
                - answers: Answers to each question
                - final_response: Synthesized response
        """
        try:
            # Step 1: Generate Socratic questions
            question_prompt = self.metagenerator.get_meta_generation(
                application="metaprompt",
                category="evaluation",
                action="question",
                prompt_id=prompt_id,
                system_prompt=None,
                model=model,
                task_prompt=task_prompt,
                return_full_response=False
            )
            
            questions = []
            for i in range(num_questions):
                question = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=question_prompt,
                    model=model,
                    temperature=1,  # Slightly higher for diversity
                    return_full_response=False
                )
                questions.append(question)
                logger.debug(f"Generated question {i+1}/{num_questions}: {question}")

            # Step 2: Answer each question
            answers = []
            for question in questions:
                answer = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=f"{task_prompt}\n\nTo address this task, let's answer: {question}",
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                answers.append(answer)
                logger.debug(f"Generated answer {i+1}/{len(questions)}: {answer}")
            
            # Step 3: Synthesize prompt            
            final_response = self.generator.get_completion(
                prompt_id=prompt_id,
                prompt=f"{task_prompt}\n\n[Insights]:\n" + "\n".join(answers),
                model=model,
                temperature=temperature,  # Lower for more focused synthesis
                return_full_response=False
            )
            
            if return_full_response:
                return {
                    "final_response": final_response,
                    "questions": questions,
                    "answers": answers
                }
            else:
                return final_response
            
        except Exception as e:
            logger.error(f"Error in Socratic dialogue topology: {e}")
            raise

    def hierarchical_decomposition(self,
                    task_prompt: str,
                    prompt_id: int,
                    model_planner: Optional[str] = "Qwen2.5-1.5B",
                    model_executor: Optional[str] = "Qwen2.5-1.5B",
                    temperature: Optional[float] = 0.7,
                    return_full_response: bool = False,
                    **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Topology 11: Hierarchical Decomposition - Breaks down complex tasks into hierarchical subtasks,
        solves them in dependency order, then reconstructs the solution bottom-up.
        
        Process:
        1. Decompose task into subquestions with explicit dependencies
        2. Build a dependency graph and determine execution order
        3. Generate solutions for each subquestion in dependency order
        4. Group solutions from each subquestions
        5. Synthesize final solutions addressing the original task prompt
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            model_planner: Model to use for planning and decomposition
            model_executor: Model to use for execution and synthesis
            temperature: Temperature for generation
            return_full_response: If True, returns dict with intermediate steps
            **kwargs: Additional arguments for generation
            
        Returns:
            If return_full_response is False:
                Final synthesized response
            If return_full_response is True:
                Dict containing:
                - decomposition: Structured decomposition with dependencies
                - execution_order: Order in which subquestions were solved
                - solutions: Solutions for all subquestions
                - final_response: Final synthesized response
        """
        try:
            # Step 1: Get structured decomposition with dependencies using the reasoning category
            try:
                decomposition_result = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",  # Using reasoning category for structured decomposition
                    action="decompose",
                    prompt_id=prompt_id,
                    task_prompt=task_prompt,
                    model=model_planner,
                    temperature=0.3,  # Lower temperature for more consistent decomposition
                    return_full_response=True
                )
                
                # Parse the structured output
                if isinstance(decomposition_result, str):
                    try:
                        decomposition_data = json.loads(decomposition_result)
                    except json.JSONDecodeError:
                        # Handle case where result is not valid JSON
                        logger.warning("Decomposition result is not valid JSON, using fallback")
                        raise ValueError("Invalid JSON in decomposition result")
                else:
                    decomposition_data = decomposition_result
                
                # Extract subquestions with their dependencies
                subquestions = decomposition_data.get("subquestions", [])
                if not subquestions:
                    logger.warning("No subquestions found in decomposition result, using fallback")
                    raise ValueError("No subquestions in decomposition result")
                
                logger.debug(f"Decomposed into {len(subquestions)} subquestions with dependencies")
                
            except Exception as e:
                logger.error(f"Structured decomposition failed: {e}")
                raise
            
            # Step 2: Build dependency graph and determine execution order
            dependency_graph = {}
            for subq in subquestions:
                subq_id = subq.get("id")
                # Handle both 'question' and 'subquestion' keys for compatibility
                subq_text = subq.get("question", subq.get("subquestion", ""))
                subq_deps = subq.get("dependencies", [])
                
                if not subq_text:
                    logger.warning(f"Empty subquestion text for ID {subq_id}, skipping")
                    continue
                
                dependency_graph[subq_id] = {
                    "id": subq_id,
                    "text": subq_text,
                    "dependencies": subq_deps,
                    "solution": None
                }
            
            # Determine execution order using topological sort
            def topological_sort(graph):
                """Sort subquestions based on dependencies."""
                visited = set()
                temp_marked = set()
                order = []
                
                def visit(node_id):
                    if node_id in temp_marked:
                        logger.warning(f"Cycle detected in dependencies involving node {node_id}")
                        return  # Skip this node to avoid infinite recursion
                    if node_id not in visited and node_id in graph:
                        temp_marked.add(node_id)
                        for dep_id in graph[node_id]["dependencies"]:
                            if dep_id in graph:  # Only visit if the dependency exists
                                visit(dep_id)
                        temp_marked.remove(node_id)
                        visited.add(node_id)
                        order.append(node_id)
                
                # Start with nodes that have no dependents (leaf nodes in terms of dependency)
                all_nodes = set(graph.keys())
                dependent_nodes = set()
                for node_data in graph.values():
                    dependent_nodes.update(node_data["dependencies"])
                
                # Find nodes that no other nodes depend on
                leaf_nodes = all_nodes - dependent_nodes
                
                # Process leaf nodes first, then any remaining nodes
                for node_id in leaf_nodes:
                    if node_id not in visited and node_id in graph:
                        visit(node_id)
                
                for node_id in graph:
                    if node_id not in visited:
                        visit(node_id)
                
                # Reverse to get dependency-first order
                return list(reversed(order))
            
            try:
                execution_order = topological_sort(dependency_graph)
                logger.debug(f"Execution order determined: {execution_order}")
            except Exception as e:
                logger.error(f"Failed to determine execution order: {e}")
                raise
            
            # Step 3: Generate solutions for subquestions in dependency order
            solutions = {}
            
            for subq_id in execution_order:
                if subq_id not in dependency_graph:
                    logger.warning(f"Subquestion ID {subq_id} not found in dependency graph, skipping")
                    continue
                
                subq_data = dependency_graph[subq_id]  # Get subquestion data
                subq_text = subq_data["text"]          # Get subquestion text
                subq_deps = subq_data["dependencies"]  # Get subquestion dependencies
                
                # Gather solutions from dependencies
                dep_context = ""
                if subq_deps:
                    dep_context = "\n\nContext from dependent questions:\n"
                    for dep_id in subq_deps:
                        if dep_id in solutions:
                            dep_text = dependency_graph[dep_id]["text"] if dep_id in dependency_graph else f"Question {dep_id}"
                            dep_context += f"Question: {dep_text}\nAnswer: {solutions[dep_id]}\n\n"
                
                # Generate solution with dependency context
                prompt = f"Question: {subq_text}"
                if dep_context:
                    prompt += f"\n{dep_context}\nNow answer the original question using this context."
                
                solution = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    model=model_executor,
                    temperature=temperature
                )
                
                solutions[subq_id] = solution
                dependency_graph[subq_id]["solution"] = solution
                logger.debug(f"Generated solution for subquestion {subq_id}")
            
            # Step 4: Synthesize final response
            # Prepare context with all subquestions and their solutions
            synthesis_context = f"Original task: {task_prompt}\n\nBreakdown of subquestions and solutions:\n"
            
            # Add subquestions in a logical order (by ID for clarity)
            ordered_subq_ids = sorted(dependency_graph.keys())
            for subq_id in ordered_subq_ids:
                subq_data = dependency_graph[subq_id]
                subq_text = subq_data["text"]
                subq_solution = subq_data["solution"] or "No solution generated"
                subq_deps = subq_data["dependencies"]
                
                deps_text = ", ".join([str(dep_id) for dep_id in subq_deps]) if subq_deps else "None"
                synthesis_context += f"\nSubquestion {subq_id}: {subq_text}\nDependencies: {deps_text}\nSolution: {subq_solution}\n"
            
            synthesis_context += "\nBased on all the above subquestions and their solutions, provide a comprehensive answer to the original task."
            
            # Generate final synthesized response
            final_response = self.generator.get_completion(
                prompt_id=prompt_id,
                prompt=synthesis_context,
                model=model_executor,
                temperature=max(0.2, temperature - 0.2)  # Slightly lower temperature for synthesis
            )
            
            if return_full_response:
                return {
                    "decomposition": subquestions,
                    "dependency_graph": dependency_graph,
                    "execution_order": execution_order,
                    "solutions": solutions,
                    "final_response": final_response
                }
            else:
                return final_response
            
        except Exception as e:
            logger.error(f"Error in hierarchical decomposition topology: {e}")
            raise

    def regenerative_majority_synthesis(self,
        task_prompt: str,
        prompt_id: int,
        num_initial_responses: int = 3,
        num_regen_responses: int = 3,
        cut_off_fraction: float = 0.5,
        synthesis_method: str = "majority_vote",
        model: Optional[str] = "Qwen2.5-1.5B",
        temperature: Optional[float] = 0.7,
        return_full_response: Optional[bool] = False,
        **kwargs
        ) -> Union[str, Dict[str, Any]]:
        """
        Topology 12: Regenerative Majority Synthesis - Generate multiple responses, truncate, regenerate, and synthesize final answer.
        
        Process:
        1. Generate num_initial_responses responses to the task prompt.
        2. Truncate each response to cut_off_fraction of its length.
        3. For each truncated response, combine with the original task prompt and generate num_regen_responses full responses.
        4. Apply synthesis_method ('majority_vote' or 'synthesis') to select or create the final response.
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            num_initial_responses: Number of initial responses to generate (default: 3)
            num_regen_responses: Number of regenerated responses per initial response (default: 3)
            cut_off_fraction: Fraction of response length to keep before regeneration (default: 0.5)
            synthesis_method: Method for final response selection ('majority_vote' or 'synthesis')
            model: Model to use for generation
            temperature: Temperature for generation
            return_full_response: If True, returns a dictionary with intermediate steps and metrics
            **kwargs: Additional arguments for generation
            
        Returns:
            If return_full_response is False:
                Final synthesized response as a string
            If return_full_response is True:
                Dictionary containing:
                - 'final_response': Final response
                - 'initial_responses': List of initial responses
                - 'truncated_responses': List of truncated responses
                - 'regenerated_responses': List of lists of regenerated responses
                - 'synthesis_details': Details of the synthesis process
        """
        start_time = time.time()
        logger.info(f"Starting regenerative_majority_synthesis with {num_initial_responses} initial responses, {num_regen_responses} regenerated responses per truncation, using {synthesis_method} synthesis method")
        logger.debug(f"Parameters: model={model}, temperature={temperature}, cut_off_fraction={cut_off_fraction}")
        
        # Track performance metrics
        metrics = {
            "step_timings": {},
            "response_counts": {}
        }
        
        try:
            # Step 1: Generate initial responses
            step1_start = time.time()
            logger.info("Step 1: Generating initial responses")
            initial_responses = []
            for i in range(num_initial_responses):
                response = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=task_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                initial_responses.append(response)
                response_summary = response[:100] + "..." if len(response) > 100 else response
                logger.debug(f"Initial response {i+1}/{num_initial_responses}: {response_summary}")
            
            metrics["step_timings"]["initial_generation"] = time.time() - step1_start
            metrics["response_counts"]["initial"] = len(initial_responses)
            logger.info(f"Generated {len(initial_responses)} initial responses in {metrics['step_timings']['initial_generation']:.2f} seconds")

            # Step 2: Truncate each response
            step2_start = time.time()
            logger.info(f"Step 2: Truncating responses at {cut_off_fraction} fraction")
            truncated_responses = []
            for i, resp in enumerate(initial_responses):
                len_resp = len(resp)
                cut_off_len = int(len_resp * cut_off_fraction)
                truncated = resp[:cut_off_len]  # Simple truncation to specified fraction
                truncated_responses.append(truncated)
                logger.debug(f"Truncated response {i+1}/{len(initial_responses)}: from {len_resp} chars to {len(truncated)} chars")
            
            metrics["step_timings"]["truncation"] = time.time() - step2_start
            metrics["response_counts"]["truncated"] = len(truncated_responses)
            logger.info(f"Truncated {len(truncated_responses)} responses in {metrics['step_timings']['truncation']:.2f} seconds")

            # Step 3: Regenerate full responses for each truncated one
            step3_start = time.time()
            logger.info(f"Step 3: Regenerating {num_regen_responses} responses for each truncated response")
            regenerated_responses = []
            total_regen = 0
            
            for i, trunc_resp in enumerate(truncated_responses):
                regen_prompt = f"{task_prompt} Continue from: {trunc_resp}"
                regen_list = []
                logger.debug(f"Regenerating from truncated response {i+1}/{len(truncated_responses)}")
                
                for j in range(num_regen_responses):
                    regen_response = self.generator.get_completion(
                        prompt_id=prompt_id,
                        prompt=regen_prompt,
                        model=model,
                        temperature=temperature,
                        return_full_response=False
                    )
                    regen_list.append(regen_response)
                    total_regen += 1
                    response_summary = regen_response[:100] + "..." if len(regen_response) > 100 else regen_response
                    logger.debug(f"Regenerated response {j+1}/{num_regen_responses} from truncation {i+1}: {response_summary}")
                    
                regenerated_responses.append(regen_list)
            
            metrics["step_timings"]["regeneration"] = time.time() - step3_start
            metrics["response_counts"]["regenerated"] = total_regen
            logger.info(f"Generated {total_regen} regenerated responses in {metrics['step_timings']['regeneration']:.2f} seconds")

            # Step 4: Apply synthesis
            step4_start = time.time()
            logger.info(f"Step 4: Applying {synthesis_method} synthesis method")
            all_regen_responses = [item for sublist in regenerated_responses for item in sublist]
            
            if synthesis_method == "majority_vote":
                logger.debug(f"Applying majority vote synthesis on {len(all_regen_responses)} responses")
                final_response = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="synthesis",
                    action="majority_vote",
                    prompt_id=prompt_id,
                    task_prompt=task_prompt,
                    responses=all_regen_responses,
                    model=model,
                    temperature=0.5,
                    return_full_response=False
                )
            elif synthesis_method == "synthesis":
                logger.debug(f"Applying combination synthesis on {len(all_regen_responses)} responses")
                final_response = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="synthesis",
                    action="combine",
                    prompt_id=prompt_id,
                    task_prompt=task_prompt,
                    responses=all_regen_responses,
                    model=model,
                    temperature=0.5,
                    return_full_response=False
                )
            else:
                logger.error(f"Invalid synthesis_method: {synthesis_method}")
                raise ValueError(f"Invalid synthesis_method: {synthesis_method}")
            
            metrics["step_timings"]["synthesis"] = time.time() - step4_start
            total_time = time.time() - start_time
            metrics["total_execution_time"] = total_time
            
            # Log final results summary
            final_summary = final_response[:150] + "..." if len(final_response) > 150 else final_response
            logger.info(f"Synthesis completed in {metrics['step_timings']['synthesis']:.2f} seconds")
            logger.info(f"Total regenerative_majority_synthesis execution time: {total_time:.2f} seconds")
            logger.debug(f"Final response summary: {final_summary}")

            if return_full_response:
                return {
                    "final_response": final_response,
                    "initial_responses": initial_responses,
                    "truncated_responses": truncated_responses,
                    "regenerated_responses": regenerated_responses,
                    "synthesis_method": synthesis_method,
                    "synthesis_details": "Synthesis applied based on method",
                    "performance_metrics": metrics
                }
            else:
                return final_response

        except Exception as e:
            # Log comprehensive error details
            total_time = time.time() - start_time
            logger.error(f"Error in regenerative_majority_synthesis after {total_time:.2f} seconds: {e}")
            logger.error(f"Error details: {type(e).__name__}, {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
    def adaptive_dag_reasoning(self,
        task_prompt: str,
        prompt_id: int,
        max_iterations: int = 3,
        contraction_threshold: float = 0.7,
        max_depth: int = 3,
        model: Optional[str] = "Qwen2.5-1.5B",
        temperature: Optional[float] = 0.7,
        return_full_response: Optional[bool] = False,
        **kwargs
        ) -> Union[str, Dict[str, Any]]:
        """
        Topology: Adaptive DAG Reasoning - Organically merges hierarchical decomposition and atom of thought
        approaches to create a flexible, iterative problem-solving system that adapts to task complexity.
        
        Process:
        1. Initial problem analysis and complexity assessment
        2. Adaptive decomposition into atomic subproblems with explicit dependencies
        3. Dynamic DAG construction with both hierarchical and atomic relationships
        4. Iterative contraction and validation cycles
        5. Opportunistic early solving of subproblems
        6. Progressive solution building with continuous validation
        7. Final synthesis with hierarchical reconstruction
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            max_iterations: Maximum number of contraction-validation cycles
            contraction_threshold: Threshold for determining when a subproblem is sufficiently contracted
            max_depth: Maximum depth for hierarchical decomposition
            model: Model to use for generation
            temperature: Temperature for generation
            return_full_response: If True, returns dict with all intermediate steps
            **kwargs: Additional arguments for generation
            
        Returns:
            If return_full_response is False:
                Final synthesized response
            If return_full_response is True:
                Dict containing:
                - 'final_solution': Final solution
                - 'problem_analysis': Initial problem analysis
                - 'dag': The dynamic DAG structure with hierarchical and atomic relationships
                - 'iterations': List of contraction-validation cycles
                - 'subproblem_solutions': Solutions for all subproblems
        """
        start_time = time.time()
        logger.info(f"Starting adaptive_dag_reasoning with max {max_iterations} iterations, contraction threshold {contraction_threshold}, and max depth {max_depth}")
        
        # Track execution data
        execution_data = {
            "problem_analysis": None,
            "dag": {
                "nodes": {},
                "edges": [],
                "hierarchical_groups": {}
            },
            "iterations": [],
            "subproblem_solutions": {},
            "performance_metrics": {
                "step_timings": {},
                "complexity_metrics": {}
            }
        }
        
        try:
            # Step 1: Initial problem analysis and complexity assessment
            analysis_start = time.time()
            logger.info("Step 1: Performing initial problem analysis and complexity assessment")
            
            problem_analysis = self.metagenerator.get_meta_generation(
                application="metaprompt",
                category="reasoning",
                action="analyze",
                prompt_id=prompt_id,
                task_prompt=f"""Analyze this problem in detail:
                {task_prompt}
                
                Provide a structured analysis including:
                1. Key components and variables in the problem
                2. Inherent complexity factors (scale, interdependencies, constraints)
                3. Knowledge domains required to solve it
                4. Potential solution approaches
                5. Estimated complexity level (1-5 scale)
                
                Format your response as JSON with these fields.""",
                model=model,
                temperature=0.3,  # Lower temperature for analytical task
                return_full_response=False
            )
            
            # Parse analysis result
            try:
                analysis_data = json.loads(problem_analysis) if isinstance(problem_analysis, str) else problem_analysis
                execution_data["problem_analysis"] = analysis_data
                complexity_level = analysis_data.get("estimated_complexity", 3)  # Default to medium complexity
                logger.info(f"Problem analysis complete. Estimated complexity level: {complexity_level}/5")
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse problem analysis as JSON, using raw text")
                execution_data["problem_analysis"] = {"raw_analysis": problem_analysis, "estimated_complexity": 3}
                complexity_level = 3  # Default to medium complexity
            
            execution_data["performance_metrics"]["step_timings"]["problem_analysis"] = time.time() - analysis_start
            execution_data["performance_metrics"]["complexity_metrics"]["initial_complexity"] = complexity_level
            
            # Step 2: Adaptive decomposition based on complexity
            decomp_start = time.time()
            logger.info("Step 2: Performing adaptive decomposition")
            
            # Adjust decomposition approach based on complexity
            decomposition_depth = min(max_depth, max(1, complexity_level - 1))  # Scale depth with complexity
            atomic_granularity = min(5, max(2, complexity_level))  # Scale granularity with complexity
            
            logger.debug(f"Using decomposition depth {decomposition_depth} and atomic granularity {atomic_granularity}")
            
            # Perform decomposition with both hierarchical and atomic elements
            decomposition_result = self.metagenerator.get_meta_generation(
                application="metaprompt",
                category="reasoning",
                action="decompose",
                prompt_id=prompt_id,
                task_prompt=f"""Decompose this problem into a structured hierarchy of subproblems:
                {task_prompt}
                
                Follow these guidelines:
                1. Create a {decomposition_depth}-level hierarchical decomposition
                2. At the lowest level, break problems into {atomic_granularity} atomic subproblems
                3. Explicitly identify dependencies between subproblems
                4. Assign unique IDs to each subproblem (e.g., H1.2.3 for hierarchical, A1, A2 for atomic)
                5. Ensure each atomic subproblem is self-contained and solvable
                
                Format your response as JSON with these fields:
                - hierarchical_structure: The hierarchical breakdown with nested subproblems
                - atomic_subproblems: List of atomic subproblems with IDs, descriptions, and dependencies
                - cross_level_dependencies: Any dependencies between hierarchical and atomic elements""",
                model=model,
                temperature=0.4,
                return_full_response=False
            )
            
            # Parse decomposition result
            try:
                decomposition_data = json.loads(decomposition_result) if isinstance(decomposition_result, str) else decomposition_result
                
                # Extract hierarchical and atomic elements
                hierarchical_structure = decomposition_data.get("hierarchical_structure", {})
                atomic_subproblems = decomposition_data.get("atomic_subproblems", [])
                cross_dependencies = decomposition_data.get("cross_level_dependencies", [])
                
                logger.info(f"Decomposition complete with {len(atomic_subproblems)} atomic subproblems")
                
                # Build the DAG structure
                # Add atomic nodes
                for subproblem in atomic_subproblems:
                    node_id = subproblem.get("id")
                    execution_data["dag"]["nodes"][node_id] = {
                        "id": node_id,
                        "type": "atomic",
                        "description": subproblem.get("description", ""),
                        "dependencies": subproblem.get("dependencies", []),
                        "status": "pending",
                        "solution": None
                    }
                
                # Add hierarchical structure
                def process_hierarchical_node(node, parent_id=None, path=""):
                    if isinstance(node, dict):
                        for key, value in node.items():
                            if key.startswith("H"):
                                # This is a hierarchical node ID
                                node_path = path + "." + key if path else key
                                node_description = value.get("description", "") if isinstance(value, dict) else ""
                                
                                # Add to hierarchical groups
                                execution_data["dag"]["hierarchical_groups"][key] = {
                                    "id": key,
                                    "description": node_description,
                                    "parent": parent_id,
                                    "children": []
                                }
                                
                                if parent_id and parent_id in execution_data["dag"]["hierarchical_groups"]:
                                    execution_data["dag"]["hierarchical_groups"][parent_id]["children"].append(key)
                                
                                # Process children
                                if isinstance(value, dict):
                                    process_hierarchical_node(value, key, node_path)
                
                process_hierarchical_node(hierarchical_structure)
                
                # Add edges based on dependencies
                for node_id, node_data in execution_data["dag"]["nodes"].items():
                    for dep_id in node_data["dependencies"]:
                        if dep_id in execution_data["dag"]["nodes"]:
                            execution_data["dag"]["edges"].append({
                                "source": dep_id,
                                "target": node_id
                            })
                
                # Add cross-level dependencies
                for cross_dep in cross_dependencies:
                    source = cross_dep.get("source")
                    target = cross_dep.get("target")
                    if source and target:
                        execution_data["dag"]["edges"].append({
                            "source": source,
                            "target": target,
                            "type": "cross_level"
                        })
                
                logger.debug(f"DAG constructed with {len(execution_data['dag']['nodes'])} nodes and {len(execution_data['dag']['edges'])} edges")
                
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse decomposition result: {e}")
                raise ValueError(f"Invalid decomposition result: {e}")
            
            execution_data["performance_metrics"]["step_timings"]["decomposition"] = time.time() - decomp_start
            
            # Step 3: Determine execution order using topological sort
            ordering_start = time.time()
            logger.info("Step 3: Determining execution order")
            
            # Implement topological sort
            def topological_sort(dag):
                nodes = dag["nodes"]
                edges = dag["edges"]
                
                # Build adjacency list
                adjacency = {node_id: [] for node_id in nodes}
                in_degree = {node_id: 0 for node_id in nodes}
                
                for edge in edges:
                    source = edge["source"]
                    target = edge["target"]
                    if source in adjacency and target in in_degree:
                        adjacency[source].append(target)
                        in_degree[target] += 1
                
                # Find nodes with no dependencies
                queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
                execution_order = []
                
                while queue:
                    current = queue.pop(0)
                    execution_order.append(current)
                    
                    for neighbor in adjacency[current]:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
                
                # Check for cycles
                if len(execution_order) != len(nodes):
                    logger.warning("DAG contains cycles, some nodes will be skipped")
                    # Add remaining nodes to avoid completely failing
                    for node_id in nodes:
                        if node_id not in execution_order:
                            execution_order.append(node_id)
                
                return execution_order
            
            execution_order = topological_sort(execution_data["dag"])
            logger.debug(f"Execution order determined: {execution_order}")
            
            execution_data["execution_order"] = execution_order
            execution_data["performance_metrics"]["step_timings"]["ordering"] = time.time() - ordering_start
            
            # Step 4: Iterative contraction and validation cycles
            current_problem = task_prompt
            
            for iteration in range(max_iterations):
                iter_start = time.time()
                logger.info(f"Iteration {iteration+1}/{max_iterations}: Contraction and validation cycle")
                iteration_data = {
                    "iteration": iteration + 1,
                    "contraction": None,
                    "validation": None,
                    "solved_subproblems": [],
                    "updated_dag": None
                }
                
                # Step 4a: Contract the problem
                contraction = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="contract",
                    prompt_id=prompt_id,
                    task_prompt=current_problem,
                    model=model,
                    temperature=0.4,
                    return_full_response=False
                )
                iteration_data["contraction"] = contraction
                logger.debug(f"Contraction result: {contraction[:100]}..." if len(contraction) > 100 else contraction)
                
                # Step 4b: Validate the contraction
                validation = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="reasoning",
                    action="validate",
                    prompt_id=prompt_id,
                    task_prompt=current_problem,
                    contraction=contraction,
                    model=model,
                    temperature=0.3,
                    return_full_response=False
                )
                iteration_data["validation"] = validation
                logger.debug(f"Validation result: {validation[:100]}..." if len(validation) > 100 else validation)
                
                # Step 4c: Opportunistically solve subproblems
                # Identify solvable subproblems in this iteration
                solvable_subproblems = []
                
                for node_id in execution_order:
                    if node_id not in execution_data["subproblem_solutions"]:
                        node = execution_data["dag"]["nodes"].get(node_id)
                        if node and node["status"] == "pending":
                            # Check if all dependencies are solved
                            dependencies_solved = True
                            for dep_id in node["dependencies"]:
                                if dep_id not in execution_data["subproblem_solutions"]:
                                    dependencies_solved = False
                                    break
                            
                            if dependencies_solved:
                                solvable_subproblems.append(node_id)
                
                # Solve identified subproblems
                for node_id in solvable_subproblems:
                    node = execution_data["dag"]["nodes"][node_id]
                    
                    # Prepare context from dependencies
                    dependency_context = ""
                    for dep_id in node["dependencies"]:
                        if dep_id in execution_data["subproblem_solutions"]:
                            dep_node = execution_data["dag"]["nodes"].get(dep_id)
                            if dep_node:
                                dependency_context += f"\nSubproblem {dep_id}: {dep_node['description']}\nSolution: {execution_data['subproblem_solutions'][dep_id]}\n"
                    
                    # Generate solution
                    solution_prompt = f"Solve this subproblem:\n{node['description']}"
                    if dependency_context:
                        solution_prompt += f"\n\nContext from solved dependencies:{dependency_context}"
                    
                    solution = self.generator.get_completion(
                        prompt_id=prompt_id,
                        prompt=solution_prompt,
                        model=model,
                        temperature=temperature,
                        return_full_response=False
                    )
                    
                    # Store solution
                    execution_data["subproblem_solutions"][node_id] = solution
                    execution_data["dag"]["nodes"][node_id]["status"] = "solved"
                    execution_data["dag"]["nodes"][node_id]["solution"] = solution
                    iteration_data["solved_subproblems"].append(node_id)
                    
                    logger.debug(f"Solved subproblem {node_id}")
                
                # Step 4d: Check termination condition
                try:
                    termination_data = json.loads(validation) if isinstance(validation, str) else validation
                    can_terminate = termination_data.get("can_terminate", False)
                    termination_reason = termination_data.get("reason", "Unknown")
                    
                    if can_terminate:
                        logger.info(f"Termination condition met at iteration {iteration+1}: {termination_reason}")
                        iteration_data["termination"] = {"terminated": True, "reason": termination_reason}
                        execution_data["iterations"].append(iteration_data)
                        break
                except (json.JSONDecodeError, TypeError):
                    # Continue if validation parsing fails
                    pass
                
                # Update for next iteration
                current_problem = contraction
                iteration_data["updated_dag"] = {
                    "nodes_count": len(execution_data["dag"]["nodes"]),
                    "edges_count": len(execution_data["dag"]["edges"]),
                    "solved_count": len(execution_data["subproblem_solutions"])
                }
                
                execution_data["iterations"].append(iteration_data)
                execution_data["performance_metrics"]["step_timings"][f"iteration_{iteration+1}"] = time.time() - iter_start
                
                # Check if all subproblems are solved
                if len(execution_data["subproblem_solutions"]) == len(execution_data["dag"]["nodes"]):
                    logger.info("All subproblems solved, proceeding to final synthesis")
                    break
            
            # Step 5: Final synthesis with hierarchical reconstruction
            synthesis_start = time.time()
            logger.info("Step 5: Performing final synthesis with hierarchical reconstruction")
            
            # Prepare synthesis context
            synthesis_context = f"Original problem:\n{task_prompt}\n\nSolutions to subproblems:\n"
            
            # Add solutions in hierarchical order
            def add_hierarchical_solutions(group_id, indent=""):
                nonlocal synthesis_context
                if group_id in execution_data["dag"]["hierarchical_groups"]:
                    group = execution_data["dag"]["hierarchical_groups"][group_id]
                    synthesis_context += f"\n{indent}Group {group_id}: {group['description']}\n"
                    
                    # Add child groups first (depth-first)
                    for child_id in group["children"]:
                        add_hierarchical_solutions(child_id, indent + "  ")
            
            # Start with top-level groups
            top_groups = [group_id for group_id, group in execution_data["dag"]["hierarchical_groups"].items() 
                         if not group["parent"]]
            
            for group_id in top_groups:
                add_hierarchical_solutions(group_id)
            
            # Add all atomic solutions
            synthesis_context += "\nAtomic subproblem solutions:\n"
            for node_id, solution in execution_data["subproblem_solutions"].items():
                node = execution_data["dag"]["nodes"].get(node_id)
                if node:
                    synthesis_context += f"\nSubproblem {node_id}: {node['description']}\nSolution: {solution}\n"
            
            synthesis_context += "\nBased on all the above solutions to subproblems, provide a comprehensive solution to the original problem."
            
            # Generate final solution
            final_solution = self.generator.get_completion(
                prompt_id=prompt_id,
                prompt=synthesis_context,
                model=model,
                temperature=max(0.2, temperature - 0.2),  # Slightly lower temperature for synthesis
                return_full_response=False
            )
            
            execution_data["performance_metrics"]["step_timings"]["synthesis"] = time.time() - synthesis_start
            execution_data["performance_metrics"]["total_time"] = time.time() - start_time
            
            logger.info(f"Adaptive DAG reasoning completed in {execution_data['performance_metrics']['total_time']:.2f} seconds")
            
            if return_full_response:
                return {
                    "final_solution": final_solution,
                    **execution_data
                }
            else:
                return final_solution
            
        except Exception as e:
            logger.error(f"Error in adaptive_dag_reasoning: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Attempt fallback solution if possible
            if execution_data["subproblem_solutions"]:
                logger.info("Attempting fallback solution using solved subproblems")
                fallback_context = f"Original problem:\n{task_prompt}\n\nPartial solutions:\n"
                
                for node_id, solution in execution_data["subproblem_solutions"].items():
                    node = execution_data["dag"]["nodes"].get(node_id, {})
                    fallback_context += f"\nSubproblem: {node.get('description', node_id)}\nSolution: {solution}\n"
                
                fallback_context += "\nBased on these partial solutions, provide the best possible answer to the original problem."
                
                try:
                    fallback_solution = self.generator.get_completion(
                        prompt_id=prompt_id,
                        prompt=fallback_context,
                        model=model,
                        temperature=temperature,
                        return_full_response=False
                    )
                    
                    if return_full_response:
                        return {
                            "final_solution": fallback_solution,
                            "error": str(e),
                            "note": "Fallback solution generated due to error",
                            **execution_data
                        }
                    else:
                        return fallback_solution
                except Exception as fallback_error:
                    logger.error(f"Fallback solution also failed: {fallback_error}")
            
            # If all else fails
            if return_full_response:
                return {
                    "final_solution": f"Error occurred during adaptive DAG reasoning: {e}",
                    "error": str(e),
                    **execution_data
                }
            else:
                return f"Error occurred during adaptive DAG reasoning: {e}"
    
    def recursive_chain_of_thought(self,
        task_prompt: str,
        prompt_id: int,
        num_recursive_steps: int = 3,
        reasoning_depth: int = 2,
        model: Optional[str] = "Qwen2.5-1.5B",
        temperature: Optional[float] = 0.7,
        return_full_response: Optional[bool] = False,
        **kwargs
        ) -> Union[str, Dict[str, Any]]:
        """
        Topology 13: Recursive Chain-of-Thought - Generate step-by-step reasoning, then recursively improve by analyzing and fixing flaws in previous reasoning.
        
        Process:
        1. Generate initial chain-of-thought reasoning for the task prompt.
        2. For each recursive step:
           a. Analyze the previous reasoning to identify logical flaws or gaps
           b. Generate improved reasoning that addresses the identified issues
           c. Increase depth by exploring sub-problems and edge cases
        3. Generate final response based on the most refined reasoning chain.
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            num_recursive_steps: Number of recursive improvement iterations (default: 3)
            reasoning_depth: How deeply to explore sub-problems in later iterations (default: 2)
            model: Model to use for generation
            temperature: Temperature for generation
            return_full_response: If True, returns a dictionary with all intermediate reasoning steps
            **kwargs: Additional arguments for generation
            
        Returns:
            If return_full_response is False:
                Final response after recursive reasoning as a string
            If return_full_response is True:
                Dictionary containing:
                - 'final_response': Final response
                - 'reasoning_chains': List of all reasoning chains generated
                - 'analyses': List of analyses of each reasoning chain
                - 'improvement_metrics': Estimated improvement metrics between iterations
        """
        start_time = time.time()
        logger.info(f"Starting recursive_chain_of_thought with {num_recursive_steps} recursive steps and reasoning depth {reasoning_depth}")
        logger.debug(f"Parameters: model={model}, temperature={temperature}")
        
        # Track performance metrics
        metrics = {
            "step_timings": {},
            "response_lengths": {},
            "improvement_metrics": []
        }
        
        try:
            # Track all reasoning chains and analyses
            reasoning_chains = []
            analyses = []
            
            # Step 1: Generate initial chain-of-thought reasoning
            step1_start = time.time()
            logger.info("Step 1: Generating initial chain-of-thought reasoning")
            initial_prompt = f"Task: {task_prompt}\n\nPlease solve this step-by-step, showing your reasoning process."
            
            logger.debug(f"Initial prompt: {initial_prompt[:150]}..." if len(initial_prompt) > 150 else initial_prompt)
            current_reasoning = self.generator.get_completion(
                prompt_id=prompt_id,
                prompt=initial_prompt,
                model=model,
                temperature=temperature,
                return_full_response=False
            )
            reasoning_chains.append(current_reasoning)
            
            reasoning_summary = current_reasoning[:150] + "..." if len(current_reasoning) > 150 else current_reasoning
            logger.debug(f"Initial reasoning summary: {reasoning_summary}")
            
            metrics["step_timings"]["initial_reasoning"] = time.time() - step1_start
            metrics["response_lengths"]["initial_reasoning"] = len(current_reasoning)
            logger.info(f"Generated initial reasoning ({len(current_reasoning)} chars) in {metrics['step_timings']['initial_reasoning']:.2f} seconds")
            
            # Step 2: Recursively improve reasoning
            logger.info(f"Step 2: Starting {num_recursive_steps} recursive improvement iterations")
            for i in range(num_recursive_steps):
                iter_start = time.time()
                logger.info(f"Iteration {i+1}/{num_recursive_steps}: Analyzing and improving reasoning")
                
                # Step 2a: Analyze previous reasoning
                analysis_start = time.time()
                logger.debug(f"Step 2a: Analyzing previous reasoning (iteration {i+1})")
                analyze_prompt = f"Task: {task_prompt}\n\nPrevious reasoning:\n{current_reasoning}\n\nAnalyze the above reasoning for logical flaws, gaps, or areas that could be improved. Be specific and detailed in your analysis."
                
                analysis_temp = max(0.2, temperature - 0.1)  # Slightly lower temperature for analysis
                logger.debug(f"Analysis temperature: {analysis_temp}")
                
                analysis = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=analyze_prompt,
                    model=model,
                    temperature=analysis_temp,
                    return_full_response=False
                )
                analyses.append(analysis)
                
                analysis_summary = analysis[:150] + "..." if len(analysis) > 150 else analysis
                logger.debug(f"Analysis {i+1} summary: {analysis_summary}")
                
                analysis_time = time.time() - analysis_start
                metrics["step_timings"][f"analysis_{i+1}"] = analysis_time
                metrics["response_lengths"][f"analysis_{i+1}"] = len(analysis)
                logger.info(f"Generated analysis {i+1} ({len(analysis)} chars) in {analysis_time:.2f} seconds")
                
                # Step 2b: Generate improved reasoning
                improve_start = time.time()
                logger.debug(f"Step 2b: Generating improved reasoning (iteration {i+1})")
                
                # Increase reasoning depth in later iterations
                current_depth = min(i + 1, reasoning_depth)
                depth_instruction = ""
                if current_depth > 1:
                    depth_instruction = f"\n\nIn this iteration, explore the problem more deeply. Consider {current_depth} levels of sub-problems and edge cases that weren't addressed in previous reasoning."
                    logger.debug(f"Increasing reasoning depth to {current_depth} for iteration {i+1}")
                
                improve_prompt = f"Task: {task_prompt}\n\nPrevious reasoning:\n{current_reasoning}\n\nAnalysis of flaws and gaps:\n{analysis}\n\nProvide an improved, more robust reasoning process that addresses the identified issues and solves the task.{depth_instruction}"
                
                improved_reasoning = self.generator.get_completion(
                    prompt_id=prompt_id,
                    prompt=improve_prompt,
                    model=model,
                    temperature=temperature,
                    return_full_response=False
                )
                reasoning_chains.append(improved_reasoning)
                
                improved_summary = improved_reasoning[:150] + "..." if len(improved_reasoning) > 150 else improved_reasoning
                logger.debug(f"Improved reasoning {i+1} summary: {improved_summary}")
                
                improve_time = time.time() - improve_start
                metrics["step_timings"][f"improvement_{i+1}"] = improve_time
                metrics["response_lengths"][f"improvement_{i+1}"] = len(improved_reasoning)
                logger.info(f"Generated improved reasoning {i+1} ({len(improved_reasoning)} chars) in {improve_time:.2f} seconds")
                
                # Calculate improvement metrics
                prev_length = len(current_reasoning)
                new_length = len(improved_reasoning)
                length_change = new_length - prev_length
                improvement_ratio = (new_length - prev_length) / prev_length if prev_length > 0 else 0
                
                improvement_metric = {
                    "iteration": i + 1,
                    "length_change": length_change,
                    "length_change_ratio": improvement_ratio,
                    "previous_length": prev_length,
                    "new_length": new_length
                }
                
                metrics["improvement_metrics"].append(improvement_metric)
                logger.info(f"Iteration {i+1} improvement: Length change {length_change} chars ({improvement_ratio:.2%} change)")
                
                # Update current reasoning for next iteration
                current_reasoning = improved_reasoning
                
                iter_time = time.time() - iter_start
                metrics["step_timings"][f"full_iteration_{i+1}"] = iter_time
                logger.info(f"Completed iteration {i+1} in {iter_time:.2f} seconds")
            
            # Step 3: Generate final response based on most refined reasoning
            final_start = time.time()
            logger.info("Step 3: Generating final response based on most refined reasoning")
            final_prompt = f"Task: {task_prompt}\n\nDetailed reasoning:\n{current_reasoning}\n\nBased on this reasoning, provide a clear, concise, and well-structured final answer to the task."
            
            final_temp = max(0.2, temperature - 0.2)  # Lower temperature for final answer
            logger.debug(f"Final response temperature: {final_temp}")
            
            final_response = self.generator.get_completion(
                prompt_id=prompt_id,
                prompt=final_prompt,
                model=model,
                temperature=final_temp,
                return_full_response=False
            )
            
            final_summary = final_response[:150] + "..." if len(final_response) > 150 else final_response
            logger.debug(f"Final response summary: {final_summary}")
            
            final_time = time.time() - final_start
            metrics["step_timings"]["final_response"] = final_time
            metrics["response_lengths"]["final_response"] = len(final_response)
            
            # Calculate overall metrics
            total_time = time.time() - start_time
            metrics["total_execution_time"] = total_time
            metrics["num_recursive_steps_completed"] = num_recursive_steps
            metrics["max_reasoning_depth"] = min(num_recursive_steps, reasoning_depth)
            
            logger.info(f"Generated final response ({len(final_response)} chars) in {final_time:.2f} seconds")
            logger.info(f"Total recursive_chain_of_thought execution time: {total_time:.2f} seconds")
            
            # Return appropriate response based on return_full_response flag
            if return_full_response:
                return {
                    "final_response": final_response,
                    "reasoning_chains": reasoning_chains,
                    "analyses": analyses,
                    "improvement_metrics": metrics["improvement_metrics"],
                    "performance_metrics": metrics
                }
            else:
                return final_response
        
        except Exception as e:
            # Log comprehensive error details
            total_time = time.time() - start_time
            logger.error(f"Error in recursive_chain_of_thought after {total_time:.2f} seconds: {e}")
            logger.error(f"Error details: {type(e).__name__}, {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
    def ensemble_weighted_voting(self,
        task_prompt: str,
        prompt_id: int,
        models: List[str] = ["Qwen2.5-1.5B"],
        weights: Optional[List[float]] = None,
        num_responses_per_model: int = 3,
        consistency_threshold: float = 0.7,
        temperature: Optional[float] = 0.7,
        return_full_response: Optional[bool] = False,
        **kwargs
        ) -> Union[str, Dict[str, Any]]:
        """
        Topology 14: Ensemble Weighted Voting - Generate responses from multiple models or with varied parameters, 
        assess consistency between responses, and produce a final output through weighted voting or synthesis.
        
        Process:
        1. Generate multiple responses from each specified model (or same model with varied parameters)
        2. Evaluate the consistency and quality of responses from each source
        3. Assign or adjust weights based on consistency scores
        4. Apply weighted voting to select the final response or synthesize a new response
        5. If consistency is below threshold, apply additional refinement
        
        Args:
            task_prompt: Original task prompt
            prompt_id: Prompt ID for tracking
            models: List of models to use for generation (default: ["Qwen2.5-1.5B"])
            weights: Optional list of weights for each model (default: equal weights)
            num_responses_per_model: Number of responses to generate per model (default: 3)
            consistency_threshold: Threshold for consistency score (default: 0.7)
            temperature: Temperature for generation
            return_full_response: If True, returns a dictionary with full response details
            **kwargs: Additional arguments for generation
            
        Returns:
            If return_full_response is False:
                Final selected or synthesized response as a string
            If return_full_response is True:
                Dictionary containing:
                - 'final_response': Final selected or synthesized response
                - 'model_responses': Dictionary mapping model names to lists of responses
                - 'consistency_scores': Dictionary mapping model names to consistency scores
                - 'weights': Dictionary mapping model names to final weights used
                - 'synthesis_method': Method used for final synthesis
        """
        start_time = time.time()
        logger.info(f"Starting ensemble_weighted_voting with {len(models)} models, {num_responses_per_model} responses per model")
        logger.debug(f"Models: {models}")
        logger.debug(f"Parameters: consistency_threshold={consistency_threshold}, temperature={temperature}")
        
        # Track performance metrics
        metrics = {
            "step_timings": {},
            "response_counts": {},
            "consistency_metrics": {}
        }

        try:
            # Initialize weights if not provided
            weights_start = time.time()
            if weights is None or len(weights) != len(models):
                weights = [1.0 / len(models)] * len(models)
                logger.debug(f"Initializing equal weights: {weights}")
            
            # Ensure weights sum to 1
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
            logger.info(f"Initial weights for models: {dict(zip(models, weights))}")
            metrics["step_timings"]["weights_initialization"] = time.time() - weights_start
            
            # Step 1: Generate responses from each model
            logger.info("Step 1: Generating responses from each model")
            generation_start = time.time()
            model_responses = {}
            total_responses = 0
            
            for i, model_name in enumerate(models):
                model_gen_start = time.time()
                logger.info(f"Generating {num_responses_per_model} responses from model {model_name} ({i+1}/{len(models)})")
                responses = []
                
                for j in range(num_responses_per_model):
                    response_start = time.time()
                    response = self.generator.get_completion(
                        prompt_id=prompt_id,
                        prompt=task_prompt,
                        model=model_name,
                        temperature=temperature,
                        return_full_response=False
                    )
                    responses.append(response)
                    total_responses += 1
                    
                    response_time = time.time() - response_start
                    metrics["step_timings"][f"response_gen_{model_name}_{j+1}"] = response_time
                    
                    response_summary = response[:100] + "..." if len(response) > 100 else response
                    logger.debug(f"Generated response {j+1}/{num_responses_per_model} from {model_name} ({len(response)} chars) in {response_time:.2f}s: {response_summary}")
                
                model_responses[model_name] = responses
                model_gen_time = time.time() - model_gen_start
                metrics["step_timings"][f"model_generation_{model_name}"] = model_gen_time
                logger.info(f"Generated all responses for {model_name} in {model_gen_time:.2f} seconds")
            
            generation_time = time.time() - generation_start
            metrics["step_timings"]["total_generation"] = generation_time
            metrics["response_counts"]["total_responses"] = total_responses
            logger.info(f"Generated {total_responses} total responses in {generation_time:.2f} seconds")
            
            # Step 2: Evaluate consistency between responses for each model
            consistency_start = time.time()
            logger.info("Step 2: Evaluating consistency between responses")
            consistency_scores = {}
            
            for model_name, responses in model_responses.items():
                model_consistency_start = time.time()
                # Simple consistency measurement: use pairwise similarity
                consistency_score = self._calculate_consistency_score(responses)
                consistency_scores[model_name] = consistency_score
                
                model_consistency_time = time.time() - model_consistency_start
                metrics["step_timings"][f"consistency_{model_name}"] = model_consistency_time
                logger.info(f"Consistency score for {model_name}: {consistency_score:.4f} (calculated in {model_consistency_time:.2f}s)")
            
            consistency_time = time.time() - consistency_start
            metrics["step_timings"]["consistency_calculation"] = consistency_time
            metrics["consistency_metrics"]["model_scores"] = consistency_scores
            logger.info(f"Calculated all consistency scores in {consistency_time:.2f} seconds")
            
            # Step 3: Adjust weights based on consistency scores
            weight_adjust_start = time.time()
            logger.info("Step 3: Adjusting weights based on consistency scores")
            adjusted_weights = {}
            total_adjusted_weight = 0
            
            for i, model_name in enumerate(models):
                # Adjust the initial weight by the consistency score
                initial_weight = weights[i]
                consistency = consistency_scores[model_name]
                adjusted_weight = initial_weight * (consistency ** 2)  # Square to emphasize differences
                adjusted_weights[model_name] = adjusted_weight
                total_adjusted_weight += adjusted_weight
                logger.debug(f"Model {model_name}: initial weight {initial_weight:.4f}, consistency {consistency:.4f}, adjusted weight (pre-norm) {adjusted_weight:.4f}")
            
            # Normalize adjusted weights
            if total_adjusted_weight > 0:
                for model_name in adjusted_weights:
                    adjusted_weights[model_name] /= total_adjusted_weight
                    logger.debug(f"Model {model_name}: normalized weight {adjusted_weights[model_name]:.4f}")
            else:
                # Fallback to original weights if something went wrong
                logger.warning(f"Total adjusted weight is {total_adjusted_weight} (≤0), falling back to original weights")
                for i, model_name in enumerate(models):
                    adjusted_weights[model_name] = weights[i]
            
            weight_adjust_time = time.time() - weight_adjust_start
            metrics["step_timings"]["weight_adjustment"] = weight_adjust_time
            metrics["consistency_metrics"]["adjusted_weights"] = adjusted_weights
            logger.info(f"Adjusted weights: {adjusted_weights}")
            logger.info(f"Completed weight adjustment in {weight_adjust_time:.2f} seconds")
            
            # Step 4: Apply weighted voting or synthesis
            synthesis_start = time.time()
            overall_consistency = sum(consistency_scores.values()) / len(consistency_scores)
            metrics["consistency_metrics"]["overall_consistency"] = overall_consistency
            logger.info(f"Step 4: Applying synthesis with overall consistency {overall_consistency:.4f} (threshold: {consistency_threshold})")
            
            synthesis_method = "weighted_selection"
            final_response = ""
            
            if overall_consistency >= consistency_threshold:
                # High consistency - use weighted selection to pick the best response
                logger.info(f"Using weighted selection due to high consistency ({overall_consistency:.4f} ≥ {consistency_threshold})")
                all_responses = []
                all_weights = []
                model_indices = {}
                current_idx = 0
                
                for model_name, responses in model_responses.items():
                    model_indices[model_name] = list(range(current_idx, current_idx + len(responses)))
                    current_idx += len(responses)
                    
                    all_responses.extend(responses)
                    response_weight = adjusted_weights[model_name] / len(responses)
                    all_weights.extend([response_weight] * len(responses))
                
                logger.debug(f"Weighted selection pool: {len(all_responses)} responses with weights distribution: min={min(all_weights):.4f}, max={max(all_weights):.4f}")
                
                # Select response based on weighted probability
                selected_idx = self._weighted_selection(all_weights)
                final_response = all_responses[selected_idx]
                
                # Find which model the selected response came from
                selected_model = None
                for model_name, indices in model_indices.items():
                    if selected_idx in indices:
                        selected_model = model_name
                        break
                
                response_summary = final_response[:100] + "..." if len(final_response) > 100 else final_response
                logger.info(f"Selected response index {selected_idx} from model {selected_model}")
                logger.debug(f"Selected response: {response_summary}")
            else:
                # Low consistency - use synthesis to combine insights
                synthesis_method = "meta_synthesis"
                logger.info(f"Using meta-synthesis due to low consistency ({overall_consistency:.4f} < {consistency_threshold})")
                
                all_responses = []
                for model_name, responses in model_responses.items():
                    all_responses.extend(responses)
                
                meta_start = time.time()
                logger.debug(f"Applying meta-synthesis on {len(all_responses)} responses from {len(models)} models")
                final_response = self.metagenerator.get_meta_generation(
                    application="metaprompt",
                    category="synthesis",
                    action="ensemble",
                    prompt_id=prompt_id,
                    task_prompt=task_prompt,
                    responses=all_responses,
                    model=models[0],  # Use first model for synthesis
                    temperature=0.5,
                    return_full_response=False
                )
                
                meta_time = time.time() - meta_start
                metrics["step_timings"]["meta_synthesis"] = meta_time
                response_summary = final_response[:100] + "..." if len(final_response) > 100 else final_response
                logger.info(f"Completed meta-synthesis in {meta_time:.2f} seconds")
                logger.debug(f"Synthesized response: {response_summary}")
            
            synthesis_time = time.time() - synthesis_start
            metrics["step_timings"]["synthesis"] = synthesis_time
            metrics["consistency_metrics"]["synthesis_method"] = synthesis_method
            logger.info(f"Completed {synthesis_method} in {synthesis_time:.2f} seconds")
            
            # Calculate overall execution time and log summary
            total_time = time.time() - start_time
            metrics["total_execution_time"] = total_time
            logger.info(f"Total ensemble_weighted_voting execution time: {total_time:.2f} seconds")
            
            if return_full_response:
                return {
                    "final_response": final_response,
                    "model_responses": model_responses,
                    "consistency_scores": consistency_scores,
                    "weights": adjusted_weights,
                    "overall_consistency": overall_consistency,
                    "synthesis_method": synthesis_method,
                    "performance_metrics": metrics
                }
            else:
                return final_response
                
        except Exception as e:
            # Log comprehensive error details
            total_time = time.time() - start_time
            logger.error(f"Error in ensemble_weighted_voting after {total_time:.2f} seconds: {e}")
            logger.error(f"Error details: {type(e).__name__}, {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def _calculate_consistency_score(self, responses: List[str]) -> float:
        """Helper method to calculate consistency score between responses."""
        if len(responses) <= 1:
            return 1.0  # Perfect consistency for single response
        
        # Use simple string similarity for consistency calculation
        # In a real implementation, you might use embedding similarity or other metrics
        logger.debug(f"Calculating consistency score for {len(responses)} responses")
        
        # Simple character-level Jaccard similarity
        num_pairs = 0
        total_similarity = 0
        similarities = []
        
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                sim = self._jaccard_similarity(responses[i], responses[j])
                similarities.append((i, j, sim))
                total_similarity += sim
                num_pairs += 1
        
        if num_pairs > 0:
            avg_similarity = total_similarity / num_pairs
            # Log min, max, and average similarities for debugging
            if similarities:
                min_sim = min(similarities, key=lambda x: x[2])
                max_sim = max(similarities, key=lambda x: x[2])
                logger.debug(f"Similarity stats: avg={avg_similarity:.4f}, min={min_sim[2]:.4f} (pair {min_sim[0]},{min_sim[1]}), max={max_sim[2]:.4f} (pair {max_sim[0]},{max_sim[1]})")
            return avg_similarity
        else:
            return 1.0
    
    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        """Compute Jaccard similarity between two strings."""
        # Convert to character sets for simple character-level similarity
        set1 = set(str1)
        set2 = set(str2)
        
        if not set1 and not set2:  # Both empty
            return 1.0
                
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def _weighted_selection(self, weights: List[float]) -> int:
        """Select an index based on weighted probability."""
        total = sum(weights)
        if total <= 0:
            logger.warning("Total weight is zero or negative, using uniform selection")
            import random
            return random.randint(0, len(weights) - 1)
            
        normalized_weights = [w / total for w in weights]
        
        # Cumulative distribution
        cumulative = []
        cumsum = 0
        for w in normalized_weights:
            cumsum += w
            cumulative.append(cumsum)
        
        # Random selection
        import random
        r = random.random()  # Random value between 0 and 1
        logger.debug(f"Weighted selection with random value: {r:.4f}")
        
        for i, cum_prob in enumerate(cumulative):
            if r <= cum_prob:
                return i
        
        logger.warning("Weighted selection fell through, returning last index")
        return len(weights) - 1  # Fallback to last index if something went wrong
