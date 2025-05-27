import os
import io
import base64
import requests
from pdf2image import convert_from_path
from dotenv import load_dotenv
import json
import math
import numpy as np
import pandas as pd
import re
import string
from typing import List, Optional, Dict, Any, Union, Tuple
import uuid
from pathlib import Path
import torch
import torch.nn.functional as F
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from openai import AzureOpenAI
from azure.identity import ClientSecretCredential
import google.generativeai as genai
from google.generativeai import types
from anthropic import Anthropic
from mistralai import Mistral
from src.pipeline.shared.logging import get_logger
from src.pipeline.shared.utility import DataUtility, StatisticsUtility, AIUtility, MemoryUtility

logger = get_logger(__name__)

class Generator:
    """Handles model interactions and text generation.
    Provides token management, Azure OpenAI API and HuggingFace model integrations.
    """
    def __init__(self):
        """Initialize Generator."""
        # Initialize utilities
        self.datautility = DataUtility()
        self.statsutility = StatisticsUtility()
        self.memoryutility = MemoryUtility()
        
        # Load configurations
        self.hf_model_dir = Path.cwd() / "model"
        self.config_dir = Path.cwd() / "config"
        
        # Default embedding dimensions for different model providers
        # These will be used as fallbacks if dimensions cannot be determined dynamically
        self.default_embedding_dimensions = {
            "azure_openai": 1536,
            "anthropic": 1024,
            "vertex": 768,
            "huggingface": 384
        }
        
        # Cache for model embedding dimensions
        self.embedding_dimensions_cache = {}
        
        # Load Main Configurations
        try:
            self.main_config_path  = self.config_dir / "main_config.json"
            self.main_config  = self.datautility.text_operation('load', self.main_config_path, file_type='json')
        except Exception as e:
            logger.error(f"Failed to load main configuration: {e}")
            raise

        # Load Model Configurations
        try:
            self.model_config_path = self.config_dir / "model_config.json"
            self.model_config = self.datautility.text_operation('load', self.model_config_path, file_type='json')
            
            # Load embedding dimensions from config if available
            # if "embedding_dimensions" in self.model_config:
            #     self.default_embedding_dimensions.update(self.model_config["embedding_dimensions"])
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            raise
        
        # Default Model Settings
        self.default_embedding_model = self.model_config["validation_rules"]["models"]["embedding"]["azure_openai"][0]
        self.default_completion_model = self.model_config["validation_rules"]["models"]["completion"]["azure_openai"][0]
        self.default_hf_embedding_model = self.model_config["validation_rules"]["models"]["embedding"]["huggingface"][0]
        self.default_hf_completion_model = self.model_config["validation_rules"]["models"]["completion"]["huggingface"][0]
        self.default_hf_reranker_model = self.model_config["validation_rules"]["models"]["reranker"]["huggingface"][0]
        self.default_hf_ocr_model = self.model_config["validation_rules"]["models"]["ocr"]["huggingface"][0]
        self.default_hf_tokeniser_model = self.model_config["validation_rules"]["models"]["tokeniser"]["huggingface"][0]
        self.default_max_attempts = self.main_config["system"]["api"]["max_attempts"]
        self.default_wait_time = self.main_config["system"]["api"]["wait_time"]
        
        # Azure OpenAI settings
        load_dotenv()
        self.scope = os.getenv("SCOPE")
        self.tenant_id = os.getenv("TENANT_ID")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.subscription_key = os.getenv("SUBSCRIPTION_KEY")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        # Google Gemini settings
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.Client(api_key=self.gemini_api_key)
            
        # Anthropic Claude settings
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        
        # Mistral settings
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")

        logger.debug(f"Generator initialized with default models:\n- Embedding Model: {self.default_embedding_model}\n- Completion Model: {self.default_completion_model}\n- HuggingFace Embedding Model: {self.default_hf_embedding_model}\n- HuggingFace Completion Model: {self.default_hf_completion_model}\n- HuggingFace OCR Model: {self.default_hf_ocr_model}")

    def _validate_model(self, model_name: str, model_type: str, provider: str = None) -> bool:
        """Validate model name against configuration.

        Parameters:
            model_name (str): Name of the model to validate
            model_type (str): Type of model (completion, embedding, ocr)
            provider (str): Optional provider specification (azure_openai, huggingface)

        Returns:
            Union[bool, Tuple[bool, str]]: 
                - If provider is specified: returns bool indicating if model is valid
                - If provider is not specified: returns tuple (is_valid, detected_provider)
        """
        try:
            # Check if model type exists
            if model_type not in self.model_config["validation_rules"]["models"]:
                logger.error(f"Invalid model type: {model_type}")
                return False
            
            model_rules = self.model_config["validation_rules"]["models"][model_type]            
            # If provider is specified, check only that provider
            if provider:
                if provider not in model_rules:
                    logger.error(f"Invalid provider: {provider}")
                    return False
                else:
                    return model_name in model_rules[provider]
            else:
                logger.error(f"Model provider not given")
                return False
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False

    def _validate_parameters(self, params: Dict[str, Any], model_type: str) -> bool:
        """Validate model parameters against configuration.

        Parameters:
            params (Dict[str, Any]): Parameters to validate
            model_type (str): Type of model (completion, embedding)

        Returns:
            bool: True if parameters are valid, False otherwise
        """
        try:
            param_rules = self.model_config["validation_rules"]["model_parameters"][model_type]
            
            for param_name, param_value in params.items():
                if param_name in param_rules:
                    rules = param_rules[param_name]
                    if not (rules["min"] <= param_value <= rules["max"]):
                        logger.error(f"Parameter {param_name} value {param_value} outside valid range [{rules['min']}, {rules['max']}]")
                        return False
            return True
        except KeyError:
            logger.error(f"Invalid model type or missing parameter rules: {model_type}")
            return False

    def refresh_token(self) -> str:
        """Refreshes the Azure API token.

        Returns:
            str: The refreshed Azure token
        """
        try:
            # Get token with Azure credentials
            client_credentials = ClientSecretCredential(
                self.tenant_id, 
                self.client_id, 
                self.client_secret
            )
            access_token = client_credentials.get_token(self.scope).token
            logger.info("Successfully refreshed Azure token")
            return access_token

        except Exception as e:
            logger.error("Failed to refresh token: %s", e)
            raise

    def get_completion(self, 
                      prompt_id: int,
                      prompt: str,
                      stored_result: Optional[pd.DataFrame] = None,
                      system_prompt: Optional[str] = None,
                      model: Optional[str] = None,  
                      temperature: Optional[float] = 0.2,
                      max_tokens: Optional[int] = 4000,
                      top_p: Optional[float] = 0.9,
                      top_k: Optional[int] = 50,
                      frequency_penalty: Optional[float] = 1, # only available for OpenAI model
                      presence_penalty: Optional[float] = 1,  # only available for OpenAI model
                      seed: Optional[int] = None,
                      logprobs: Optional[bool] = False,
                      num_beam: Optional[int] = None,
                      json_schema: Optional[Dict[str, Any]] = None,
                      return_full_response: Optional[bool] = False) -> Union[Dict[str, Any], str]:
        """Get text completion using specified model or fall back to HuggingFace model.
        
        Parameters:
            prompt (str): Input prompt or chat messages
            prompt_id (int): Identifier for the prompt
            model (str): Specific model to use (e.g., "gpt-4", "Mistral-7B-v0.2", "gemini-1.5-pro", "claude-3-opus")
                                 If None or invalid, falls back to initialized HuggingFace model
            temperature (float): Sampling temperature (0-1)
            max_tokens (int): Maximum tokens in response
            top_p (float): Nucleus sampling parameter
            top_k (float):
            frequency_penalty (float): Frequency penalty parameter
            presence_penalty (float): Presence penalty parameter
            seed (Optional[int]): Random seed for reproducibility, only available when using azure openai
            logprobs (bool): Whether to return log probabilities
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation
        
        Returns:
            Dict[str, Any]: Completion response containing:
                - prompt_id: Identifier for the prompt
                - prompt: Original prompt
                - response: Generated response
                - perplexity: Perplexity score
                - tokens_in: Number of input tokens
                - tokens_out: Number of output tokens
                - model: Model used
                - seed: Random seed used
                - description: Response description
                - top_p: Top-p value used
                - temperature: Temperature used
        """
        # Initialise a dataframe to store the results
        if stored_result is None:
            stored_result = pd.DataFrame()

        if model is None:
            logger.warning("Model not specified, falling back to the default HuggingFace model")
            model = self.default_hf_completion_model
        
        try:
            if self._validate_model(model, "completion", "azure_openai"):
                try:
                    logger.info(f"Using Azure OpenAI model '{model}' for completion")
                    df_result, final_response = self._get_azure_completion(
                        prompt=prompt,
                        prompt_id=prompt_id,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        logprobs=logprobs,
                        json_schema=json_schema
                    )
                except Exception as e:
                    logger.warning(f"Azure OpenAI completion failed: {e}")
            
            # Try Google Gemini if it's a Gemini model
            elif self._validate_model(model, "completion", "vertex"):
                try:
                    logger.info(f"Using Google Gemini model '{model}' for completion")
                    df_result, final_response = self._get_gemini_completion(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": json_schema,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Google Gemini completion failed: {e}")
            
            # Try Anthropic Claude if it's a Claude model
            elif self._validate_model(model, "completion", "anthropic"):
                try:
                    logger.info(f"Using Anthropic Claude model '{model}' for completion")
                    df_result, final_response = self._get_claude_completion(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                        # no option to restrict json schema output, however, could be prompted. #
                    )
                except Exception as e:
                    logger.warning(f"Anthropic Claude completion failed: {e}")
            
            # Try HuggingFace if it's a HF model
            elif self._validate_model(model, "completion", "huggingface"):
                try:
                    logger.info(f"Using HuggingFace model '{model}' for completion")
                    df_result, final_response = self._get_hf_completion(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        frequency_penalty=frequency_penalty,
                        num_beam=num_beam,
                        # presence_penalty=presence_penalty,
                        # stop=stop,
                        # seed=seed,
                        logprobs=logprobs
                    )
                except Exception as e:
                    logger.warning(f"HuggingFace completion failed: {e}")
            
            else:
                logger.info(f"Using default HuggingFace model '{model}' for completion")
                df_result, final_response = self._get_hf_completion(
                    prompt=prompt,
                    prompt_id=prompt_id,
                    model=self.default_hf_completion_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    num_beam=num_beam, 
                    # presence_penalty=presence_penalty,
                    # stop=stop,
                    # seed=seed,
                    logprobs=logprobs
                )
            
            # Store results as passive control
            stored_result = pd.concat([stored_result, self.datautility.format_conversion(df_result, "dataframe").T], ignore_index=True)
            
            if return_full_response: 
                return df_result
            else: 
                return final_response

        except Exception as e:
            logger.error(f"Completion failed: {str(e)}")
            raise
            
    def _get_azure_completion(self,
                            prompt_id: int,
                            prompt: str, # Union[str, List[Dict[str, str]]],
                            system_prompt: Optional[str] = None,
                            model: Optional[str] = "gpt-4o",  
                            temperature: Optional[float] = 1,
                            max_tokens: Optional[int] = 3000,
                            top_p: Optional[float] = 1,
                            top_k: Optional[float] = 10,
                            frequency_penalty: Optional[float] = 1.1,
                            presence_penalty: Optional[float] = 1,
                            seed: Optional[int] = 100,
                            logprobs: Optional[bool] = False,
                            json_schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """Internal method for Azure OpenAI API completion."""        
        # Process prompt into messages
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Fill in the random seed
        seed = self.statsutility.set_random_seed(min_value = 0, max_value = 100) if seed is None else seed
        
        # Determine response format
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        elif re.search(r'\bJSON\b', prompt, re.IGNORECASE):
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        # Allow a number of attempts when calling API    
        for attempt in range(self.default_max_attempts):
            try:
                # Refresh token
                access_token = self.refresh_token()
                client = AzureOpenAI(
                    api_version=self.api_version, 
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token = access_token
                )

                # Make API call
                response = client.chat.completions.create(
                    engine=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    response_format=response_format,
                    seed=seed,
                    logprobs=logprobs,
                    extra_headers = {
                        'x-correlation-id': str(uuid.uuid4()),
                        'x-subscription-key': self.subscription_key
                    }
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Process JSON response if needed
                if response_format["type"] == "json_object":
                    try:
                        response_text = json.loads(response_text.strip('```json').strip('```'))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        raise ValueError("Invalid JSON response from model")
                
                # Calculate perplexity
                log_probs = response.choices[0].logprobs if logprobs else None
                perplexity = self._calculate_perplexity(log_probs) if log_probs else None
                
                # Prepare result dictionary
                results = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response": response_text,
                    "perplexity": perplexity if perplexity is not None else None,
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens,
                    "model": model,
                    "seed": seed if seed is not None else None,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature
                }
                logger.debug("Successfully got completion from Azure OpenAI")
                return (results, response_text)
                
            except Exception as e:
                logger.warning("Azure attempt %d failed: %s", attempt + 1, e)
                if attempt < self.default_max_attempts - 1:
                    self.refresh_token()
                else:
                    raise
    
    def _get_gemini_completion(self,
                         prompt_id: int,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         model: Optional[str] = "gemini-1.5-pro",
                         temperature: Optional[float] = 0.7,
                         max_tokens: Optional[int] = 2048,
                         top_p: Optional[float] = 0.95,
                         top_k: Optional[int] = 40,
                         json_schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """Get completion from Google Gemini model.
        
        Parameters:
            prompt_id (int): Identifier for the prompt
            prompt (str): Input prompt text
            system_prompt (Optional[str]): System instructions to guide the model
            model (Optional[str]): Google Gemini model to use (gemini-1.5-pro, gemini-1.5-flash, etc.)
            temperature (Optional[float]): Sampling temperature (0-1)
            max_tokens (Optional[int]): Maximum tokens in response
            top_p (Optional[float]): Nucleus sampling parameter
            top_k (Optional[int]): Top-k sampling parameter
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation
            
        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - A dictionary with standardized result keys
                - The generated completion as text
        """
        if not self.gemini_api_key:
            raise ValueError("Google Vertex API key is not set. Please set the GEMINI_API_KEY environment variable.")
        
        try:
            if json_schema:
                client = genai.Client(api_key=self.gemini_api_key)
                # Create the prompt with system instructions if provided
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        response_schema=json_schema,
                    )
                )
                response_format = "json_schema"
            else:
                client = genai.Client(api_key=self.gemini_api_key)
                # Create the prompt with system instructions if provided
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_output_tokens=max_tokens,
                        response_mime_type="text/plain",
                    )
                )
                if re.search(r'\bJSON\b', prompt, re.IGNORECASE):
                    response_format = "json_object"
                else:
                    response_format = "text"
            
            logger.debug(f"Response format: {response_format}")
            logger.info(f"Initiating response parsing")
            if response_format == "json_object":
                response_text = json.loads(response.text)
                
            elif response_format == "json_schema":
                response_text = response.parsed

            elif response_format == "text":
                response_text = response.text
            
            else:
                raise ValueError(f"Unsupported response format: {response_format}")
            
            logger.info(f"Response parsed successfully")
            # TO DO - to include perplexity, statistics and specific output parser for different output type.

            # Create a standardized result dictionary
            result = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response_text,
                "perplexity": None,  # Gemini doesn't provide log probabilities for perplexity calculation
                "tokens_in": None,    # Estimate if available
                "tokens_out": None,  # Estimate if available
                "model": model,
                "seed": None,        # Gemini doesn't provide a seed
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature
            }
            
            logger.debug("Successfully got completion from Google Gemini")
            return (result, response_text)
            
        except Exception as e:
            logger.error(f"Error getting completion from Google Gemini: {e}")
            raise
            
    def _get_claude_completion(self,
                         prompt_id: int,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         model: Optional[str] = "claude-3-opus-20240229",
                         temperature: Optional[float] = 0.7,
                         max_tokens: Optional[int] = 4096,
                         top_p: Optional[float] = 0.95,
                         json_schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """Get completion from Anthropic Claude model.
        
        Parameters:
            prompt_id (int): Identifier for the prompt
            prompt (str): Input prompt text
            system_prompt (Optional[str]): System instructions to guide the model
            model (Optional[str]): Claude model to use (claude-3-opus-20240229, claude-3-sonnet-20240229, etc.)
            temperature (Optional[float]): Sampling temperature (0-1)
            max_tokens (Optional[int]): Maximum tokens in response
            top_p (Optional[float]): Nucleus sampling parameter
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation
            
        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - A dictionary with standardized result keys
                - The generated completion as text
        """
        if not self.claude_api_key:
            raise ValueError("Anthropic Claude API key is not set. Please set the CLAUDE_API_KEY environment variable.")
        
        try:
            # Initialize the Claude client
            client = Anthropic(api_key=self.claude_api_key)
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Send the request to Claude API
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=messages,
                system=system_prompt if system_prompt else None
            )
            
            # Extract the response text
            response_text = response.content
            
            # Calculate usage 
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            # Create a standardized result dictionary
            result = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response_text,
                "perplexity": None,  # Claude doesn't provide log probabilities for perplexity calculation
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "model": model,
                "seed": None,        # Claude doesn't provide a seed
                "top_p": top_p,
                "temperature": temperature
            }
            
            logger.debug("Successfully got completion from Anthropic Claude")
            return (result, response_text)
            
        except Exception as e:
            logger.error(f"Error getting completion from Anthropic Claude: {e}")
            raise

    def _get_hf_completion(self,
                         prompt_id: int,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         model: Optional[str] = None,
                         temperature: Optional[float] = 1,
                         max_tokens: Optional[int] = 2000,
                         top_p: Optional[float] = 1,
                         top_k: Optional[float] = 10,
                         frequency_penalty: Optional[float] = 1.3,
                         num_beam: Optional[int] = None,
                         logprobs: Optional[bool] = None) -> Tuple[Dict[str, Any], str]:
        """Get completion from HuggingFace model."""
        # if system_prompt:
        #     messages = [
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": prompt}
        #     ]
        # else:
        #     messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        else:
            messages = f"User: {prompt}\nAssistant:"
        device = "mps" if torch.mps.is_available() else "cpu"
        orig_model = model
        if model:
            model = self.hf_model_dir / model
        else:
            model = self.hf_model_dir / self.default_hf_completion_model
        
        try:
            # Load the tokenizer and the model
            hf_tokenizer = AutoTokenizer.from_pretrained(model)
    
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token  # or add a new token with add_special_tokens

            # Customised the stop token id
            user_token_ids = hf_tokenizer("User:", add_special_tokens=False).input_ids
            if len(user_token_ids) == 1:
                combined_eos_ids = [hf_tokenizer.eos_token_id] + user_token_ids
            else:
                combined_eos_ids = [hf_tokenizer.eos_token_id] + user_token_ids[1:]
                
            hf_model = AutoModelForCausalLM.from_pretrained(model).to(device)
            # Craft the message template
            # message_template = hf_tokenizer.apply_chat_template(messages,
            #                                                     tokenize=False,
            #                                                     add_generation_prompt=True)
            # Tokenize the message template
            model_inputs = hf_tokenizer(messages, return_tensors="pt", padding=True).to(hf_model.device)
            generation_config = GenerationConfig(
                max_new_tokens = max_tokens,  
                do_sample = True if num_beam is None else False,  # Enable sampling
                num_beams = num_beam if num_beam is not None else 1,  # Use beam search
                temperature=temperature, 
                top_k=top_k,
                top_p=top_p,     
                repetition_penalty = frequency_penalty,  # Penalize repeated tokens #SC to double check
                pad_token_id=hf_tokenizer.pad_token_id,
                bos_token_id=hf_tokenizer.bos_token_id,
                eos_token_id=combined_eos_ids
            )

            # Start model inferencing
            with torch.no_grad():
                generated_ids = hf_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    generation_config=generation_config
                )
                # Create labels from generated ids
                labels = generated_ids.clone()
                # Mask out the prompt tokens when computing the loss and perplexity score
                for i, input_id in enumerate(model_inputs.input_ids):
                    prompt_length = input_id.size(0)
                    labels[i, :prompt_length] = -100    
                outputs = hf_model(generated_ids, labels = labels)
                loss = outputs.loss
                perplexity_score = torch.exp(loss)
            
            # Remove prompt tokens from the generated sequence
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Calculate token counts
            tokens_in  = len(model_inputs.input_ids[0])
            tokens_out = len(generated_ids[0])
        
            results = {
                "prompt_id": prompt_id if prompt_id is not None else None,
                "prompt": prompt,
                "response": response,
                "perplexity": perplexity_score if logprobs else None,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": orig_model,
                "seed": None,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature
            }
            logger.debug("Successfully got completion from HuggingFace model")
            return (results, response)
            
        except Exception as e:
            logger.error(f"Failed to get completion from HuggingFace model: {e}")
            raise
            
    def get_embeddings(self, 
              text: Union[str, List[str]], 
              model: Optional[str] = None,
              output_format: Optional[str] = "Array",
              batch_size: Optional[int] = None,
              max_tokens_per_batch: int = 8000,
              buffer_ratio: float = 0.9):
        """Get embeddings using specified model or fall back to HuggingFace model.

        Parameters:
            text (Union[str, List[str]]): Text(s) to embed
            model (Optional[str]): Specific model to use (e.g., "text-embedding-ada-002", "text-embedding-gecko", "claude-3-embedding")
                              If None or invalid, falls back to initialized HuggingFace model
            output_format (str): Output format, either "Array" or "List"
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                 based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)

        Returns:
            Union[List[List[float]], np.ndarray]: Embedding vectors in the requested format
        """            
        try:
            # Verify HuggingFace fallback model is available
            if not self._validate_model(self.default_hf_embedding_model, "embedding", "huggingface"):
                raise ValueError(f"Fallback HuggingFace model '{self.default_hf_embedding_model}' is not available")

            response = None
            # If specific model provided, try to use it
            if model:
                # Try Azure OpenAI first if it's an Azure model
                if self._validate_model(model, "embedding", "azure_openai"):
                    try:
                        logger.info(f"Using Azure OpenAI model '{model}' for embeddings")
                        # Use batch processing for both single and multiple texts
                        # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                        kwargs = {
                            "text": text,
                            "model": model,
                            "max_tokens_per_batch": max_tokens_per_batch,
                            "buffer_ratio": buffer_ratio
                        }
                        if batch_size is not None:
                            kwargs["batch_size"] = batch_size
                            
                        response = self._get_azure_embeddings(**kwargs)
                        # Ensure response is in a list format for standardization
                        if isinstance(text, str) and not isinstance(response, list):
                            response = [response]
                    except Exception as e:
                        logger.error(f"Azure OpenAI embeddings failed due to error {e}, falling back to HuggingFace default model")
                
                # Try Google Vertex if it's a Vertex model
                elif self._validate_model(model, "embedding", "vertex") and not response:
                    try:
                        logger.info(f"Using Google Vertex model '{model}' for embeddings")
                        # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                        kwargs = {
                            "text": text,
                            "model": model,
                            "max_tokens_per_batch": max_tokens_per_batch,
                            "buffer_ratio": buffer_ratio
                        }
                        if batch_size is not None:
                            kwargs["batch_size"] = batch_size
                            
                        response = self._get_vertex_embeddings(**kwargs)
                        # Ensure response is in a list format for standardization
                        if isinstance(text, str) and not isinstance(response[0], List):
                            response = [response]
                    except Exception as e:
                        logger.error(f"Google Vertex embeddings failed due to error {e}, falling back to HuggingFace default model")
                
                # Try Anthropic if it's an Anthropic model
                elif self._validate_model(model, model_type="embedding", provider="anthropic") and not response:
                    try:
                        logger.info(f"Using Anthropic model '{model}' for embeddings")
                        # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                        kwargs = {
                            "text": text,
                            "model": model,
                            "max_tokens_per_batch": max_tokens_per_batch,
                            "buffer_ratio": buffer_ratio
                        }
                        if batch_size is not None:
                            kwargs["batch_size"] = batch_size
                            
                        response = self._get_anthropic_embeddings(**kwargs)
                        # Ensure response is in a list format for standardization
                        if isinstance(text, str) and not isinstance(response[0], List):
                            response = [response]
                    except Exception as e:
                        logger.error(f"Anthropic embeddings failed due to error {e}, falling back to HuggingFace default model")
                
                # Try HuggingFace if it's a HF model
                elif self._validate_model(model, model_type="embedding", provider="huggingface") and not response:
                    try:
                        logger.info(f"Using HuggingFace model '{model}' for embeddings")
                        # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                        kwargs = {
                            "text": text,
                            "model": model,
                            "max_tokens_per_batch": max_tokens_per_batch,
                            "buffer_ratio": buffer_ratio
                        }
                        if batch_size is not None:
                            kwargs["batch_size"] = batch_size
                            
                        response = self._get_hf_embeddings(**kwargs)
                        # Ensure response is in a list format for standardization
                        if isinstance(text, str) and isinstance(response, np.ndarray) and response.ndim == 1:
                            response = [response.tolist()]
                        elif isinstance(text, str) and isinstance(response, List) and not isinstance(response[0], List):
                            response = [response]
                    except Exception as e:
                        # If model is invalid, log warning and continue to fallback
                        logger.error(f"Specified model '{model}' failed due to error {e}, falling back to HuggingFace default model")
            
            # If no response yet or model not specified, use HuggingFace fallback
            if not response or not model:
                logger.info(f"Using HuggingFace fallback model '{self.default_hf_embedding_model}' for embeddings")
                # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                kwargs = {
                    "text": text,
                    "model": self.default_hf_embedding_model,
                    "max_tokens_per_batch": max_tokens_per_batch,
                    "buffer_ratio": buffer_ratio
                }
                if batch_size is not None:
                    kwargs["batch_size"] = batch_size
                    
                response = self._get_hf_embeddings(**kwargs)
                
                # Ensure response is in a list format for standardization
                if isinstance(text, str) and isinstance(response, np.ndarray) and response.ndim == 1:
                    response = [response.tolist()]
                elif isinstance(text, str) and isinstance(response, List) and not isinstance(response[0], List):
                    response = [response]

            # Format response according to requested output format
            if output_format and output_format.lower() == "array":
                # Convert to numpy array for array format
                if isinstance(response, List):
                    return np.array(response)
                elif isinstance(response, np.ndarray):
                    return response
            else: 
                # Convert to list for list format
                if isinstance(response, List):
                    return response
                else:
                    return response.tolist()


        except Exception as e:
            logger.error(f"Embeddings failed: {str(e)}")
            raise
    
    def _get_azure_embeddings(self, text: Union[str, List[str]], model: str, batch_size: Optional[int] = None, max_tokens_per_batch: int = 8000, buffer_ratio: float = 0.9) -> Union[List[float], List[List[float]]]:
        """Get embeddings using Azure OpenAI API with batch processing support.

        Parameters:
            text (Union[str, List[str]]): Text or list of texts to embed
            model (str): Azure OpenAI model to use
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                     based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s)
        """
        # Handle single text case
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text
        
        # Prepare batching
        try:
            # Use the get_tokenisation method for token counting
            token_counts = [self.get_tokenisation(t, model="cl100k_base") for t in texts]
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, using character count as proxy for tokens")
            token_counts = [len(t) // 4 for t in texts]  # Rough approximation
        
        # Dynamically calculate batch_size if not provided
        if batch_size is None or batch_size <= 0:
            total_texts = len(texts)
            if total_texts > 1:
                total_tokens = sum(token_counts)
                avg_tokens_per_text = total_tokens / total_texts
                dynamic_batch_size = max(1, int((max_tokens_per_batch * buffer_ratio) / avg_tokens_per_text))
                logger.info(f"Dynamically calculated batch_size: {dynamic_batch_size} (avg tokens/text={avg_tokens_per_text:.2f}, buffer_ratio={buffer_ratio})")
                batch_size = dynamic_batch_size
            else:
                batch_size = 1
        
        # Create batches
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, (t, tokens) in enumerate(zip(texts, token_counts)):
            if tokens > max_tokens_per_batch:
                logger.warning(f"Text at index {idx} exceeds max_tokens_per_batch ({tokens} > {max_tokens_per_batch}), truncating")
                # Truncate or handle oversized text
                continue
                
            if (current_tokens + tokens > max_tokens_per_batch) or (len(current_batch) >= batch_size):
                if current_batch:  # Add the current batch if not empty
                    batches.append(current_batch)
                current_batch = [t]
                current_tokens = tokens
            else:
                current_batch.append(t)
                current_tokens += tokens
        
        if current_batch:  # Add the last batch if not empty
            batches.append(current_batch)
        
        logger.info(f"Processing {len(texts)} texts in {len(batches)} batches")
        
        all_embeddings = []
        
        for batch_idx, batch_texts in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_texts)} texts")
        
            for attempt in range(self.default_max_attempts):
                try:
                    # Refresh token and create client
                    access_token = self.refresh_token()
                    if not access_token:
                        raise ValueError("Failed to refresh Azure AD token")
                        
                    client = AzureOpenAI(
                        api_version=self.api_version,
                        azure_endpoint=self.azure_endpoint,
                        azure_ad_token=access_token
                    )
                    
                    # Get embeddings from Azure OpenAI
                    response = client.embeddings.create(
                        model=model,
                        input=batch_texts,
                        extra_headers={'x-correlation-id': str(uuid.uuid4()), 'x-subscription-key': self.subscription_key}
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt == self.default_max_attempts - 1:
                        logger.error(f"Failed to get embeddings for batch {batch_idx+1} after {self.default_max_attempts} attempts: {e}")
                        # Try fallback to HuggingFace if available
                        try:
                            logger.warning(f"Attempting fallback to HuggingFace model for batch {batch_idx+1}")
                            fallback_embeddings = self._get_hf_embeddings(batch_texts, self.default_hf_embedding_model)
                            if isinstance(fallback_embeddings, np.ndarray):
                                all_embeddings.extend(fallback_embeddings)
                            else:
                                all_embeddings.extend(fallback_embeddings)
                        except Exception as fallback_error:
                            logger.error(f"Fallback to HuggingFace failed: {fallback_error}")
                            raise e  # Re-raise the original error if fallback fails
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for batch {batch_idx+1}, retrying: {e}")
        
        # Return single embedding or list based on input type
        return all_embeddings[0] if is_single_text else all_embeddings

    def _get_vertex_embeddings(self,
                        text: Union[str, List[str]],
                        model: str = "text-embedding-gecko",
                        batch_size: Optional[int] = None,
                        max_tokens_per_batch: int = 8000,
                        buffer_ratio: float = 0.9) -> Union[List[float], List[List[float]]]:
        """Get embeddings using Google Vertex AI embeddings with batch processing.
        
        Parameters:
            text (Union[str, List[str]]): Text(s) to embed
            model (str): Google Vertex embedding model to use
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                      based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)
            
        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s)
        """
        if not self.gemini_api_key:
            raise ValueError("Google Vertex API key is not set. Please set the GEMINI_API_KEY environment variable.")
            
        try:
            # Handle single text case
            is_single_text = isinstance(text, str)
            texts = [text] if is_single_text else text
            
            # Prepare batching
            try:
                token_counts = [self.get_tokenisation(t, model="gemini-2.5-pro") for t in texts]
            except ImportError:
                logger.warning("tiktoken not installed, using character count as proxy for tokens")
                token_counts = [len(t) // 4 for t in texts]  # Rough approximation
            
            # Dynamically calculate batch_size if not provided
            if batch_size is None or batch_size <= 0:
                total_texts = len(texts)
                if total_texts > 1:
                    total_tokens = sum(token_counts)
                    avg_tokens_per_text = total_tokens / total_texts
                    dynamic_batch_size = max(1, int((max_tokens_per_batch * buffer_ratio) / avg_tokens_per_text))
                    logger.info(f"Dynamically calculated batch_size: {dynamic_batch_size} (avg tokens/text={avg_tokens_per_text:.2f}, buffer_ratio={buffer_ratio})")
                    batch_size = dynamic_batch_size
                else:
                    batch_size = 1
            
            # Create batches
            batches = []
            current_batch = []
            current_tokens = 0
            
            for idx, (t, tokens) in enumerate(zip(texts, token_counts)):
                if tokens > max_tokens_per_batch:
                    logger.warning(f"Text at index {idx} exceeds max_tokens_per_batch ({tokens} > {max_tokens_per_batch}), truncating")
                    # Skip oversized text for now
                    continue
                    
                if (current_tokens + tokens > max_tokens_per_batch) or (len(current_batch) >= batch_size):
                    if current_batch:  # Add the current batch if not empty
                        batches.append(current_batch)
                    current_batch = [t]
                    current_tokens = tokens
                else:
                    current_batch.append(t)
                    current_tokens += tokens
            
            if current_batch:  # Add the last batch if not empty
                batches.append(current_batch)
            
            logger.info(f"Processing {len(texts)} texts in {len(batches)} batches using Google Vertex")
            
            all_embeddings = []
            
            # Initialize the embedding model once outside the batch loop
            embedding_model = genai.GenerativeModel(model_name=model)
            
            # Process each batch
            for batch_idx, batch_texts in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_texts)} texts")
                
                batch_embeddings = []
                for single_text in batch_texts:
                    try:
                        # Generate embedding
                        embedding_response = embedding_model.embed_content(
                            model=model,
                            content=single_text
                        )
                        
                        # Extract and store the embedding vector
                        embedding_vector = embedding_response.embedding
                        batch_embeddings.append(embedding_vector)
                    except Exception as e:
                        logger.error(f"Error generating embedding for text in batch {batch_idx+1}: {str(e)}")
                        # Add a zero vector as placeholder for failed embedding
                        # Get the embedding dimension for this model or use a default
                        embedding_dim = self._get_embedding_dimensions(model)
                        batch_embeddings.append([0.0] * embedding_dim)
                
                all_embeddings.extend(batch_embeddings)
            
            # Return a single embedding for a single input, otherwise return list of embeddings
            if is_single_text:
                embedding_dim = self._get_embedding_dimensions(model)
                return all_embeddings[0] if all_embeddings else [0.0] * embedding_dim
            else:
                return all_embeddings
                
        except Exception as e:
            logger.error(f"Error generating Google Vertex embeddings: {str(e)}")
            raise
    
    def _get_anthropic_embeddings(self,
                          text: Union[str, List[str]],
                          model: str = "claude-3-embedding",
                          batch_size: Optional[int] = None,
                          max_tokens_per_batch: int = 8000,
                          buffer_ratio: float = 0.9) -> Union[List[float], List[List[float]]]:
        """Get embeddings using Anthropic Claude embedding model with batch processing.
        
        Parameters:
            text (Union[str, List[str]]): Text(s) to embed
            model (str): Anthropic Claude embedding model to use
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                      based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)
            
        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s)
        """
        if not self.claude_api_key:
            raise ValueError("Anthropic API key is not set. Please set the CLAUDE_API_KEY environment variable.")
            
        try:
            # Initialize the Anthropic client
            client = anthropic.Anthropic(api_key=self.claude_api_key)
            
            # Handle single text case
            is_single_text = isinstance(text, str)
            texts = [text] if is_single_text else text
            
            # Prepare batching
            try:
                token_counts = [self.get_tokenisation(t, model="claude-3-7-sonnet-20250219") for t in texts]
            except ImportError:
                logger.warning("tiktoken not installed, using character count as proxy for tokens")
                token_counts = [len(t) // 4 for t in texts]  # Rough approximation
            
            # Dynamically calculate batch_size if not provided
            if batch_size is None or batch_size <= 0:
                total_texts = len(texts)
                if total_texts > 1:
                    total_tokens = sum(token_counts)
                    avg_tokens_per_text = total_tokens / total_texts
                    dynamic_batch_size = max(1, int((max_tokens_per_batch * buffer_ratio) / avg_tokens_per_text))
                    logger.info(f"Dynamically calculated batch_size: {dynamic_batch_size} (avg tokens/text={avg_tokens_per_text:.2f}, buffer_ratio={buffer_ratio})")
                    batch_size = dynamic_batch_size
                else:
                    batch_size = 1
            
            # Create batches
            batches = []
            current_batch = []
            current_tokens = 0
            
            for idx, (t, tokens) in enumerate(zip(texts, token_counts)):
                if tokens > max_tokens_per_batch:
                    logger.warning(f"Text at index {idx} exceeds max_tokens_per_batch ({tokens} > {max_tokens_per_batch}), truncating")
                    # Skip oversized text for now
                    continue
                    
                if (current_tokens + tokens > max_tokens_per_batch) or (len(current_batch) >= batch_size):
                    if current_batch:  # Add the current batch if not empty
                        batches.append(current_batch)
                    current_batch = [t]
                    current_tokens = tokens
                else:
                    current_batch.append(t)
                    current_tokens += tokens
            
            if current_batch:  # Add the last batch if not empty
                batches.append(current_batch)
            
            logger.info(f"Processing {len(texts)} texts in {len(batches)} batches using Anthropic")
            
            all_embeddings = []
            
            # Process each batch
            for batch_idx, batch_texts in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_texts)} texts")
                
                batch_embeddings = []
                for single_text in batch_texts:
                    try:
                        # Generate embedding
                        embedding_response = client.embeddings.create(
                            model=model,
                            input=single_text
                        )
                        
                        # Extract and store the embedding vector
                        embedding_vector = embedding_response.embeddings[0].embedding
                        batch_embeddings.append(embedding_vector)
                    except Exception as e:
                        logger.error(f"Error generating embedding for text in batch {batch_idx+1}: {str(e)}")
                        # Add a zero vector as placeholder for failed embedding
                        # Get the embedding dimension for this model or use a default
                        embedding_dim = self._get_embedding_dimensions(model)
                        batch_embeddings.append([0.0] * embedding_dim)
                
                all_embeddings.extend(batch_embeddings)
            
            # Return a single embedding for a single input, otherwise return list of embeddings
            if is_single_text:
                embedding_dim = self._get_embedding_dimensions(model)
                return all_embeddings[0] if all_embeddings else [0.0] * embedding_dim
            else:
                return all_embeddings
                
        except Exception as e:
            logger.error(f"Error generating Anthropic embeddings: {str(e)}")
            raise
    
    def _get_hf_embeddings(self,
              text: Union[str, List[str]],
              model: str,
              batch_size: Optional[int] = None,
              max_tokens_per_batch: int = 8000,
              buffer_ratio: float = 0.9) -> Union[List[float], List[List[float]], np.ndarray]:
        """Get embeddings using local HuggingFace model with SentenceTransformer with batch processing.

        Parameters:
            text (Union[str, List[str]]): Text to embed (can be a single string or a list of strings)
            model (str): HuggingFace model to use
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                     based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)

        Returns:
            Union[List[float], List[List[float]], np.ndarray]: Embedding vector(s) with consistent formatting
        """
        # Handle single text case
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text
        
        # Load model from local directory or download
        try:
            if model:
                model_path = self.hf_model_dir / model
                logger.info(f"Loading model from local directory: {model_path}")
            else:
                model_path = self.hf_model_dir / self.default_hf_embedding_model
                logger.info(f"Using default embedding model: {model_path}")
            
            # Load the model
            embed_model = SentenceTransformer(model_path)
            
            # Prepare batching
            try:
                token_counts = [self.get_tokenisation(t, model="Qwen3-7B-Instruct") for t in texts]
            except ImportError:
                logger.warning("tiktoken not installed, using character count as proxy for tokens")
                token_counts = [len(t) // 4 for t in texts]  # Rough approximation
            
            # Dynamically calculate batch_size if not provided
            if batch_size is None or batch_size <= 0:
                total_texts = len(texts)
                if total_texts > 1:
                    total_tokens = sum(token_counts)
                    avg_tokens_per_text = total_tokens / total_texts
                    dynamic_batch_size = max(1, int((max_tokens_per_batch * buffer_ratio) / avg_tokens_per_text))
                    logger.info(f"Dynamically calculated batch_size: {dynamic_batch_size} (avg tokens/text={avg_tokens_per_text:.2f}, buffer_ratio={buffer_ratio})")
                    batch_size = dynamic_batch_size
                else:
                    batch_size = 1
            
            # Create batches
            batches = []
            current_batch = []
            current_tokens = 0
            
            for idx, (t, tokens) in enumerate(zip(texts, token_counts)):
                if tokens > max_tokens_per_batch:
                    logger.warning(f"Text at index {idx} exceeds max_tokens_per_batch ({tokens} > {max_tokens_per_batch}), truncating")
                    # Skip oversized text for now - could implement truncation
                    continue
                    
                if (current_tokens + tokens > max_tokens_per_batch) or (len(current_batch) >= batch_size):
                    if current_batch:  # Add the current batch if not empty
                        batches.append(current_batch)
                    current_batch = [t]
                    current_tokens = tokens
                else:
                    current_batch.append(t)
                    current_tokens += tokens
            
            if current_batch:  # Add the last batch if not empty
                batches.append(current_batch)
            
            logger.info(f"Processing {len(texts)} texts in {len(batches)} batches using {model_path}")
            
            all_embeddings = []
            
            # Process each batch
            for batch_idx, batch_texts in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_texts)} texts")
                try:
                    batch_embeddings = embed_model.encode(batch_texts)
                    if isinstance(batch_embeddings, np.ndarray) and batch_embeddings.ndim == 2:
                        # Multiple embeddings in correct format
                        all_embeddings.extend(batch_embeddings)
                    elif isinstance(batch_embeddings, np.ndarray) and batch_embeddings.ndim == 1:
                        # Single embedding, reshape to 2D
                        all_embeddings.append(batch_embeddings)
                    else:
                        # Handle other formats
                        all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Failed to get embeddings for batch {batch_idx+1}: {e}")
                    # Try with a smaller batch or skip if it fails
                    if len(batch_texts) > 1:
                        logger.warning(f"Retrying batch {batch_idx+1} with individual texts")
                        for single_text in batch_texts:
                            try:
                                single_embedding = embed_model.encode([single_text])
                                all_embeddings.append(single_embedding[0] if isinstance(single_embedding, list) else single_embedding)
                            except Exception as inner_e:
                                logger.error(f"Failed to get embedding for individual text: {inner_e}")
            
            # Convert to numpy array for consistent handling
            all_embeddings = np.array(all_embeddings)
            
            # Return based on input type
            if is_single_text:
                embedding_dim = self._get_embedding_dimensions(model)
                return all_embeddings[0] if len(all_embeddings) > 0 else np.zeros(embedding_dim)  # Return first embedding or zeros
            else:
                return all_embeddings
                
        except Exception as e:
            logger.error(f"Failed to get embeddings using HuggingFace models: {e}")
            # Return zeros as fallback
            embedding_dim = self._get_embedding_dimensions(model)
            if is_single_text:
                return np.zeros(embedding_dim)
            else:
                return np.array([np.zeros(embedding_dim) for _ in range(len(texts))])

    def _get_embedding_dimensions(self, model_name):
        """Get the embedding dimensions for a specific model.
        
        This method attempts to determine the embedding dimensions in the following order:
        1. Check the cache for previously determined dimensions
        2. Check if the model has specific dimensions in the config
        3. For HuggingFace models, try to query the model directly
        4. Use provider-specific defaults
        
        Parameters:
            model_name (str): Name of the embedding model
            
        Returns:
            int: Embedding dimensions for the model
        """
        # Check if we've already determined the dimensions for this model
        if model_name in self.embedding_dimensions_cache:
            return self.embedding_dimensions_cache[model_name]
            
        embedding_dim = None
        
        # Try to determine the provider
        provider = None
        for provider_name in ["azure_openai", "anthropic", "vertex", "huggingface"]:
            if self._validate_model(model_name, "embedding", provider_name):
                provider = provider_name
                break
                
        # Check if the model has specific dimensions in the config
        if "model_dimensions" in self.model_config and model_name in self.model_config["model_dimensions"]:
            embedding_dim = self.model_config["model_dimensions"][model_name]
        
        # For HuggingFace models, we can try to query the model directly
        elif provider == "huggingface":
            try:
                # Try to load the model to get its dimensions
                model_path = Path.cwd() / "local_models" / model_name if os.path.exists(Path.cwd() / "local_models" / model_name) else model_name
                embed_model = SentenceTransformer(model_path)
                # Get dimensions by encoding a test string and checking the output shape
                test_embedding = embed_model.encode("test")
                embedding_dim = len(test_embedding)
                logger.info(f"Determined embedding dimension for {model_name}: {embedding_dim}")
            except Exception as e:
                logger.warning(f"Could not determine embedding dimensions for {model_name}: {e}")
                
        # Use provider-specific defaults if all else fails
        if embedding_dim is None and provider:
            embedding_dim = self.default_embedding_dimensions.get(provider)
            logger.info(f"Using default embedding dimension for {provider}: {embedding_dim}")
        
        # Final fallback
        if embedding_dim is None:
            embedding_dim = 384  # Absolute fallback
            logger.warning(f"Could not determine embedding dimensions for {model_name}, using fallback: {embedding_dim}")
            
        # Cache the result for future use
        self.embedding_dimensions_cache[model_name] = embedding_dim
        return embedding_dim
        
    def get_reranking(self,
                    query: str,
                    passages: List[str],
                    model: Optional[str] = None,
                    batch_size: int = 32,
                    return_scores: bool = True) -> Union[List[Tuple[str, float]], List[str]]:
        """Get reranking scores using cross-encoder models.
        
        Parameters:
            query (str): The query to rank passages against
            passages (List[str]): List of passages to be reranked
            model (Optional[str]): Specific cross-encoder model to use (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")
                                 If None or invalid, falls back to default cross-encoder model
            batch_size (int): Batch size for processing passages
            return_scores (bool): Whether to return scores with passages
            
        Returns:
            Union[List[Tuple[str, float]], List[str]]: Reranked passages with scores (if return_scores=True) or just passages
        """
        if model:
            model = Path.cwd() / "local_models" / model
        else:
            model = Path.cwd() / "local_models" / self.default_hf_reranker_model
            
        # Use MPS if available, else CPU
        device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device for CrossEncoder: {device}")

        try:
            reranker = CrossEncoder(model, device=device)
            logger.info(f"Loaded cross-encoder model: {model} on {device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model '{model}' on {device}: {e}")
            raise
        
        try:
            # Create query-passage pairs for scoring
            query_passage_pairs = [[query, passage] for passage in passages]
            
            # Get scores from the model
            scores = reranker.predict(query_passage_pairs, batch_size=batch_size)
        
            # Combine passages with scores
            passage_score_pairs = list(zip(passages, scores))
        
            # Sort by score in descending order
            reranked_pairs = sorted(passage_score_pairs, key=lambda x: x[1], reverse=True)
        
            if return_scores:
                return reranked_pairs
            else:
                return [pair[0] for pair in reranked_pairs]
                
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise

    # TBC on the needs for a distinct function for calling reasoning models
    def get_reasoning_simulator(self,
                            prompt_id: int,
                            prompt: str,
                            stored_result: Optional[pd.DataFrame] = None,
                            system_prompt: Optional[str] = None,
                            model: Optional[str] = None,  
                            return_full_response: Optional[bool] = False) -> Union[Dict[str, Any], str]:
        """Get reasoning simulation output using a specified model.
            
            Parameters:
                prompt_id (int): Identifier for the prompt.
                prompt (str): The input prompt.
                stored_result (Optional[pd.DataFrame]): DataFrame to store results.
                system_prompt (Optional[str]): Optional system prompt.
                model (Optional[str]): Specific model to use (e.g., "o1", "o3_mini", or a HuggingFace reasoning model identifier).
                return_full_response (Optional[bool]): If True, returns the full response (e.g., DataFrame); otherwise, returns just the response text.
            
            Returns:
                Union[Dict[str, Any], str]: A standardized output dictionary containing:
                    - prompt_id: Identifier for the prompt
                    - prompt: Original prompt
                    - response: Generated reasoning output
                    - perplexity: Perplexity score (if available)
                    - tokens_in: Number of input tokens
                    - tokens_out: Number of output tokens
                    - model: Model used
                    - seed: Random seed used (if applicable)
                    - top_p: Top-p value used (if applicable)
                    - temperature: Temperature used (if applicable)
            """
        # Initialise a dataframe to store the results
        if stored_result is None:
            stored_result = pd.DataFrame()
        
        # Use a default reasoning model if none is specified
        if model is None:
            logger.warning("Model not specified, falling back to the default reasoning model")
            model = self.default_reasoning_model  # assumes this is defined in your class
        
        try:
            # Try using Azure OpenAI reasoning simulator if applicable
            if self._validate_model(model, "reasoning", "azure_openai"):
                try:
                    logger.info(f"Using Azure OpenAI reasoning model '{model}'")
                    df_result, final_response = self._get_azure_reasoning_simulator(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model
                        # Additional reasoning-specific parameters can be added here if needed
                    )
                except Exception as e:
                    logger.warning(f"Azure OpenAI reasoning simulator failed: {e}")
                    # Optionally, one could set a fallback flag here.
                    raise
            # Try using HuggingFace reasoning simulator if validated as such
            elif self._validate_model(model, "reasoning", "huggingface"):
                try:
                    logger.info(f"Using HuggingFace reasoning simulator model '{model}'")
                    df_result, final_response = self._get_hf_reasoning_simulator(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model
                    )
                except Exception as e:
                    logger.warning(f"HuggingFace reasoning simulator failed: {e}")
                    raise
            else:
                # Fall back to default HuggingFace reasoning simulator if model is not explicitly validated
                logger.info(f"Using default reasoning model '{model}'")
                df_result, final_response = self._get_hf_reasoning_simulator(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=self.default_reasoning_model
                )
            
            # Store the result using a utility method to ensure consistent formatting
            stored_result = pd.concat([stored_result, self.datautility.format_conversion(df_result, "dataframe").T], ignore_index=True)
            
            if return_full_response:
                return df_result
            else:
                return final_response
        
        except Exception as e:
            logger.error(f"Reasoning simulator failed: {str(e)}")
            raise
            
    def _get_azure_reasoning_simulator(self,
                                    prompt_id: int,
                                    prompt: str,
                                    system_prompt: Optional[str] = None,
                                    model: Optional[str] = "gpt-4o_reasoning",  
                                    temperature: Optional[float] = 1,
                                    max_tokens: Optional[int] = 3000,
                                    top_p: Optional[float] = 1,
                                    top_k: Optional[float] = 10,
                                    frequency_penalty: Optional[float] = 1.1,
                                    presence_penalty: Optional[float] = 1,
                                    seed: Optional[int] = 100,
                                    logprobs: Optional[bool] = False,
                                    json_schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """
        Internal method for Azure OpenAI reasoning simulation.
        
        Parameters:
            prompt_id (int): Identifier for the prompt.
            prompt (str): The input prompt.
            system_prompt (Optional[str]): Optional system prompt.
            model (Optional[str]): Specific Azure reasoning model.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens in response.
            top_p (float): Nucleus sampling parameter.
            top_k (float): Top-K sampling parameter.
            frequency_penalty (float): Frequency penalty parameter.
            presence_penalty (float): Presence penalty parameter.
            seed (Optional[int]): Random seed for reproducibility.
            logprobs (bool): Whether to return log probabilities.
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation.
        
        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - A dictionary with standardized result keys.
                - The generated reasoning output as text.
        """
        # Process prompt into messages; include system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Set random seed if not provided
        seed = self.statsutility.set_random_seed(min_value=0, max_value=100) if seed is None else seed
        
        # Determine response format based on prompt content
        if json_schema:
            response_format = {"type": "json_schema", "json_schema": json_schema}
        elif re.search(r'\bJSON\b', prompt, re.IGNORECASE):
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}
        
        # Allow a number of attempts when calling the API    
        for attempt in range(self.default_max_attempts):
            try:
                # Refresh token if necessary
                access_token = self.refresh_token()
                client = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token=access_token
                )
                
                # Make API call to the reasoning endpoint (assumed similar to chat completions)
                response = client.chat.completions.create(
                    engine=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    response_format=response_format,
                    seed=seed,
                    logprobs=logprobs,
                    extra_header={
                        'x-correlation-id': str(uuid.uuid4()),
                        'x-subscription-key': self.subscription_key
                    }
                )
                
                # Extract response text; for reasoning tasks, we expect chain-of-thought output
                response_text = response.choices[0].message.content
                
                # Process JSON response if needed
                if response_format["type"] == "json_object":
                    try:
                        response_text = json.loads(response_text.strip('```json').strip('```'))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        raise ValueError("Invalid JSON response from model")
                
                # Calculate perplexity if log probabilities are available
                log_probs = response.choices[0].logprobs if logprobs else None
                perplexity = self._calculate_perplexity(log_probs) if log_probs else None
                
                # Prepare the standardized result dictionary
                results = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response": response_text,
                    "perplexity": perplexity,
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens,
                    "model": model,
                    "seed": seed,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature
                }
                logger.debug("Successfully got reasoning simulation from Azure OpenAI")
                return (results, response_text)
                
            except Exception as e:
                logger.warning("Azure reasoning attempt %d failed: %s", attempt + 1, e)
                if attempt < self.default_max_attempts - 1:
                    self.refresh_token()
                else:
                    raise

    def _get_hf_reasoning_simulator(self,
                                    prompt_id: int,
                                    prompt: str,
                                    system_prompt: Optional[str] = None,
                                    model: Optional[str] = None,
                                    temperature: Optional[float] = 1,
                                    max_tokens: Optional[int] = 2000,
                                    top_p: Optional[float] = 1,
                                    top_k: Optional[float] = 10,
                                    frequency_penalty: Optional[float] = 1.3,
                                    num_beam: Optional[int] = None,
                                    logprobs: Optional[bool] = False) -> Tuple[Dict[str, Any], str]:
        """
        Get reasoning simulation output from a HuggingFace model.
        
        Parameters:
            prompt_id (int): Identifier for the prompt.
            prompt (str): The input prompt.
            system_prompt (Optional[str]): Optional system prompt.
            model (Optional[str]): HuggingFace model identifier or local model name.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens to generate.
            top_p (float): Nucleus sampling parameter.
            top_k (float): Top-K sampling parameter.
            frequency_penalty (float): Penalty to discourage repetition.
            num_beam (Optional[int]): Number of beams for beam search (if any).
            logprobs (bool): Whether to return log probabilities.
        
        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - A standardized dictionary with the reasoning simulation results.
                - The generated reasoning output as text.
        """
        # Construct the message template; include a "Reasoning:" marker to prompt chain-of-thought output
        if system_prompt:
            messages = f"System: {system_prompt}\nUser: {prompt}\nReasoning:"
        else:
            messages = f"User: {prompt}\nReasoning:"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        original_model = model
        if model:
            model_path = os.path.join(self.hf_model_dir, model)
        else:
            model_path = os.path.join(self.hf_model_dir, self.default_reasoning_model)
        
        try:
            # Load tokenizer and model from HuggingFace
            hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token

            # For reasoning tasks, we might want to set a custom end-of-sequence marker
            user_token_ids = hf_tokenizer("User:", add_special_tokens=False).input_ids
            if len(user_token_ids) == 1:
                combined_eos_ids = [hf_tokenizer.eos_token_id] + user_token_ids
            else:
                combined_eos_ids = [hf_tokenizer.eos_token_id] + user_token_ids[1:]
            
            hf_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            model_inputs = hf_tokenizer(messages, return_tensors="pt", padding=True).to(hf_model.device)
            
            # Set up the generation configuration
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=True if num_beam is None else False,
                num_beams=num_beam if num_beam is not None else 1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=frequency_penalty,
                pad_token_id=hf_tokenizer.pad_token_id,
                bos_token_id=hf_tokenizer.bos_token_id,
                eos_token_id=combined_eos_ids
            )
            
            # Generate output from the model
            with torch.no_grad():
                generated_ids = hf_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    generation_config=generation_config
                )
                # Create labels to mask out prompt tokens for loss calculation if needed
                labels = generated_ids.clone()
                for i, input_id in enumerate(model_inputs.input_ids):
                    prompt_length = input_id.size(0)
                    labels[i, :prompt_length] = -100
                outputs = hf_model(generated_ids, labels=labels)
                loss = outputs.loss
                perplexity_score = torch.exp(loss)
            
            # Remove prompt tokens from the generated sequence
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Calculate token counts
            tokens_in = len(model_inputs.input_ids[0])
            tokens_out = len(generated_ids[0])
            
            # Prepare standardized results
            results = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response,
                "perplexity": perplexity_score if logprobs else None,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": original_model,
                "seed": None,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature
            }
            logger.debug("Successfully got reasoning simulation from HuggingFace model")
            return (results, response)
        
        except Exception as e:
            logger.error(f"Failed to get reasoning simulation from HuggingFace model: {e}")
            raise

    def get_ocr(self, 
                pdf_file_path: str, 
                model: Optional[str] = None,
                output_format: str = "markdown") -> None:
        """Get OCR results using specified model or fall back to HuggingFace model.

        Parameters:
            file_path (str): Path to file(s) to process (PDF)
            model (Optional[str]): Specific model to use
                            If None or invalid, falls back to initialized HuggingFace model
            output_format (str): Format to save output ("markdown", "json", "txt", etc.)

        Returns:
            None
        """
        if pdf_file_path:
            # Extract the PDF file name without extension
            pdf_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
            
            # Create the output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(pdf_file_path), "db", "preprocess")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create the output file path with .md extension in the preprocess folder
            output_file = os.path.join(output_dir, f"{pdf_name}.{output_format}")

        try:
            # Verify HuggingFace fallback model is available
            if not self._validate_model(self.default_hf_ocr_model, "ocr", "huggingface"):
                raise ValueError(f"Fallback HuggingFace model '{self.default_hf_ocr_model}' is not available")

            # If specific model provided, try to use it
            if model:
                # Try Mistral first if it's a Mistral model
                if self._validate_model(model, "ocr", "mistral"):
                    try:
                        logger.info(f"Using Mistral model '{model}' for OCR")
                        self._get_mistral_ocr(model=model, 
                                            pdf_file_path=pdf_file_path, 
                                            output_file=output_file,
                                            output_format=output_format)
                        logger.info("Successfully got OCR from Mistral model")
                        return None 
                    except Exception as e:
                        logger.error(f"Mistral OCR failed due to error {e}, falling back to HuggingFace default model")
                
                # Try HuggingFace if it's an HF model
                elif self._validate_model(model, "ocr", "huggingface"):
                    try:
                        logger.info(f"Using HuggingFace model '{model}' for OCR")
                        self._get_hf_ocr(model=model, 
                                        pdf_file_path=pdf_file_path, 
                                        output_file=output_file,
                                        output_format=output_format)
                        logger.info("Successfully got OCR from HuggingFace model")
                        return None
                    except Exception as e:
                        logger.error(f"HuggingFace OCR failed due to error {e}, falling back to HuggingFace default model")

                # Fallback to HuggingFace default model
                else:   
                    try:
                        logger.info(f"Using HuggingFace default model '{self.default_hf_ocr_model}' for OCR")
                        self._get_hf_ocr(model=self.default_hf_ocr_model, 
                                        pdf_file_path=pdf_file_path, 
                                        output_file=output_file,
                                        output_format=output_format)
                        logger.info("Successfully got OCR from HuggingFace default model")
                        return None
                    except Exception as e:
                        logger.error(f"HuggingFace default OCR failed due to error {e}, falling back to HuggingFace default model")
            
        except Exception as e:
            logger.error(f"Completion failed: {str(e)}")
            raise                
                
    def _get_mistral_ocr(self, 
                  model: str = "mistral-ocr-latest", 
                  pdf_file_path: str = None,
                  output_file: str = None,
                  output_format: str = "markdown") -> None:
        """Get OCR results using Mistral OCR API.

        Parameters:
            model (str): Mistral OCR model to use
            pdf_file_path (str): Path to PDF file to process
            output_format (str): Format to save output ("markdown", "json", etc.)

        Returns:
            None
        """
        try:
            client = Mistral(api_key=self.mistral_api_key)
            # Check if API key is available
            if not self.mistral_api_key:
                raise ValueError("Mistral API key not found in environment variables")

            with open(pdf_file_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            
            # Get base64 encoded PDF
            ocr_response = client.ocr.process(
                model = model,
                document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
                include_image_base64=True
            )  
            with open(output_file, 'w') as f:
                f.write(ocr_response.pages)
            logger.info("Successfully got OCR from Mistral model")
            return None
        except Exception as e:
            logger.error(f"Mistral OCR processing failed: {e}")
            raise

    def _get_hf_ocr(self,
                    model: str,
                    pdf_file_path: str,
                    output_file: str) -> None:
        """
        Convert PDF file to text using a specified HuggingFace OCR model.
        The model and tokenizer are loaded once. Pages are processed sequentially.
        Assumes model.chat() takes a tokenizer, an image file path, and ocr_type.

        Parameters:
            model (str): Specific HuggingFace OCR model name to use.
            pdf_file_path (str): Path to the PDF file.
            output_file (str): Path to save the OCR output.

        Raises:
            ValueError: If pdf_file_path is not provided or if the model_name_to_use is invalid.
            Exception: For underlying OCR processing or model loading errors.
        """
        if not pdf_file_path:
            logger.error("PDF file path must be provided for HuggingFace OCR.")
            raise ValueError("PDF file path must be provided")

        if not self._validate_model(model, "ocr", "huggingface"):
            logger.error(f"The HuggingFace OCR model '{model}' is not valid as per configuration.")
            raise ValueError(f"Invalid HuggingFace OCR model specified: {model}")

        logger.info(f"Performing OCR with HuggingFace model: {model} on {pdf_file_path}")

        try:
            # Construct model path (assuming models are in "local_models" subdirectory)
            # Ensure model_name_to_use is just the model's directory name, not a path itself.
            if model:
                model_path = os.path.join(self.hf_model_dir, model)
            else:
                model_path = os.path.join(self.hf_model_dir, self.default_hf_ocr_model)
            logger.debug(f"Loading HuggingFace tokenizer and model from: {model_path}")

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # Set pad_token if not present, common for some models used in generation/chat
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Tokenizer pad_token set to eos_token for {model}")
                
            hf_model_instance = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Useful for large models
                device_map='auto',       # Automatically uses CUDA, MPS, or CPU
                use_safetensors=True,
                # pad_token_id might be set on tokenizer or during generation call if needed
            )
            hf_model_instance = hf_model_instance.eval() # Set model to evaluation mode
            logger.info(f"Successfully loaded HuggingFace model '{model}' and tokenizer.")

            # Convert PDF to a list of PIL Image objects
            logger.info(f"Converting PDF '{pdf_file_path}' to images...")
            images = convert_from_path(pdf_file_path)

            if not images:
                logger.warning(f"No images were extracted from PDF: {pdf_file_path}")
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write("") # Create empty file
                    logger.info(f"Created empty output file at {output_file} as PDF yielded no images.")
                return

            logger.info(f"Extracted {len(images)} page(s) from '{pdf_file_path}'. Processing with OCR...")
            all_page_texts = []
            temp_image_path = None # Define for potential cleanup in case of error during image saving

            for i, pil_image in enumerate(images):
                page_num = i + 1
                logger.debug(f"Processing page {page_num}/{len(images)} with {model}.")
                try:
                    # Save PIL image to a temporary file to pass its path to model.chat()
                    # This step is necessary if model.chat() expects a file path.
                    # If model.chat() can directly process PIL.Image objects, this can be simplified.
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                        pil_image.save(tmp_img_file.name, format='PNG')
                        temp_image_path = tmp_img_file.name
                    
                    # Call the model's chat method (assuming this is the correct API for your OCR model)
                    # The structure `model.chat(tokenizer, image_path, ocr_type="ocr")` is preserved from original.
                    page_text_result = hf_model_instance.chat(tokenizer, temp_image_path, ocr_type="ocr")
                    
                    # Ensure result is a string
                    page_text = page_text_result if isinstance(page_text_result, str) else str(page_text_result)

                    if page_text.strip():
                        if len(images) > 1:
                            all_page_texts.append(f"Page {page_num}:\n{page_text.strip()}\n")
                        else:
                            all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"OCR returned no text for page {page_num} of '{pdf_file_path}' using {model}.")
                        if len(images) > 1: # Add a placeholder for empty pages if desired
                            all_page_texts.append(f"Page {page_num}:\n[No text extracted]\n")
                
                except Exception as page_error:
                    logger.error(f"Error processing page {page_num} of '{pdf_file_path}' with {model}: {page_error}")
                    if len(images) > 1:
                        all_page_texts.append(f"Page {page_num}:\n[Error processing page: {str(page_error)}]\n")
                    else:
                        all_page_texts.append(f"[Error processing document: {str(page_error)}]")
                    # Optionally, re-raise page_error here if one page failure should stop all processing

                finally:
                    # Clean up the temporary image file
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                        temp_image_path = None

            final_text = "\n".join(all_page_texts).strip()

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                logger.info(f"Successfully saved HuggingFace OCR output to {output_file}")
            else:
                logger.warning("No output_file was specified for HuggingFace OCR. Text not saved to disk.")

        except Exception as e:
            logger.error(f"Overall HuggingFace OCR processing failed for model '{model}' on file '{pdf_file_path}': {e}")
            raise


    def _calculate_perplexity(self, logprobs: Dict[str, Any]) -> float:
        """Calculate perplexity score from token log probabilities.
        
        Perplexity is a measurement of how well a probability model predicts a sample.
        Lower perplexity indicates better prediction (lower uncertainty).
        
        Parameters:
            logprobs (Dict[str, Any]): Log probabilities from model response
            
        Returns:
            float: Perplexity score, lower is better
        """
        if not logprobs or not isinstance(logprobs, Dict):
            return None
            
        try:
            # Extract token logprobs if available
            token_logprobs = logprobs.get('token_logprobs', [])
            if not token_logprobs or len(token_logprobs) == 0:
                return None
                
            # Filter out None values that might be in the logprobs
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]
            if not valid_logprobs or len(valid_logprobs) == 0:
                return None
                
            # Calculate average negative log probability
            avg_negative_logprob = -sum(valid_logprobs) / len(valid_logprobs)
            
            # Perplexity is the exponentiation of the average negative log probability
            perplexity = math.exp(avg_negative_logprob)
            
            return perplexity
            
        except Exception as e:
            logger.warning(f"Failed to calculate perplexity: {e}")
            return None
    
    def get_tokenisation(self, text: Union[str, List[str]], model: Optional[str] = None) -> int:
        """Initialize or return cached tiktoken encoder.
        
        Args:
            model: Name of the tokeniser to use. Common options:
                - "o200k_base" (used by GPT-4o)
                - "cl100k_base" (default, used by GPT-4, GPT-3.5)
                - "p50k_base" (used by GPT-3)

        Returns:
            Length of the tokenised text
        """
        # Initialize the encoder
        if model is None:
            logger.warning("Model not specified, falling back to the default HuggingFace model")
            model = self.default_hf_tokeniser_model
        
        try:
            if self._validate_model(model, "tokeniser", "azure_openai"):
                try:
                    logger.info(f"Using Azure OpenAI model '{model}' for tokeniser")
                    access_token = self.refresh_token()
                    client = AzureOpenAI(
                        api_version=self.api_version, 
                        azure_endpoint=self.azure_endpoint,
                        azure_ad_token = access_token
                    )
                    response = tiktoken.get_encoding(model)
                    token_length = len(response.encode(text))
                    return token_length
                except Exception as e:
                    logger.warning(f"Azure OpenAI tokeniser failed: {e}")
            elif self._validate_model(model, "tokeniser", "anthropic"):
                try:
                    logger.info(f"Using Anthropic model '{model}' for tokenisation")
                    response = client.messages.count_tokens(
                        model=model,
                        messages=[{"role": "user", "content": text}]
                    )
                    token_length = response['input_tokens']
                    return token_length
                except Exception as e:
                    logger.warning(f"Anthropic tokenisation failed: {e}")
            elif self._validate_model(model, "tokeniser", "vertex"):
                try:
                    response = client.models.count_tokens(
                        model=model,
                        contents=text
                    )
                    token_length = response
                    return token_length
                except Exception as e:
                    logger.warning(f"Vertex tokeniser failed: {e}")
            elif self._validate_model(model, "tokeniser", "huggingface"):
                try:
                    logger.info(f"Using HuggingFace model '{model}' for tokeniser")
                    tokenizer = AutoTokenizer.from_pretrained(model)
                    token_length = len(tokenizer.tokenize(text))
                    return token_length
                except Exception as e:
                    logger.warning(f"HuggingFace tokeniser failed: {e}")
            else:
                # fall back to default huggingface model
                try:
                    logger.info(f"Using default HuggingFace model '{self.default_hf_tokeniser_model}' for tokeniser")
                    tokenizer = AutoTokenizer.from_pretrained(self.default_hf_tokeniser_model)
                    token_length = len(tokenizer.tokenize(text))
                    return token_length
                except Exception as e:
                    logger.warning(f"Default HuggingFace tokeniser failed: {e}")

        except Exception as e:
            logger.error(f"Invalid model '{model}' for tokeniser")
            raise        

class MetaGenerator:
    def __init__(self, generator = None):
        self.aiutility = AIUtility()
        self.generator = generator if generator else Generator()

    def get_meta_generation(self, 
                           application: str,
                           category: str,
                           action: str,
                           prompt_id: Optional[int] = 100,
                           system_prompt: Optional[str] = None,
                           model: Optional[str] = "Qwen2.5-1.5B",  
                           temperature: Optional[float] = 1,
                           max_tokens: Optional[int] = 1000,
                           top_p: Optional[float] = 1,
                           top_k: Optional[int] = 10,
                           frequency_penalty: Optional[float] = 1, # only available for OpenAI model
                           presence_penalty: Optional[float] = 1,  # only available for OpenAI model
                           seed: Optional[int] = None,
                           logprobs: Optional[bool] = True,
                           num_beam: Optional[int] = None,
                           json_schema: Optional[Dict[str, Any]] = None,
                           return_full_response: Optional[bool] = False,
                           **kwargs) -> Optional[str]:
        """
        Execute a meta-prompt by retrieving the template and filling in the placeholders.
        
        Args:
            application: Application area (e.g., 'metaprompt', 'metaresponse')
            category: Category of the template (e.g., 'manipulation', 'evaluation')
            action: Specific action within the category
            prompt_id: ID for tracking the prompt
            system_prompt: System prompt to guide the model
            model: Model to use for completion
            temperature: Temperature for generation
            **kwargs: Values for template placeholders
            
        Returns:
            Generated response if successful, None otherwise
            
        Raises:
            ValueError: If template is not found or required keys are missing
        """
        
        formatted_prompt = self.aiutility.apply_meta_prompt(application, category, action, **kwargs)
        """
        # Get the meta-prompt template
        template = self.aiutility.get_meta_prompt(application, category, action)
        
        if not template:
            raise ValueError(f"Template not found for {application}/{category}/{action}")
            
        # Parse required keys from template
        formatter = string.Formatter()
        required_keys = {field_name for _, field_name, _, _ in formatter.parse(template) if field_name is not None}
        
        # Check for missing keys
        missing = required_keys - kwargs.keys()
        if missing:
            raise ValueError(f"Missing required keys for meta-prompt: {missing}")
        
        # Check if elements of kwargs are lists and flattern out
        if "task_prompt" in kwargs and kwargs["task_prompt"] is not None:
            if isinstance(kwargs["task_prompt"], list):
                kwargs["task_prompt"] = self.aiutility.format_text_list(kwargs["task_prompt"], "prompt")
            elif isinstance(kwargs["task_prompt"], dict):
                kwargs["task_prompt"] = str(kwargs["task_prompt"])
        if "response" in kwargs and kwargs["response"] is not None:
            if isinstance(kwargs["response"], list):
                kwargs["response"] = self.aiutility.format_text_list(kwargs["response"], "response")
            elif isinstance(kwargs["response"], dict):
                kwargs["response"] = str(kwargs["response"])

        # Format the meta prompt template with **kwargs
        try:
            formatted_prompt = template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Error formatting template: {e}")
        
        logger.debug(f"Formatted meta-prompt: {formatted_prompt}")
        """

        logger.debug(f"Model: {model}")
        # Execute using the generator
        try:
            response = self.generator.get_completion(
                prompt_id=prompt_id,  # Using default prompt ID
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                model=model,
                stored_result = None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty, 
                presence_penalty=presence_penalty,
                seed=seed,
                logprobs=logprobs,
                num_beam=num_beam,
                json_schema=json_schema,
                return_full_response=return_full_response
            )
            return response

        except Exception as e:
            logger.error(f"Error executing meta-prompt: {e}")
            return None




# standardise encoder models
# developing
class Encoder:
    """
    Handles encoding of text using pre-trained models, including bi-encoders, cross-encoders, and rerankers.
    Supports loading models from local directories or using pre-trained models from HuggingFace.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model_type: str = "bi-encoder", local_models_dir: str = None):
        """
        Initialize the Encoder.
        
        Args:
            model_name: Name of the pre-trained model or path to local model
            model_type: Type of encoder model ('bi-encoder', 'cross-encoder', 'reranker')
            local_models_dir: Directory containing local models. If None, uses current working directory / local_models
        """
        self.model_name = model_name
        self.model_type = model_type
        self.local_models_dir = Path(local_models_dir) if local_models_dir else Path.cwd() / "local_models"
        self.model = None
        
        # Load model configuration
        try:
            config_dir = Path.cwd() / "config"
            model_config_path = config_dir / "model_config.json"
            with open(model_config_path, 'r') as f:
                self.model_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            self.model_config = None
        
        # Load the appropriate model based on model_type
        self._load_model()
    
    def _load_model(self):
        """
        Load the appropriate model based on model_type.
        """
        try:
            # Check if model exists locally
            model_path = self.local_models_dir / self.model_name
            use_local = model_path.exists()
            model_source = model_path if use_local else self.model_name
            
            logger.info(f"Loading {self.model_type} model from {'local path' if use_local else 'HuggingFace'}: {model_source}")
            
            if self.model_type == "bi-encoder":
                self.model = SentenceTransformer(model_source)
            elif self.model_type == "cross-encoder":
                self.model = CrossEncoder(model_source)
            elif self.model_type == "reranker":
                # Rerankers are typically implemented as CrossEncoders
                self.model = CrossEncoder(model_source)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            logger.info(f"Successfully loaded {self.model_type} model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_type} model: {e}")
            raise
    
    def _validate_model(self, model_name: str, model_type: str) -> bool:
        """
        Validate model name against configuration.

        Parameters:
            model_name (str): Name of the model to validate
            model_type (str): Type of model (bi-encoder, cross-encoder, reranker)

        Returns:
            bool: True if model is valid, False otherwise
        """
        if not self.model_config:
            # If config couldn't be loaded, assume model is valid
            return True
            
        try:
            # Map model_type to config model type
            config_model_type = {
                "bi-encoder": "embedding",
                "cross-encoder": "embedding",
                "reranker": "reranker"
            }.get(model_type)
            
            if not config_model_type or config_model_type not in self.model_config["validation_rules"]["models"]:
                logger.warning(f"Model type {model_type} not found in configuration")
                return False
                
            # Check if model exists in huggingface models list
            huggingface_models = self.model_config["validation_rules"]["models"][config_model_type].get("huggingface", [])
            return model_name in huggingface_models
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False
    
    def encode(self, text: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode text using the pre-trained bi-encoder model.
        
        Args:
            text: Text to encode
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            np.ndarray: Encoded text
        """
        if self.model_type != "bi-encoder":
            raise ValueError(f"encode method is only available for bi-encoder models, not {self.model_type}")
        return self.model.encode(text, **kwargs)
    
    def predict(self, texts: Union[List[str], List[Tuple[str, str]]], **kwargs) -> np.ndarray:
        """
        Predict similarity scores using cross-encoder or reranker model.
        
        Args:
            texts: For cross-encoders/rerankers, a list of text pairs (sentence1, sentence2)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            np.ndarray: Similarity scores
        """
        if self.model_type not in ["cross-encoder", "reranker"]:
            raise ValueError(f"predict method is only available for cross-encoder or reranker models, not {self.model_type}")
        return self.model.predict(texts, **kwargs)
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return, if None returns all
            
        Returns:
            List[Dict]: List of dictionaries with document index, score, and text
        """
        if self.model_type != "reranker":
            raise ValueError(f"rerank method is only available for reranker models, not {self.model_type}")
            
        # Create text pairs for the reranker
        text_pairs = [(query, doc) for doc in documents]
        
        # Get similarity scores
        scores = self.model.predict(text_pairs)
        
        # Combine passages with scores
        passage_score_pairs = list(zip(documents, scores))
        
        # Sort by score in descending order
        reranked_pairs = sorted(passage_score_pairs, key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            reranked_pairs = reranked_pairs[:top_k]
            
        return reranked_pairs
    