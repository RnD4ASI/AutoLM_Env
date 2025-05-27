from src.pipeline.shared.logging import get_logger
import json
import pandas as pd
from typing import Any, Dict, List, Optional, Literal, Union, Tuple, Callable, TypeVar
from pathlib import Path
import random
import string
import uuid

logger = get_logger(__name__)

class DataUtility:
    """Static utility class for data operations."""

    @staticmethod
    def generate_uuid() -> str:
        """Generate a UUID string.
        
        Returns:
            str: A new UUID string in standard format
        """
        return str(uuid.uuid4())

    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """Create directory if it does not exist.
        
        Args:
            directory: Directory path to create
            
        Returns:
            Path object for created directory
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def text_operation(operation: str, file_path: Union[str, Path], data: Optional[Any] = None, 
                      file_type: str = 'text', **kwargs) -> Optional[Any]:
        """Perform text file operations (load/save/delete).
        
        Args:
            operation: One of 'load', 'save', 'delete'
            file_path: Path to the text file
            data: Data to save (required for 'save' operation)
            file_type: One of 'text' or 'json'
            **kwargs: Additional arguments for json operations (e.g. indent)
            
        Returns:
            Loaded data for 'load', True for successful 'delete', None for 'save'
            
        Raises:
            ValueError: If operation or file_type is invalid
            FileNotFoundError: If file does not exist for 'load'
        """
        path = Path(file_path)
        
        if operation == 'delete':
            if path.exists():
                path.unlink()
                logger.info(f"Deleted file: {path}")
            logger.info(f"File not found: {path}")
            return None
            
        elif operation == 'load':
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
                
            if file_type == 'text':
                return path.read_text()
            elif file_type == 'json':
                return json.loads(path.read_text())
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        elif operation == 'save':
            if data is None:
                raise ValueError("Data is required for save operation")
                
            path.parent.mkdir(parents=True, exist_ok=True)
            if file_type == 'text':
                path.write_text(str(data))
            elif file_type == 'json':
                path.write_text(json.dumps(data, **kwargs))
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            logger.info(f"Saved file: {path}")
            return None
            
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @staticmethod
    def csv_operation(operation: str, file_path: Union[str, Path], 
                     data: Optional[pd.DataFrame] = None, **kwargs) -> Optional[Union[pd.DataFrame, bool]]:
        """Perform CSV file operations (load/save/delete).
        
        Args:
            operation: One of 'load', 'save', 'delete'
            file_path: Path to the CSV file
            data: DataFrame to save (required for 'save' operation)
            **kwargs: Additional arguments for pandas read_csv/to_csv
            
        Returns:
            DataFrame for 'load', True for successful 'delete', None for 'save'
            
        Raises:
            ValueError: If operation is invalid
            FileNotFoundError: If file does not exist for 'load'
        """
        path = Path(file_path)
        
        if operation == 'delete':
            if path.exists():
                path.unlink()
                logger.info(f"Deleted file: {path}")
            logger.info(f"File not found: {path}")
            return None
            
        elif operation == 'load':
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return pd.read_csv(path, **kwargs)
            
        elif operation == 'save':
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame for save operation")
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, **kwargs)
            logger.info(f"Saved file: {path}")
            return None
            
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @staticmethod
    def parquet_operation(operation: str, file_path: Union[str, Path], 
                          data: Optional[pd.DataFrame] = None, **kwargs) -> Optional[Union[pd.DataFrame, bool]]:
        """Perform Parquet file operations (load/save/delete).
        
        Args:
            operation: One of 'load', 'save', 'delete'
            file_path: Path to the Parquet file
            data: DataFrame to save (required for 'save' operation)
            **kwargs: Additional arguments for pandas read_parquet/to_parquet
            
        Returns:
            DataFrame for 'load', True for successful 'delete', None for 'save'
            
        Raises:
            ValueError: If operation is invalid
            FileNotFoundError: If file does not exist for 'load'
        """
        path = Path(file_path)
        
        if operation == 'delete':
            if path.exists():
                path.unlink()
                logger.info(f"Deleted file: {path}")
            logger.info(f"File not found: {path}")
            return None
            
        elif operation == 'load':
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return pd.read_parquet(path, **kwargs)
            
        elif operation == 'save':
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame for save operation")
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(path, **kwargs)
            logger.info(f"Saved file: {path}")
            return None
            
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @staticmethod
    def format_conversion(data: Any, output_format: str, **kwargs) -> Any:
        """Convert data between different formats (dict/dataframe/string).
        
        Args:
            data: Data to convert (string in markdown format, DataFrame, or dict)
            output_format: One of 'dict', 'dataframe', 'string'
            **kwargs: Format-specific parameters:
                - dict_format: For dataframe to dict conversion (e.g. 'records', 'list')
                - headers: Optional list of column headers for markdown table
                - alignment: Optional list of column alignments ('left', 'center', 'right')
                
        Returns:
            Converted data in the specified output format
            
        Raises:
            ValueError: If format is invalid or markdown table is malformed
        """
        # Auto-detect input type
        input_format = None
        if isinstance(data, str):
            input_format = 'string'
        elif isinstance(data, pd.DataFrame):
            input_format = 'dataframe'
        elif isinstance(data, dict):
            input_format = 'dict'
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
            
        # Convert input to intermediate DataFrame format
        df = None
        if input_format == 'string':
            # Parse markdown table
            lines = [line.strip() for line in data.split('\n') if line.strip()]
            if not lines or '|' not in lines[0]:
                raise ValueError("Input string must be a markdown table with '|' separators")
            
            # Extract headers
            headers = [col.strip() for col in lines[0].strip('|').split('|')]
            headers = [h.strip() for h in headers if h.strip()]
            
            # Skip separator line if present (e.g., |:---:|:---:|)
            start_idx = 1
            if len(lines) > 1 and ':---:' in lines[1] or '---' in lines[1]:
                start_idx = 2
            
            # Parse data rows
            rows = []
            for line in lines[start_idx:]:
                if '|' not in line:
                    continue
                values = [val.strip() for val in line.strip('|').split('|')]
                values = [v.strip() for v in values if v.strip()]
                if len(values) == len(headers):
                    rows.append(values)
            
            df = pd.DataFrame(rows, columns=headers)
            
        elif input_format == 'dataframe':
            df = data
            
        elif input_format == 'dict':
            # Check if all values are scalar (not lists, arrays, or dicts)
            all_scalar = all(not isinstance(val, (list, dict, tuple, set, pd.Series, pd.DataFrame, pd.Index)) 
                            for val in data.values())
            
            if all_scalar and 'orient' not in kwargs:
                # Handle scalar values by default using 'index' orientation
                # This will create a DataFrame with the dict keys as the index
                # and a single column containing the values
                df = pd.DataFrame.from_dict(data, orient='index', columns=[0])
            else:
                # Use the provided kwargs or default handling for non-scalar values
                df = pd.DataFrame.from_dict(data, **kwargs)
            
        # Convert DataFrame to output format
        if output_format == 'dict':
            dict_format = kwargs.get('dict_format', 'records')
            return df.to_dict(dict_format)
            
        elif output_format == 'dataframe':
            return df
            
        elif output_format == 'string':
            # Convert DataFrame to markdown table
            headers = kwargs.get('headers', df.columns.tolist())
            alignments = kwargs.get('alignment', ['center'] * len(headers))
            
            # Validate alignments
            align_map = {
                'left': ':---',
                'center': ':---:',
                'right': '---:'
            }
            separators = [align_map.get(a.lower(), ':---:') for a in alignments]
            
            # Build markdown table
            lines = []
            
            # Add headers
            lines.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
            
            # Add separator line with alignments
            lines.append('| ' + ' | '.join(separators) + ' |')
            
            # Add data rows
            for _, row in df.iterrows():
                values = [str(row[col]) for col in df.columns]
                lines.append('| ' + ' | '.join(values) + ' |')
            
            return '\n'.join(lines)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    @staticmethod
    def dataframe_operations(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply a sequence of operations to a DataFrame.
        
        Args:
            df: Input DataFrame
            operations: List of operation dictionaries, each with:
                - 'type': Operation type
                - 'params': Operation parameters
                
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If operation type is unsupported
        """
        result = df.copy()
        for op in operations:
            op_type = op['type']
            params = op['params']
            
            if op_type == 'filter':
                result = result[result[params['column']].isin(params['values'])]
                
            elif op_type == 'sort':
                result = result.sort_values(**params)
                
            elif op_type == 'group':
                result = result.groupby(params['by']).agg(params['agg']).reset_index()
                
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")
        
        return result

class StatisticsUtility:
    """Static utility class for statistical operations."""
    @staticmethod
    def set_random_seed(size: int = 1, min_value: int = 0, max_value: int = 2**32-1) -> Union[int, List[int]]:
        """Generate and set unique random seeds within a specified range.
        
        Args:
            size: Number of unique random seeds to generate. Default is 1.
            min_value: Minimum value for random seeds (inclusive). Default is 0.
            max_value: Maximum value for random seeds (inclusive). Default is 2**32-1.
            
        Returns:
            If size is 1, returns a single random seed as an integer.
            If size > 1, returns a list of unique random seeds.
            
        Raises:
            ValueError: If size is greater than the range of possible values,
                      or if min_value is greater than max_value.
        """
        if min_value > max_value:
            raise ValueError(f"min_value ({min_value}) must be less than or equal to max_value ({max_value})")
            
        possible_values = max_value - min_value + 1
        if size > possible_values:
            raise ValueError(
                f"Cannot generate {size} unique seeds in range [{min_value}, {max_value}] "
                f"(only {possible_values} possible values)"
            )
            
        if size == 1:
            # Generate a single seed
            seed = random.randrange(min_value, max_value + 1)
            return seed
        else:
            # Generate multiple unique seeds
            seeds = set()
            while len(seeds) < size:
                seeds.add(random.randrange(min_value, max_value + 1))
            return list(seeds)

class AIUtility:
    """
    Static utility class for AI-related operations. 
    Mainly deals with prompts and responses.
    """
    _meta_templates = None
    _task_chains = None
    
    # Valid reasoning templates
    _valid_templates = ["chain_of_thought", "tree_of_thought", "program_synthesis", "deep_thought"]
    # The order in which fix types should be applied (replace should always come first)
    _fix_type_order = ["replace", "prefix", "postfix"]
    
    @classmethod
    def _load_prompts(cls):
        """Load prompt templates and task chains if not already loaded."""
        if cls._meta_templates is None or cls._task_chains is None:
            try:
                config_dir = Path.cwd() / "config"
                cls._meta_templates = DataUtility.text_operation('load', config_dir / "meta_prompt_library.json", file_type='json')
                cls._task_chains = DataUtility.text_operation('load', config_dir / "task_prompt_library.json", file_type='json')
                logger.info("Prompt templates and task chains loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load prompt files: {e}")
                raise

    @classmethod
    def apply_meta_prompt(cls, application: Optional[Literal["metaprompt", "metaresponse", "metaworkflow"]] = "metaprompt", category: str = None, action: str = None, **kwargs):
        cls._load_prompts()
        try:
            template = cls._meta_templates[application][category][action]['template']
        except KeyError:
            logger.error(f"Meta template not found: application={application}, category={category}, action={action}")
            return None

        try:
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
            formatted_prompt = template.format(**kwargs)
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Error formatting template: {e}")
            return None

    @classmethod
    def apply_affix_prompt(
        cls, fix_template: str, 
        fix_type: Optional[str] = None, 
        component: Optional[str] = None, 
        prompt_dict: Optional[Dict[str, Any]] = None,
        components: Optional[List[str]] = None) -> Union[Dict[str, str], Optional[str], Dict[str, Any]]:
        """Get component modifications for a template and fix type, or apply full prompt transformation.
        
        This method handles both:
        1. Original functionality: Retrieve specific modifications from templates
        2. Transformation functionality: Apply full prompt transformation (TemplateAdopter logic)
        
        Args:
            fix_template: The reasoning template to use (e.g., "chain_of_thought")
            fix_type: The type of modification (prefix, postfix, replace)
            component: Optional specific component to retrieve (e.g., 'task', 'instruction', 'response_format')
            prompt_dict: Optional prompt dictionary to transform (enables transformation mode)
            components: Optional list of specific components to modify in transformation mode
            
        Returns:
            - If prompt_dict provided: Returns transformed prompt dictionary
            - If component is specified: Returns the modification string for that component
            - If fix_type is specified: Returns dictionary mapping component names to modifications
            - Otherwise: Returns all fix types for the template
            Returns None or empty dict if not found.
        """
        cls._load_prompts()
        
        # Validate the fix_template
        if fix_template not in cls._valid_templates:
            raise ValueError(f"fix_template must be one of: {cls._valid_templates}")
        
        # TRANSFORMATION MODE: If prompt_dict is provided, apply full transformation
        if prompt_dict is not None:
            # Create a deep copy to avoid modifying the original
            transformed_prompt = copy.deepcopy(prompt_dict)
            
            # If no components specified, use all components in the prompt
            if components is None:
                components = list(prompt_dict.keys())
            
            try:
                # Get available fix types from the template configuration
                try:
                    available_fix_types = cls._meta_templates["ppfix"][fix_template]
                except KeyError:
                    logger.warning(f"No fix types found for template {fix_template}")
                    return transformed_prompt
                
                if not available_fix_types:
                    logger.warning(f"No fix types found for template {fix_template}")
                    return transformed_prompt
                
                # Track which components have been replaced to avoid applying prefix/postfix to them
                replaced_components = set()
                
                # Process fix types in the predefined order (replace, prefix, postfix)
                for fix_type_order in cls._fix_type_order:
                    # Skip if this fix type is not available in the template
                    if fix_type_order not in available_fix_types:
                        logger.debug(f"Fix type '{fix_type_order}' not available for template '{fix_template}'")
                        continue
                    
                    # Get the modifications for this fix type
                    try:
                        fix_mods = cls._meta_templates["ppfix"][fix_template][fix_type_order]
                    except KeyError:
                        logger.debug(f"No modifications found for fix type '{fix_type_order}'")
                        continue
                    
                    # Apply modifications to each specified component
                    for component_name in components:
                        # Skip if component doesn't exist in prompt
                        if component_name not in transformed_prompt:
                            continue
                            
                        # Skip if component has been replaced and we're trying to apply prefix/postfix
                        if component_name in replaced_components and fix_type_order in ["prefix", "postfix"]:
                            continue
                        
                        # Get the modification for this component
                        modification = fix_mods.get(component_name)
                        
                        # Skip if no modification exists or if it's None
                        if modification is None or modification == "":
                            logger.debug(f"No modification found for component '{component_name}' with fix type '{fix_type_order}'")
                            continue
                        
                        # Apply the appropriate modification based on fix_type
                        if fix_type_order == "prefix":
                            transformed_prompt[component_name] = modification + transformed_prompt[component_name]
                            logger.debug(f"Added prefix to component '{component_name}'")
                        elif fix_type_order == "postfix":
                            transformed_prompt[component_name] = transformed_prompt[component_name] + modification
                            logger.debug(f"Added postfix to component '{component_name}'")
                        elif fix_type_order == "replace":
                            transformed_prompt[component_name] = modification
                            replaced_components.add(component_name)
                            logger.debug(f"Replaced component '{component_name}' with template content")
                
                logger.info(f"Successfully transformed prompt with {fix_template} reasoning")
                return transformed_prompt
                
            except Exception as e:
                logger.error(f"Error transforming prompt with {fix_template}: {str(e)}")
                raise RuntimeError(f"Failed to transform prompt: {str(e)}") from e
        
        # ORIGINAL MODE: Return specific modifications from templates
        try:
            # If a specific component is requested, return just that component's modification
            if component and fix_type:
                return cls._meta_templates["ppfix"][fix_template][fix_type][component]
            
            elif fix_type:
                return cls._meta_templates["ppfix"][fix_template][fix_type]
            else:
                return cls._meta_templates["ppfix"][fix_template]
        
        except KeyError:
            if component:
                logger.warning(f"Meta fix not found: template={fix_template}, fix_type={fix_type}, component={component}")
                return None
            else:
                logger.warning(f"Meta fix not found: template={fix_template}, fix_type={fix_type}")
                return {}


    # def get_meta_fix(cls, fix_template: str, fix_type: Optional[str] = None, component: Optional[str] = None) -> Union[Dict[str, str], Optional[str]]:
    #     """Get the component modifications for a specific template and fix type.
        
    #     Args:
    #         fix_template: The reasoning template to use (e.g., "chain_of_thought")
    #         fix_type: The type of modification (prefix, postfix, replace)
    #         component: Optional specific component to retrieve (e.g., 'task', 'instruction', 'response_format')
            
    #     Returns:
    #         If component is specified, returns the modification string for that component.
    #         Otherwise, returns a dictionary mapping component names to their modifications.
    #         Returns None or empty dict if not found.
    #     """
    #     cls._load_prompts()
    #     try:
    #         # If a specific component is requested, return just that component's modification
    #         if component:
    #             return cls._meta_templates["ppfix"][fix_template][fix_type][component]
            
    #         elif fix_type:
    #             return cls._meta_templates["ppfix"][fix_template][fix_type]
    #         else:
    #             return cls._meta_templates["ppfix"][fix_template]
        
    #     except KeyError:
    #         if component:
    #             logger.warning(f"Meta fix not found: template={fix_template}, fix_type={fix_type}, component={component}")
    #             return None
    #         else:
    #             logger.warning(f"Meta fix not found: template={fix_template}, fix_type={fix_type}")
    #             return {}

    """
    =========================================================
    """
    @classmethod
    def parse_list_response(cls, response: str) -> List[str]:
        return None
    
    @classmethod
    def parse_list_prompt(cls, prompt: str) -> str:
        return None
    
    def format_text_list(cls, texts: List[str], text_type: str = "prompt") -> str:
        """Format a list of prompts or responses into a single string, 
        such that multiple prompts or multiple responses can be fed into one context window.
        
        Args:
            texts: List of text strings (prompts or responses)
            text_type: Type of text being formatted ('prompt' or 'response')
            
        Returns:
            A formatted string with numbered entries
            
        Example outputs:
            Prompt 1: <prompt text>
            Prompt 2: <prompt text>
            ...
            
            Response 1: <response text>
            Response 2: <response text>
            ...
            
        Raises:
            ValueError: If text_type is not 'prompt' or 'response'
        """
        if text_type.lower() not in ["prompt", "response"]:
            raise ValueError("text_type must be either 'prompt' or 'response'")
            
        # Capitalize first letter for display
        display_type = text_type.capitalize()
        
        # Format each text with its number
        formatted_texts = [
            f"{display_type} {i+1}: {text.strip()}"
            for i, text in enumerate(texts)
        ]
        
        # Join with newlines
        return "\n\n".join(formatted_texts)

    """
    =========================================================
    """

    @classmethod
    def parse_json_response(cls, response: str) -> Dict[str, Any]:
        return json.loads(response)
    
    def format_json_response(cls, response: str) -> Dict[str, Any]:
        """Format a JSON response string into a dictionary.
        
        Args:
            response: JSON response string
            
        Returns:
            Dictionary representation of the JSON string
            
        Raises:
            JSONDecodeError: If the input string is not valid JSON
        """
        try:
            tmp = json.loads(response.strip('```json').strip('```'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
        return tmp


    # TO DO: need to revisit when it is used. Should be in topologist.py
    # Also needs to consider removal and storage of the response format.
    @classmethod
    def attach_response_format(cls, prompt_json: Dict[str, Any], prompt_id: str) -> Dict[str, Any]:
        """Attach a response_format to a prompt JSON from task_prompt_library.json.
        
        Args:
            prompt_json: Prompt in JSON format without response_format
            prompt_id: ID of the prompt in task_prompt_library.json to get response_format from
            
        Returns:
            Merged prompt JSON with response_format added from reference prompt
            
        Example:
            Input prompt_json:
                {
                    "role": "Engineer",
                    "task": "Review code"
                }
            
            Reference prompt (prompt_id="prompt_001"):
                {
                    "promptid": 1,
                    "components": {
                        "response_format": "List format with severity"
                    }
                }
                
            Output:
                {
                    "role": "Engineer",
                    "task": "Review code",
                    "response_format": "List format with severity"
                }
        """
        try:
            # Load reference prompt
            cls._load_prompts()
            if not cls._task_chains or prompt_id not in cls._task_chains:
                raise ValueError(f"Prompt ID {prompt_id} not found in task chains")
            
            reference_prompt = cls._task_chains[prompt_id]
            
            # Get response_format from reference prompt
            if ("components" in reference_prompt and 
                "response_format" in reference_prompt["components"]):
                response_format = reference_prompt["components"]["response_format"]
            else:
                raise ValueError(f"No response_format found for prompt ID {prompt_id}")
            
            # Create deep copy of input prompt
            task_prompt = json.loads(json.dumps(prompt_json))
            
            # Add response_format directly to the dictionary
            task_prompt["response_format"] = response_format
            
            return task_prompt
            
        except Exception as e:
            logger.error(f"Error merging prompt response format: {e}")
            raise
    

class MemoryUtility:
    @classmethod
    def store_vector_db(cls, db, data)-> None:
        """Store data in a vector database."""
        return None

    @classmethod
    def store_graph_db(cls, db, data)-> None:
        """Store data in a graph database."""
        return None

    @classmethod
    def classify_data(cls, db, data)-> None:
        """Classify data."""
        return None
    

