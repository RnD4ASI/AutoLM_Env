import json
import uuid
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Assuming retrievers, generator, evaluator are in pipeline.processing
from src.pipeline.processing.retriever import VectorRetriever, GraphRetriever, MemoryRetriever
from src.pipeline.processing.generator import Generator, MetaGenerator
from src.pipeline.processing.evaluator import Evaluator
from src.pipeline.shared.logging import get_logger # Assuming shared logging

logger = get_logger(__name__)

class NewExecutionService:
    def __init__(self,
                 main_config_path: Union[str, Path],
                 mle_config_path: Union[str, Path],
                 model_config_path: Union[str, Path],
                 vector_db_json_path: Union[str, Path],
                 episodic_memory_json_path: Union[str, Path],
                 personality_memory_json_path: Union[str, Path],
                 graph_db_pkl_path: Union[str, Path]
                 # Plan and task library paths might be passed to execute_plan directly
                 ):
        logger.info("Initializing NewExecutionService")
        self.main_config = self._load_json_config(main_config_path)
        self.mle_config = self._load_json_config(mle_config_path)
        self.model_config = self._load_json_config(model_config_path)

        # Initialize utilities from pipeline.processing
        self.generator = Generator() # May need config
        self.meta_generator = MetaGenerator(generator=self.generator) # May need config
        self.evaluator = Evaluator(generator=self.generator) # May need config

        # Initialize retrievers with paths to JSON/PKL files
        self.vector_retriever = VectorRetriever(vector_db_path=vector_db_json_path)
        self.graph_retriever = GraphRetriever(graph_db=graph_db_pkl_path) # graph_db parameter name
        self.episodic_memory_retriever = MemoryRetriever(memory_db_path=episodic_memory_json_path, memory_type="episodic") # Added memory_type
        self.personality_memory_retriever = MemoryRetriever(memory_db_path=personality_memory_json_path, memory_type="personality") # Added memory_type
        
        # For working memory during a plan execution
        # This will use the MemoryRetriever but point to task-specific files.
        # No separate class instance needed here, will be configured in execute_plan.

        self.current_execution_id = None
        self.current_plan = None
        self.current_task_library = None
        self.working_memory_base_dir = None # Initialize attribute

    def _load_json_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}

    def _load_plan(self, plan_file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        logger.info(f"Loading plan from {plan_file_path}")
        # Schema: db/schema/schema_plan.json (expects a 'graphs' collection)
        # PlanService output (prompt_flow_config.json) is a single graph structure.
        plan_data = self._load_json_config(plan_file_path)
        if not plan_data: # Check if loading failed (returned {} or None)
            logger.error(f"Plan file {plan_file_path} could not be loaded or is empty.")
            return None
        if "nodes" not in plan_data or "edges" not in plan_data:
            logger.warning(f"Loaded plan from {plan_file_path} is missing 'nodes' or 'edges'. Plan content: {plan_data}")
            # Depending on strictness, could return None or the partial data.
            # For robustness, ensure it has a known "bad" state or minimal required structure.
        return plan_data

    def _load_task_library(self, task_library_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        logger.info(f"Loading task library from {task_library_path}")
        # PlanService output (shortlisted_prompts.json) is a direct dict of prompts.
        library_data = self._load_json_config(task_library_path)
        if not library_data: # Check if loading failed
            logger.error(f"Task library from {task_library_path} could not be loaded or is empty.")
            return None
        if not isinstance(library_data, dict):
            logger.error(f"Loaded task library from {task_library_path} is not a dictionary. Type: {type(library_data)}")
            return None
        return library_data

    def execute_plan(self,
                     plan_file_path: Union[str, Path],
                     task_library_path: Union[str, Path],
                     working_memory_base_dir: Union[str, Path] = "/tmp/autolm_working_memory"
                     ) -> Dict[str, Any]:
        logger.info(f"Starting execution of plan: {plan_file_path}")
        self.current_execution_id = str(uuid.uuid4())
        self.working_memory_base_dir = Path(working_memory_base_dir)
        self.working_memory_base_dir.mkdir(parents=True, exist_ok=True)

        self.current_plan = self._load_plan(plan_file_path)
        self.current_task_library = self._load_task_library(task_library_path)

        if not self.current_plan or not self.current_task_library:
            logger.error("Plan or task library could not be loaded. Aborting execution.")
            return {"status": "error", "message": "Failed to load plan or task library."}

        execution_results = {}
        # Simple sequential execution based on node IDs, assuming nodes are named node_1, node_2 etc.
        # A proper scheduler would use the 'edges' in self.current_plan.
        # For now, let's assume nodes in the plan are somewhat ordered or identified for sequence.
        
        # TODO: Implement proper task sequencing based on plan 'edges'
        for node_id, task_details in sorted(self.current_plan.get("nodes", {}).items()):
            prompt_id = task_details.get("prompt_id")
            task_prompt_template = self.current_task_library.get(prompt_id)

            if not task_prompt_template:
                logger.warning(f"Task prompt template for {prompt_id} not found in library. Skipping task {node_id}.")
                execution_results[node_id] = {"status": "skipped", "message": "Prompt template not found."}
                continue
            
            logger.info(f"Executing task {node_id} (Prompt: {prompt_id})")
            task_result = self._execute_task_prompt(node_id, task_details, task_prompt_template)
            execution_results[node_id] = task_result
            
            # Basic break on error for now
            if task_result.get("status") == "error":
                logger.error(f"Error in task {node_id}. Halting plan execution.")
                break
        
        logger.info(f"Plan execution finished for {plan_file_path}")
        return {"status": "completed", "execution_id": self.current_execution_id, "results": execution_results}

    def _execute_task_prompt(self,
                             task_id: str,
                             task_details: Dict[str, Any],
                             task_prompt_template: Dict[str, Any]
                             ) -> Dict[str, Any]:
        try:
            # 1. Retrieve contexts (VectorDB, GraphDB)
            retrieved_contexts = self._retrieve_contexts_from_dbs(task_prompt_template)
            
            # 2. Store these contexts to working memory for this task
            self._store_contexts_to_working_memory(retrieved_contexts, task_id, "initial_retrieval")

            # 3. Retrieve from working memory (e.g. episodic, or prior step's output for this task)
            #    This might involve multiple calls or a more complex query to MemoryRetriever
            working_contexts = self._retrieve_contexts_from_memory(
                self._get_working_memory_path(task_id, "initial_retrieval"),
                query_or_criteria="*" # retrieve all for now
            )
            
            # 4. Get iteration parameters (if any)
            iteration_parameters = self._get_iteration_parameters(task_details, task_prompt_template)
            
            all_iteration_results = []
            for i, iter_param in enumerate(iteration_parameters):
                logger.info(f"Running iteration {i+1}/{len(iteration_parameters)} for task {task_id}")
                
                # 5. Contextualize prompt
                current_prompt_text = self._contextualize_prompt(task_prompt_template, iter_param, working_contexts)
                
                # 6. Classify task
                task_type, task_complexity = self._classify_task(current_prompt_text)
                
                # 7. Get personality traits
                personality_traits = self._get_personality_traits(task_prompt_template)
                
                # 8. Apply personality to system prompt
                system_prompt = task_prompt_template.get("components", {}).get("system_prompt", "") # Assuming system_prompt is a component
                enriched_system_prompt = self._apply_personality_to_system_prompt(system_prompt, personality_traits)
                
                # 9. Select execution config (model, topology, params)
                llm_config = self._select_execution_config(task_type, task_complexity)
                
                # 10. Execute LLM call
                llm_response = self._execute_llm_call(current_prompt_text, enriched_system_prompt, llm_config)
                
                # 11. Parse output
                parsed_output = self._parse_output(llm_response, task_prompt_template.get("response_format", "text"))
                
                # 12. Evaluate output
                evaluation_result = self._evaluate_output(parsed_output, task_prompt_template.get("evaluation_criteria", []))
                
                # 13. Store result to working memory for this task & iteration
                iter_result_data = {
                    "iteration": i, "param": iter_param, "prompt": current_prompt_text, 
                    "response": parsed_output, "evaluation": evaluation_result, "llm_config": llm_config
                }
                self._store_result_to_working_memory(iter_result_data, task_id, f"iteration_{i}_result")
                all_iteration_results.append(iter_result_data)

            return {"status": "success", "iterations_run": len(iteration_parameters), "results": all_iteration_results}

        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            logger.debug(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _get_working_memory_path(self, task_id: str, context_name: str) -> Path:
        task_memory_dir = self.working_memory_base_dir / self.current_execution_id / task_id
        task_memory_dir.mkdir(parents=True, exist_ok=True)
        return task_memory_dir / f"{context_name}.json"

    def _retrieve_contexts_from_dbs(self, task_prompt_template: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.debug(f"Retrieving contexts from DBs for prompt: {task_prompt_template.get('prompt_id')}")
        # Placeholder implementation
        all_contexts = []
        query = task_prompt_template.get("components",{}).get("task", "") # Use task as query for now
        
        # VectorDB
        vector_config = {"search": {"semantic": True, "top_k": 3}} 
        vector_contexts = []
        if self.vector_retriever:
            try:
                query_for_vdb = task_prompt_template.get("components", {}).get("task", task_prompt_template.get("description", ""))
                logger.info(f"Querying VectorDB with: {query_for_vdb[:100]}...")
                retrieved_vector_items = self.vector_retriever.retrieve(query_for_vdb, config=vector_config)
                for item in retrieved_vector_items:
                    item['source_db'] = 'vector_db' 
                    vector_contexts.append(item)
            except Exception as e:
                logger.error(f"Error retrieving from VectorDB: {e}")
        if vector_contexts: 
            all_contexts.extend(vector_contexts)
        
        # GraphDB
        graph_config = {"search": {"path_based": True, "top_k": 3}} 
        graph_contexts_processed = []
        if self.graph_retriever:
            try:
                query_for_gdb = task_prompt_template.get("components", {}).get("task", task_prompt_template.get("description", ""))
                logger.info(f"Querying GraphDB with: {query_for_gdb[:100]}...")
                retrieved_graph_items = self.graph_retriever.retrieve(query_for_gdb, config=graph_config)
                for item in retrieved_graph_items:
                    item['source_db'] = 'graph_db' 
                    graph_contexts_processed.append(item)
            except Exception as e:
                logger.error(f"Error retrieving from GraphDB: {e}")
        if graph_contexts_processed:
            all_contexts.extend(graph_contexts_processed)
        
        logger.info(f"Retrieved {len(all_contexts)} total contexts from VDB and GDB.")
        return all_contexts


    def _store_contexts_to_working_memory(self, contexts: List[Dict[str, Any]], task_id: str, context_name: str) -> None:
        path = self._get_working_memory_path(task_id, context_name)
        logger.debug(f"Storing {len(contexts)} contexts to working memory: {path}")
        try:
            with open(path, 'w') as f:
                json.dump(contexts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to store contexts to {path}: {e}")

    def _retrieve_contexts_from_memory(self, retriever_instance: MemoryRetriever, query: str, config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Retrieve data using the provided MemoryRetriever instance."""
        if not retriever_instance:
            logger.warning("Memory retriever instance not provided for _retrieve_contexts_from_memory.")
            return []
        
        final_config = config if config is not None else {"limits": {"top_k": 5}}

        logger.debug(f"Retrieving from {retriever_instance.memory_type} memory with query: {query[:100]}")
        
        try:
            return retriever_instance.retrieve(query=query, config=final_config)
        except Exception as e:
            logger.error(f"Failed to retrieve from {retriever_instance.memory_type} memory: {e}")
            logger.debug(traceback.format_exc())
            return []


    def _get_iteration_parameters(self, task_details: Dict[str, Any], task_prompt_template: Dict[str, Any]) -> List[Any]:
        logger.debug(f"Getting iteration parameters for task: {task_details.get('prompt_id')}")
        if "iterations" in task_details: # Iterations defined in the plan for this specific task node
            return task_details["iterations"]
        # Placeholder: no iterations by default if not in plan
        return [None] # Must return a list, [None] means one iteration with no specific param

    def _contextualize_prompt(self, task_prompt_template: Dict[str, Any], iteration_value: Any, retrieved_contexts: List[Dict[str, Any]]) -> str:
        logger.debug(f"Contextualizing prompt: {task_prompt_template.get('prompt_id')}")
        components = task_prompt_template.get("components", {})
        base_prompt_text = components.get("instructions", components.get("task", "No task instructions provided."))

        # Construct prompt header
        final_prompt_text = f"Role: {components.get('role', 'AI Assistant')}\n"
        final_prompt_text += f"Primary Task: {components.get('task', '')}\n"
        final_prompt_text += f"Purpose: {components.get('purpose', '')}\n"
        final_prompt_text += f"Audience: {components.get('audience', '')}\n"
        if components.get('context'):
             final_prompt_text += f"Scenario Context: {components.get('context')}\n"
        
        # Handle iteration value
        if iteration_value is not None:
            if "{{iteration_value}}" in base_prompt_text: # Check if placeholder exists
                base_prompt_text = base_prompt_text.replace("{{iteration_value}}", str(iteration_value))
            else: # If no placeholder, append the iteration value information
                final_prompt_text += f"Current Iteration Value: {str(iteration_value)}\n"

        final_prompt_text += f"\nInstructions:\n{base_prompt_text}\n"
        
        # Add retrieved contexts
        if retrieved_contexts:
            context_str = "\n--- Relevant Context Start ---\n"
            for i, ctx in enumerate(retrieved_contexts[:3]): # Limit to top 3 contexts
                ctx_content = ctx.get('content', 'No content')
                ctx_source = ctx.get('source_db', ctx.get('source', 'unknown_source')) 
                ctx_score = ctx.get('score', 0.0)
                context_str += f"Context Item {i+1} (Source: {ctx_source}, Relevance: {ctx_score:.2f}):\n{ctx_content}\n\n"
            context_str += "--- Relevant Context End ---\n"
            final_prompt_text += context_str
        
        final_prompt_text += f"\nExpected Response Format: {components.get('response_format', 'Clear, concise text.')}\n"
        return final_prompt_text

    def _classify_task(self, contextualized_prompt: str) -> tuple[str, str]:
        logger.debug("Classifying task (stub).")
        # Placeholder: use MetaGenerator - actual call structure TBD
        # task_type = self.meta_generator.get_meta_generation(...)
        return "general", "medium" # task_type, task_complexity

    def _get_personality_traits(self, task_prompt_template: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Getting personality traits (stub).")
        # Placeholder: use self.personality_memory_retriever
        # query = task_prompt_template.get("prompt_id","")
        # traits = self.personality_memory_retriever.retrieve(query, config={})
        return {"mbti_type": "INTJ", "mode_description": "Analytical and strategic."}

    def _apply_personality_to_system_prompt(self, system_prompt: str, personality_traits: Dict[str, Any]) -> str:
        logger.debug("Applying personality to system prompt (stub).")
        if not personality_traits: return system_prompt
        return system_prompt + f"\n[Personality: {personality_traits.get('mbti_type','N/A')} - {personality_traits.get('mode_description','')}]"

    def _select_execution_config(self, task_type: str, task_complexity: str) -> Dict[str, Any]:
        logger.debug(f"Selecting execution config for task type: {task_type}, complexity: {task_complexity} (stub).")
        # TODO: Implement logic using self.mle_config and self.model_config
        return {
            "model_name": self.model_config.get("default_model", "default_llm"),
            "temperature": self.model_config.get("default_temperature", 0.7),
            "max_tokens": self.model_config.get("default_max_tokens", 1000),
            "topology": None 
        }

    def _execute_llm_call(self, prompt_text: str, system_prompt: str, llm_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Executing LLM call with model: {llm_config.get('model_name')} (stub)")
        # response = self.generator.get_completion(...)
        return {"content": f"Placeholder response for prompt: {prompt_text[:50]}...", "usage": {"total_tokens": 50}}

    def _parse_output(self, llm_response: Dict[str, Any], output_format_instructions: Any) -> Any:
        logger.debug("Parsing LLM output (stub).")
        # TODO: Implement parsing based on output_format_instructions
        return llm_response.get("content") # Simple passthrough for now

    def _evaluate_output(self, parsed_output: Any, evaluation_criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.debug("Evaluating output (stub).")
        # Placeholder: use self.evaluator
        # evaluation = self.evaluator.evaluate(...)
        if not evaluation_criteria: return {"score": 1.0, "feedback": "No criteria, approved."}
        return {"score": 0.8, "feedback": "Placeholder evaluation."}

    def _store_result_to_working_memory(self, result_data: Dict[str, Any], task_id: str, context_name: str) -> None:
        path = self._get_working_memory_path(task_id, context_name)
        logger.debug(f"Storing result to working memory: {path}")
        try:
            with open(path, 'w') as f:
                json.dump(result_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to store result to {path}: {e}")

if __name__ == '__main__':
    # Basic test setup - paths would need to be valid or mocked
    # This is for standalone testing of the skeleton, not part of the AutoLM run
    # Create dummy config and data files if needed for testing this script directly
    
    # Dummy file creation for testing
    Path("/tmp/dummy_main_config.json").write_text(json.dumps({"app_name": "AutoLM_Test"}))
    Path("/tmp/dummy_mle_config.json").write_text(json.dumps({"default_mle_setting": True}))
    Path("/tmp/dummy_model_config.json").write_text(json.dumps({"default_model": "test_model", "default_temperature": 0.5, "default_max_tokens": 100}))
    Path("/tmp/dummy_vector_db.json").write_text(json.dumps([{"id": "v1", "text": "vec data"}]))
    Path("/tmp/dummy_episodic_memory.json").write_text(json.dumps([{"id": "e1", "event": "ep data"}]))
    Path("/tmp/dummy_personality_memory.json").write_text(json.dumps([{"id": "p1", "trait": "pers data"}]))
    Path("/tmp/dummy_graph_db.pkl").write_text("dummy_graph_content") # Not a real PKL
    Path("/tmp/dummy_plan.json").write_text(json.dumps({
        "nodes": {"node_1": {"prompt_id": "P001", "task_summary": "First task"}},
        "edges": []
    }))
    Path("/tmp/dummy_task_library.json").write_text(json.dumps({
        "P001": {"components": {"task": "Do the first thing."}}
    }))

    try:
        service = NewExecutionService(
            main_config_path="/tmp/dummy_main_config.json",
            mle_config_path="/tmp/dummy_mle_config.json",
            model_config_path="/tmp/dummy_model_config.json",
            vector_db_json_path="/tmp/dummy_vector_db.json",
            episodic_memory_json_path="/tmp/dummy_episodic_memory.json",
            personality_memory_json_path="/tmp/dummy_personality_memory.json",
            graph_db_pkl_path="/tmp/dummy_graph_db.pkl"
        )
        logger.info("NewExecutionService initialized for basic test.")
        
        results = service.execute_plan(
            plan_file_path="/tmp/dummy_plan.json",
            task_library_path="/tmp/dummy_task_library.json",
            working_memory_base_dir="/tmp/test_exec_working_mem"
        )
        logger.info(f"Test execution results: {results}")

    except Exception as e:
        logger.error(f"Error during basic test: {e}")
        logger.debug(traceback.format_exc())
