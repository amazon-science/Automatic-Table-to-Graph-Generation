"""
Simplified AutoG2 Service for Web App
Handles the core AutoG2 functionality with better error handling
"""

import os
import ast
import yaml
import tempfile
import shutil
import numpy as np
import pandas as pd
import sys
import io
import re
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path

# Import backend modules with fallbacks
BACKEND_AVAILABLE = True
try:
    from models.autog.agent_old import AutoG_Agent
    from models.llm.bedrock import get_bedrock_llm
    from prompts.task import get_task_description
    from prompts.identify import identify_prompt
    from utils.misc import seed_everything
    from models.llm.gconstruct import analyze_dataframes
    from dbinfer_bench.rdb_dataset import DBBRDBDataset
    from dbinfer_bench.dataset_meta import DBBRDBDatasetMeta
except ImportError as e:
    print(f"Warning: Failed to import backend modules: {e}")
    BACKEND_AVAILABLE = False


class OutputCapture:
    """Capture stdout/stderr and forward to callback"""
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self.callback = callback
        self.captured_output = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.current_round = None
        
    def write(self, text: str):
        """Write method for stdout/stderr replacement"""
        # Forward to original stdout for debugging
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Clean and process the text
        text = text.strip()
        if not text:
            return
            
        # Store the output
        self.captured_output.append(text)
        
        # Parse round information
        round_match = re.search(r'Round:\s*(\d+)', text)
        if round_match:
            self.current_round = int(round_match.group(1))
            
        # Send to callback if available
        if self.callback:
            try:
                self.callback(text)
            except Exception as e:
                # Don't let callback errors break the capture
                self.original_stdout.write(f"Callback error: {e}\n")
    
    def flush(self):
        """Flush method for stdout/stderr replacement"""
        self.original_stdout.flush()
    
    @contextmanager
    def capture(self):
        """Context manager to capture output"""
        try:
            # Replace stdout and stderr
            sys.stdout = self
            sys.stderr = self
            yield self
        finally:
            # Restore original stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr


class SimpleAutoGService:
    """Simplified AutoG2 service for web app"""
    
    def __init__(self, cache_strategy="hybrid"):
        self.temp_dir = None
        self.cache_strategy = cache_strategy
        self.memory_cache = {}
        self.cache_size_limit = 10 * 1024 * 1024  # 10MB
        
        # Data type mapping
        self.dtype_mapping = {
            'object': 'category',
            'string': 'category', 
            'category': 'category',
            'int8': 'float',
            'int16': 'float',
            'int32': 'float', 
            'int64': 'float',
            'float16': 'float',
            'float32': 'float',
            'float64': 'float',
            'bool': 'float',
            'datetime64[ns]': 'datetime',
            'timedelta64[ns]': 'datetime',
            'period[D]': 'datetime'
        }
    
    def get_llm_config(self, llm_name: str) -> Dict[str, Any]:
        """Get LLM configuration"""
        CONTEXT_SIZE = 65536
        OUTPUT_SIZE = 65536
        
        configs = {
            "sonnet4": {
                "model_name": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "context_size": CONTEXT_SIZE,
                "output_size": OUTPUT_SIZE
            },
            "sonnet3": {
                "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "context_size": CONTEXT_SIZE,
                "output_size": OUTPUT_SIZE
            },
            "haiku": {
                "model_name": "anthropic.claude-3-haiku-20240307-v1:0",
                "context_size": CONTEXT_SIZE,
                "output_size": OUTPUT_SIZE
            }
        }
        return configs.get(llm_name, configs["sonnet3"])
    
    def capitalize_first_alpha(self, text: str) -> str:
        """Capitalize first alphabetic character"""
        for i, char in enumerate(text):
            if char.isalpha():
                return text[:i] + text[i:].replace(char, char.upper(), 1)
        return text
    
    def create_temp_workspace(self, dataframes: Dict[str, pd.DataFrame], dataset_name: str) -> str:
        """Create temporary workspace"""
        self.temp_dir = tempfile.mkdtemp(prefix="autog2_webapp_")
        data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save DataFrames as parquet files
        for table_name, df in dataframes.items():
            clean_name = self.capitalize_first_alpha(table_name)
            if not clean_name.endswith('.parquet'):
                clean_name += '.parquet'
            
            file_path = os.path.join(data_dir, clean_name)
            df.to_parquet(file_path, index=False)
        
        return self.temp_dir
    
    def generate_metadata(self, dataframes: Dict[str, pd.DataFrame], dataset_name: str) -> Dict[str, Any]:
        """Generate metadata from DataFrames"""
        meta_dict = {
            'dataset_name': dataset_name,
            'tables': []
        }
        
        for table_name, df in dataframes.items():
            clean_name = self.capitalize_first_alpha(table_name)
            
            table_meta = {
                'name': clean_name,
                'columns': [],
                'format': 'parquet',
                'source': f'data/{clean_name}.parquet'
            }
            
            for col_name, col_dtype in df.dtypes.to_dict().items():
                dtype_str = str(col_dtype)
                if col_dtype == 'object':
                    try:
                        pd.to_numeric(df[col_name])
                        dtype_str = 'int32'
                    except:
                        dtype_str = 'object'
                
                table_meta['columns'].append({
                    'name': col_name,
                    'dtype': self.dtype_mapping.get(dtype_str, 'category'),
                })
            
            meta_dict['tables'].append(table_meta)
        
        return meta_dict
    
    def run_autog2(
        self,
        dataframes: Dict[str, pd.DataFrame],
        llm_name: str = "sonnet3",
        method: str = "autog-s",
        task_name: str = "custom:kg",
        dataset_name: str = "webapp_dataset",
        seed: int = 0,
        max_rounds: int = 5,
        progress_callback=None,
        custom_task_description: str = None
    ) -> Tuple[str, str, str, Dict[str, str]]:
        """
        Run AutoG2 processing
        
        Args:
            dataframes: Dictionary of table name to DataFrame
            llm_name: Name of the LLM model to use
            method: AutoG processing method (autog-s, autog-m, baseline)
            task_name: Task identifier (e.g., "custom:kg")
            dataset_name: Name for the dataset
            seed: Random seed for reproducibility
            max_rounds: Maximum number of processing rounds
            progress_callback: Optional callback for progress updates
            custom_task_description: Optional custom task description (overrides default task description)
        
        Returns:
            Tuple of (agent_history, analysis_result, output_path, generated_files)
        """
        try:
            # Set random seed
            seed_everything(seed)
            
            # Get LLM configuration
            llm_config = self.get_llm_config(llm_name)
            
            # Create temporary workspace
            temp_path = self.create_temp_workspace(dataframes, dataset_name)
            
            # Generate metadata
            metadata_dict = self.generate_metadata(dataframes, dataset_name)
            
            # Save metadata
            metadata_path = os.path.join(temp_path, 'metadata.yaml')
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata_dict, f, default_flow_style=False)
            
            # Create DBBRDBDataset
            data = DBBRDBDataset(temp_path)
            
            # Parse task and get description
            dataset, task = task_name.split(':')[0], task_name.split(':')[1]
            
            # Use custom task description if provided, otherwise get from task definitions
            if custom_task_description:
                task_description = custom_task_description
            else:
                task_description = get_task_description(dataset, task)
            
            # Analyze DataFrames
            table_meta_dict = {
                f'Table {table_name}': table for table_name, table in data.tables.items()
            }
            analysis_result = analyze_dataframes(table_meta_dict)
            
            # Save analysis
            info_path = os.path.join(temp_path, 'information.txt')
            with open(info_path, 'w') as f:
                f.write(analysis_result)
            
            # Cache analysis result
            self._cache_file_content(info_path)
            
            # Get LLM response for identification
            identify_inputs = identify_prompt(analysis_result)
            bedrock_llm = get_bedrock_llm(
                llm_config["model_name"], 
                context_size=llm_config["context_size"]
            )
            response = bedrock_llm.complete(identify_inputs, max_tokens=llm_config["output_size"]).text
            
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No valid JSON found in LLM response")
            
            val_response = response[start:end]
            metainfo = ast.literal_eval(val_response)
            metainfo = {
                self.capitalize_first_alpha(key): value 
                for key, value in metainfo.items()
            }
            
            # Generate schema input
            schema_input = self._generate_training_metainfo(data, metainfo, task)
            
            # Setup AutoG paths
            autog_path = os.path.join(temp_path, "autog")
            os.makedirs(autog_path, exist_ok=True)
            
            # Get DeepJoin path (optional)
            deepjoin_path = self._get_deepjoin_path()
            
            # Validate method parameter before creating agent
            if method is None or method == "None" or str(method).strip() == "":
                print(f"WARNING: Invalid method '{method}' in service, using 'autog-s'")
                method = "autog-s"
            
            valid_methods = ["autog-s", "autog-m", "baseline"]
            if method not in valid_methods:
                print(f"WARNING: Unknown method '{method}' in service, using 'autog-s'")
                method = "autog-s"
            
            # print(f"DEBUG: Creating AutoG_Agent with mode='{method}'")
            
            # Initialize AutoG Agent
            agent = AutoG_Agent(
                initial_schema=schema_input,
                mode=method,
                oracle=None,
                llm_model_name=llm_config["model_name"],
                context_size=llm_config["context_size"],
                path_to_file=autog_path,
                llm_sleep=1,
                use_cache=False,
                threshold=max_rounds,  # Use max_rounds as threshold
                output_size=llm_config["output_size"],
                task_description=task_description,
                dataset=dataset,
                task_name=task,
                schema_info=analysis_result,
                lm_path=deepjoin_path,
                recalculate=False
            )
            
            # Run agent with real-time output capture
            if progress_callback:
                progress_callback("Starting AutoG2 agent...")
            
            # Create output capture with callback
            def output_handler(text: str):
                if progress_callback:
                    progress_callback(text)
            
            # Execute the agent with output capture
            with OutputCapture(callback=output_handler).capture() as capture:
                agent.augment()
            
            # Get the captured output and agent history
            captured_output = capture.captured_output
            agent_history = "\\n".join(agent.history)
            
            # Process captured output for final summary
            if progress_callback and captured_output:
                # Count actual rounds from captured output
                round_count = 0
                for output_line in captured_output:
                    round_match = re.search(r'Round:\s*(\d+)', output_line)
                    if round_match:
                        round_num = int(round_match.group(1))
                        round_count = max(round_count, round_num + 1)  # +1 because rounds are 0-indexed
                
                if round_count > 0:
                    progress_callback(f"AutoG2 completed {round_count} rounds total")
            
            # Save agent history
            final_path = os.path.join(autog_path, "final")
            os.makedirs(final_path, exist_ok=True)
            
            history_file = os.path.join(final_path, "agent_history.txt")
            with open(history_file, "w") as f:
                f.write(agent_history)
            
            # Cache history
            self._cache_file_content(history_file)
            
            # Collect generated files
            generated_files = self._collect_generated_files(autog_path)
            
            return agent_history, analysis_result, autog_path, generated_files
            
        except Exception as e:
            raise Exception(f"AutoG2 execution failed: {str(e)}")
    
    def _generate_training_metainfo(self, data: DBBRDBDataset, meta_dict: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Generate training metadata"""
        overall_meta = {
            'dataset_name': data.dataset_name,
            'tables': []
        }
        
        # Convert data metadata to dict
        data_meta_dict = {key.name: key for key in data.metadata.tables}
        
        for table in data.tables:
            table_val = data.tables[table]
            table_meta = {
                'name': table,
                'columns': [],
                'format': data_meta_dict[table].format.value,
                'source': data_meta_dict[table].source
            }
            
            for column_name, column_value in table_val.items():
                if column_name not in meta_dict.get(table, {}):
                    continue
                
                # Check if numerical and primary key
                is_numerical = column_value.dtype in ['int64', 'float64']
                if not is_numerical and column_value.dtype == 'object':
                    try:
                        column_value.astype(int)
                        is_numerical = True
                    except:
                        is_numerical = False
                
                is_primary_key = is_numerical and np.unique(column_value).size == column_value.size
                
                if is_primary_key:
                    table_meta['columns'].append({
                        'name': column_name,
                        'dtype': 'primary_key',
                        'description': meta_dict[table][column_name][1]
                    })
                else:
                    table_meta['columns'].append({
                        'name': column_name,
                        'dtype': meta_dict[table][column_name][0],
                        'description': meta_dict[table][column_name][1]
                    })
            
            overall_meta['tables'].append(table_meta)
        
        overall_meta['tasks'] = []
        return overall_meta
    
    def _get_deepjoin_path(self) -> Optional[str]:
        """Get DeepJoin path with fallbacks"""
        possible_paths = [
            "deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27",
            "../deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27",
            "deepjoin",
            "../deepjoin"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        print("⚠️ DeepJoin model not found. Join discovery will be disabled.")
        return None
    
    def _collect_generated_files(self, autog_path: str) -> Dict[str, str]:
        """Collect generated files"""
        files = {}
        final_path = os.path.join(autog_path, "final")
        
        # Common files to look for
        file_patterns = {
            'agent_history': os.path.join(final_path, "agent_history.txt"),
            'metadata': os.path.join(self.temp_dir, "metadata.yaml"),
            'final_metadata': os.path.join(final_path, "metadata.yaml"),
            'schema_png': os.path.join(final_path, "schema.png"),
            'schema_pdf': os.path.join(final_path, "schema.pdf"),
            'information': os.path.join(self.temp_dir, "information.txt"),
        }
        
        for file_type, file_path in file_patterns.items():
            if os.path.exists(file_path):
                files[file_type] = file_path
        
        return files
    
    def get_file_content(self, file_path: str) -> bytes:
        """Get file content for download"""
        # Try cache first
        cached = self._get_cached_content(file_path)
        if cached is not None:
            return cached
        
        # Read from disk
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            self._cache_file_content(file_path, content)
            return content
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def _should_cache_in_memory(self, file_path: str, content_size: int = None) -> bool:
        """Determine if file should be cached in memory"""
        if self.cache_strategy == "disk":
            return False
        elif self.cache_strategy == "memory":
            return True
        
        # Hybrid strategy
        file_name = os.path.basename(file_path).lower()
        
        # Always cache small text files
        text_extensions = ['.txt', '.yaml', '.yml', '.json', '.csv']
        if any(file_name.endswith(ext) for ext in text_extensions):
            if content_size is None:
                try:
                    content_size = os.path.getsize(file_path)
                except:
                    content_size = 0
            return content_size < self.cache_size_limit
        
        return False
    
    def _cache_file_content(self, file_path: str, content: bytes = None) -> None:
        """Cache file content in memory"""
        if content is None:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
            except:
                return
        
        if self._should_cache_in_memory(file_path, len(content)):
            self.memory_cache[file_path] = content
    
    def _get_cached_content(self, file_path: str) -> Optional[bytes]:
        """Get cached content"""
        return self.memory_cache.get(file_path)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        total_size = sum(len(content) for content in self.memory_cache.values())
        
        return {
            'strategy': self.cache_strategy,
            'memory_files': len(self.memory_cache),
            'memory_size': total_size,
            'memory_size_mb': total_size / (1024 * 1024),
            'cached_files': [os.path.basename(path) for path in self.memory_cache.keys()]
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()