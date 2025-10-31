"""
AutoG2 Service for Streamlit Web App
Handles in-memory DataFrame processing instead of file-based operations
"""

import os
import ast
import yaml
import tempfile
import shutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import sys
import subprocess
from main.autog2 import main as autog2_main, get_llm_config as autog2_get_llm_config
from models.llm.gconstruct import analyze_dataframes
from dbinfer import DBBRDBDataset


class InMemoryAutoGService:
    """Service to run AutoG2 with in-memory DataFrames"""
    
    def __init__(self, cache_strategy="hybrid"):
        self.temp_dir = None
        self.cache_strategy = cache_strategy  # "memory", "disk", "hybrid"
        self.memory_cache = {}  # In-memory cache for small files
        self.cache_size_limit = 10 * 1024 * 1024  # 10MB limit for memory cache
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
        """Get LLM configuration using the exact same function as main.autog2"""
        return autog2_get_llm_config(llm_name)
    
    def get_deepjoin_path(self) -> str:
        """Get the path to DeepJoin model, with fallback options"""
        # Try different possible paths
        possible_paths = [
            "deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27",
            "../deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27",
            "./deepjoin",
            "../deepjoin"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no DeepJoin found, return None (will disable join discovery)
        print("⚠️ DeepJoin model not found. Join discovery will be disabled.")
        return None
    
    def capitalize_first_alpha_concise(self, text: str) -> str:
        """Capitalize the first alphabetic character in text"""
        for i, char in enumerate(text):
            if char.isalpha():
                return text[:i] + text[i:].replace(char, char.upper(), 1)
        return text
    
    def create_temp_workspace(self, dataframes: Dict[str, pd.DataFrame], dataset_name: str) -> str:
        """Create temporary workspace with DataFrames saved as files"""
        self.temp_dir = tempfile.mkdtemp(prefix="autog_webapp_")
        data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save DataFrames as parquet files
        for table_name, df in dataframes.items():
            # Capitalize first letter for LLM processing
            clean_name = self.capitalize_first_alpha_concise(table_name)
            if not clean_name.endswith('.parquet'):
                clean_name += '.parquet'
            
            file_path = os.path.join(data_dir, clean_name)
            df.to_parquet(file_path, index=False)
        
        return self.temp_dir
    
    def generate_metadata_from_dataframes(self, dataframes: Dict[str, pd.DataFrame], dataset_name: str) -> Dict[str, Any]:
        """Generate metadata dictionary from DataFrames"""
        meta_dict = {
            'dataset_name': dataset_name,
            'tables': []
        }
        
        for table_name, df in dataframes.items():
            clean_name = self.capitalize_first_alpha_concise(table_name)
            
            table_meta_dict = {
                'name': clean_name,
                'columns': [],
                'format': 'parquet',
                'source': f'data/{clean_name}.parquet'
            }
            
            for column_name, column_dtype in df.dtypes.to_dict().items():
                dtype = str(column_dtype)
                if column_dtype == 'object':
                    try:
                        pd.to_numeric(df[column_name])
                        dtype = 'int32'
                    except:
                        dtype = 'object'
                
                table_meta_dict['columns'].append({
                    'name': column_name,
                    'dtype': self.dtype_mapping.get(dtype, 'category'),
                })
            
            meta_dict['tables'].append(table_meta_dict)
        
        return meta_dict
    
    def create_dbb_dataset_from_dataframes(self, dataframes: Dict[str, pd.DataFrame], dataset_name: str) -> DBBRDBDataset:
        """Create DBBRDBDataset from in-memory DataFrames"""
        # Create temporary workspace
        temp_path = self.create_temp_workspace(dataframes, dataset_name)
        
        # Generate metadata
        metadata_dict = self.generate_metadata_from_dataframes(dataframes, dataset_name)
        
        # Save metadata
        metadata_path = os.path.join(temp_path, 'metadata.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata_dict, f, default_flow_style=False)
        
        # Cache metadata immediately
        self._cache_file_content(metadata_path)
        
        # Create DBBRDBDataset
        return DBBRDBDataset(temp_path)
    

    
    def run_autog2(
        self,
        dataframes: Dict[str, pd.DataFrame],
        llm_name: str = "sonnet3",
        method: str = "autog-s",
        task_name: str = "custom:kg",
        dataset_name: str = "webapp_dataset",
        seed: int = 0,
        lm_path: str = None
    ) -> Tuple[str, str, str, Dict[str, str]]:
        """
        Run AutoG2 with in-memory DataFrames using the main.autog2 pipeline
        
        Returns:
            Tuple of (agent_history, analysis_result, output_path, generated_files)
        """
        try:
            # Create temporary workspace with DataFrames
            temp_path = self.create_temp_workspace(dataframes, dataset_name)
            
            # Get DeepJoin path
            if lm_path is None:
                lm_path = self.get_deepjoin_path()
                if lm_path is None:
                    lm_path = "deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27"
            
            # Prepare arguments for autog2_main
            import sys
            from unittest.mock import patch
            
            # Mock sys.argv to simulate command line arguments
            mock_argv = [
                'autog2',
                temp_path,           # dataset_path
                llm_name,           # llm_name  
                method,             # method
                task_name,          # task_name
                '--seed', str(seed),
                '--lm-path', lm_path,
                '--dataset-name', dataset_name,
                '--data-format', 'parquet'
            ]
            
            print(f"🚀 Running AutoG2 with arguments: {mock_argv[1:]}")
            
            # Capture stdout to get the agent history
            from io import StringIO
            captured_output = StringIO()
            
            with patch('sys.argv', mock_argv):
                with patch('sys.stdout', captured_output):
                    # Call the main autog2 function directly
                    autog2_main(
                        dataset_path=temp_path,
                        llm_name=llm_name,
                        method=method,
                        task_name=task_name,
                        seed=seed,
                        lm_path=lm_path,
                        dataset_name=dataset_name,
                        data_format='parquet'
                    )
            
            # Get the captured output
            output_text = captured_output.getvalue()
            
            # Read the generated files
            autog_path = os.path.join(temp_path, "autog")
            
            # Read agent history
            history_file = os.path.join(autog_path, "final", "agent_history.txt")
            agent_history = ""
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    agent_history = f.read()
                self._cache_file_content(history_file)
            
            # Read analysis results
            info_file = os.path.join(temp_path, "information.txt")
            analysis_result = ""
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    analysis_result = f.read()
                self._cache_file_content(info_file)
            
            # Collect generated files
            generated_files = self._collect_generated_files(autog_path)
            
            print(f"✅ AutoG2 completed successfully!")
            print(f"📁 Output path: {autog_path}")
            print(f"📊 Generated files: {list(generated_files.keys())}")
            
            return agent_history, analysis_result, autog_path, generated_files
            
        except Exception as e:
            print(f"❌ AutoG2 execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"AutoG2 execution failed: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def _collect_generated_files(self, autog_path: str) -> Dict[str, str]:
        """Collect paths to generated files"""
        files = {}
        
        # Check for common generated files
        final_path = os.path.join(autog_path, "final")
        
        # Agent history
        history_file = os.path.join(final_path, "agent_history.txt")
        if os.path.exists(history_file):
            files['agent_history'] = history_file
        
        # Metadata files
        metadata_file = os.path.join(self.temp_dir, "metadata.yaml")
        if os.path.exists(metadata_file):
            files['metadata'] = metadata_file
            
        # Final metadata (if different)
        final_metadata = os.path.join(final_path, "metadata.yaml")
        if os.path.exists(final_metadata):
            files['final_metadata'] = final_metadata
        
        # Schema images
        schema_png = os.path.join(final_path, "schema.png")
        if os.path.exists(schema_png):
            files['schema_png'] = schema_png
            
        schema_pdf = os.path.join(final_path, "schema.pdf")
        if os.path.exists(schema_pdf):
            files['schema_pdf'] = schema_pdf
        
        # Information file
        info_file = os.path.join(self.temp_dir, "information.txt")
        if os.path.exists(info_file):
            files['information'] = info_file
        
        return files
    
    def get_file_content(self, file_path: str) -> bytes:
        """Get file content as bytes for download"""
        # Try memory cache first
        cached_content = self._get_cached_content(file_path)
        if cached_content is not None:
            print(f"📋 Retrieved {os.path.basename(file_path)} from memory cache")
            return cached_content
        
        # Fall back to disk
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Cache for future use
            self._cache_file_content(file_path, content)
            return content
            
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def _should_cache_in_memory(self, file_path: str, content_size: int = None) -> bool:
        """Determine if a file should be cached in memory based on size and type"""
        if self.cache_strategy == "disk":
            return False
        elif self.cache_strategy == "memory":
            return True
        
        # Hybrid strategy logic
        file_name = os.path.basename(file_path).lower()
        
        # Always cache small text files in memory
        small_text_files = ['.txt', '.yaml', '.yml', '.json', '.csv']
        if any(file_name.endswith(ext) for ext in small_text_files):
            if content_size is None:
                try:
                    content_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                except:
                    content_size = 0
            return content_size < self.cache_size_limit
        
        # Never cache large binary files in memory
        large_binary_files = ['.pkl', '.parquet', '.npy', '.npz']
        if any(file_name.endswith(ext) for ext in large_binary_files):
            return False
        
        # Cache images only if small
        image_files = ['.png', '.jpg', '.jpeg', '.pdf']
        if any(file_name.endswith(ext) for ext in image_files):
            if content_size is None:
                try:
                    content_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                except:
                    content_size = 0
            return content_size < 1024 * 1024  # 1MB limit for images
        
        return False
    
    def _cache_file_content(self, file_path: str, content: bytes = None) -> None:
        """Cache file content in memory if appropriate"""
        if content is None:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
            except:
                return
        
        if self._should_cache_in_memory(file_path, len(content)):
            self.memory_cache[file_path] = content
            print(f"📝 Cached {os.path.basename(file_path)} in memory ({len(content):,} bytes)")
    
    def _get_cached_content(self, file_path: str) -> Optional[bytes]:
        """Get cached content from memory"""
        return self.memory_cache.get(file_path)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache usage"""
        total_memory_size = sum(len(content) for content in self.memory_cache.values())
        
        return {
            'strategy': self.cache_strategy,
            'memory_files': len(self.memory_cache),
            'memory_size': total_memory_size,
            'memory_size_mb': total_memory_size / (1024 * 1024),
            'cached_files': list(os.path.basename(path) for path in self.memory_cache.keys())
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()