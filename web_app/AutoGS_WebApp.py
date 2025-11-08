#!/usr/bin/env python3
"""
AutoG-S Web Application
Clean, simplified interface for Automatic Table-to-Graph Generation
"""

import os
import sys
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Setup paths
current_dir = Path(__file__).parent
root_dir = current_dir.parent

# Add necessary paths
paths_to_add = [
    str(current_dir),
    str(root_dir),
    str(root_dir / "multi-table-benchmark"),
    str(root_dir / "dbinfer"),
    str(root_dir / "models"),
    str(root_dir / "prompts"),
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import AutoG-S components
try:
    from services.simple_autog_service import SimpleAutoGService
    from webutils.config import LLM_MODELS, AUTOG_CONFIG, DEFAULT_CONFIG
except ImportError as e:
    st.error(f"Failed to import AutoG-S components: {e}")
    st.error("Make sure you're running from the correct directory with all dependencies installed.")
    
    # Write import error to log file
    import datetime
    import traceback
    import sys
    error_log_path = "autogs_import_error.log"
    try:
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"AutoG-S Import Error Log - {datetime.datetime.now()}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Import Error: {str(e)}\n")
            f.write(f"Error Type: {type(e).__name__}\n")
            f.write(f"\nFull Traceback:\n")
            f.write(traceback.format_exc())
            f.write(f"\nPython Path:\n")
            for i, path in enumerate(sys.path[:15]):
                f.write(f"  {i}: {path}\n")
            f.write(f"\nCurrent Working Directory: {os.getcwd()}\n")
            f.write(f"{'='*80}\n")
        st.info(f"📝 Import error log written to: `{error_log_path}`")
    except Exception as log_error:
        st.warning(f"Could not write import error log: {log_error}")
    
    st.stop()

# Load environment variables
load_dotenv()

# Create startup log
def log_startup_info():
    """Log startup information for debugging"""
    startup_log_path = "autogs_startup.log"
    try:
        with open(startup_log_path, "w", encoding="utf-8") as f:
            f.write(f"AutoG-S Web App Startup Log - {datetime.now()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Current Working Directory: {os.getcwd()}\n")
            f.write(f"Python Executable: {sys.executable}\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"\nPython Path (first 15 entries):\n")
            for i, path in enumerate(sys.path[:15]):
                f.write(f"  {i}: {path}\n")
            f.write(f"\nEnvironment Variables:\n")
            for key in ['PYTHONPATH', 'CONDA_DEFAULT_ENV', 'AWS_ACCESS_KEY_ID', 'AWS_DEFAULT_REGION']:
                value = os.environ.get(key, 'Not set')
                if key == 'AWS_ACCESS_KEY_ID' and value != 'Not set':
                    value = value[:8] + '...'  # Mask sensitive info
                f.write(f"  {key}: {value}\n")
            f.write(f"{'='*60}\n")
    except Exception as e:
        print(f"Could not write startup log: {e}")

# Log startup info
log_startup_info()

# Page configuration
st.set_page_config(
    page_title="AutoG-S - Automatic Table-to-Graph Generation",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .file-info {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .processing-section {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #1f77b4;
        margin: 2rem 0;
        text-align: center;
    }
    .round-indicator {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .log-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .round-header {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    /* Enhanced Run AutoG-S Button Styling */
    .stButton > button[key="run_autogs_main_btn"] {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        padding: 0.75rem 2rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[key="run_autogs_main_btn"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* Center the button container */
    div[data-testid="column"]:has(button[key="run_autogs_main_btn"]) {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class TaskConfig:
    """Task configuration"""
    llm_model: str
    method: str
    task: str
    dataset_name: str
    custom_task: Optional[str] = None
    use_custom_task: bool = False
    cache_strategy: str = "hybrid"
    seed: int = 0
    max_rounds: int = 20
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Ensure method is never None or empty
        if not self.method or self.method == "None":
            self.method = "autog-s"  # Default fallback
        
        # Ensure method is valid
        valid_methods = ["autog-s"]
        if self.method not in valid_methods:
            self.method = "autog-s"

class SessionState:
    """Session state manager"""
    
    @staticmethod
    def init():
        """Initialize session state"""
        defaults = {
            "session_id": uuid.uuid4().hex[:8],
            "uploaded_files": {},
            "dataframes": {},
            "task_running": False,
            "results": None,
            "autog_service": SimpleAutoGService(),
            "aws_credentials_valid": False,
            "current_round": 0,
            "rounds_completed": 0,
            "processing_logs": [],
            "round_logs": {},

            "processing_started": False,
            "widget_counter": 0,
            "uploader_reset_counter": 0,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

# AWS credentials are loaded from environment variables
# No validation is done in the UI - credentials are validated when used

def load_file_to_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    """Load uploaded file to DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.tsv', '.tab')):
            return pd.read_csv(uploaded_file, sep='\t')
        elif uploaded_file.name.endswith(('.txt', '.dat')):
            # Try to detect delimiter automatically for .txt and .dat files
            try:
                return pd.read_csv(uploaded_file, sep=None, engine='python')
            except:
                # Fallback to comma separator if auto-detection fails
                uploaded_file.seek(0)  # Reset file pointer
                return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.parquet', '.pq', '.pqt')):
            return pd.read_parquet(uploaded_file)
        elif uploaded_file.name.endswith(('.npy', '.npz')):
            import io
            bytes_data = uploaded_file.read()
            arr = np.load(io.BytesIO(bytes_data), allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):
                data = {}
                for k in arr.files:
                    data[k] = arr[k] if k != 'feat' else list(arr[k])
                return pd.DataFrame(data)
            else:
                return pd.DataFrame(arr)
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return None

def render_sidebar():
    """Render sidebar configuration"""
    with st.sidebar:
        
        # Check if AutoG-S is currently running
        task_running = st.session_state.get('task_running', False)
        
        # AWS Credentials Section
        st.markdown("### 🔐 AWS Credentials")
        
        # Load AWS credentials from environment variables
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
        aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        
        # Check if credentials are present
        if aws_access_key and aws_secret_key:
            st.success("✅ AWS credentials loaded from environment variables")
            st.session_state.aws_credentials_valid = True
            
            # Show masked credential info
            st.info(f"🔑 Access Key: `{aws_access_key[:-4]}...` | Region: `{aws_region}`")
            if aws_session_token:
                st.info("🔑 Session token present (temporary credentials)")
        else:
            st.error("❌ AWS credentials not found in environment variables")
            st.session_state.aws_credentials_valid = False
            st.markdown("Please export your AWS credentials before running the web app:")
            st.code("""
export AWS_ACCESS_KEY_ID=<your_access_key_here>
export AWS_SECRET_ACCESS_KEY=<your_secret_key_here>
export AWS_SESSION_TOKEN=<your_session_token_here>  # Optional
export AWS_DEFAULT_REGION=<your_default_region>
            """, language="bash")
        
        st.markdown("---")
        
        # Model Configuration
        st.markdown("### 🤖 LLM Configuration")
        # Use session counter to reset widgets
        widget_counter = st.session_state.get('widget_counter', 0)
        
        # LLM Model selection with default
        llm_options = list(LLM_MODELS.keys())
        llm_index = 0  # Default to first item
        default_llm = DEFAULT_CONFIG.get("llm_model")
        if default_llm and default_llm in llm_options:
            llm_index = llm_options.index(default_llm)
        elif not llm_options:
            # No options available
            llm_options = [None]
            llm_index = 0
        
        llm_model = st.selectbox(
            "LLM Model",
            options=llm_options,
            index=llm_index,
            help="Select the language model for AutoG-S processing",
            disabled=task_running,
            key=f"llm_model_{widget_counter}"
        )
        
        st.info(LLM_MODELS[llm_model]["description"])
        
        # Method Configuration
        st.markdown("### ⚙️ Processing Configuration")
        # Method selection with default
        method_options = AUTOG_CONFIG["methods"]
        method_index = 0  # Default to first item
        default_method = DEFAULT_CONFIG.get("method")
        if default_method and default_method in method_options:
            method_index = method_options.index(default_method)
        
        method = st.selectbox(
            "Method",
            options=method_options,
            index=method_index,
            help="AutoG-S processing method",
            disabled=task_running,
            key=f"method_{widget_counter}"
        )
        
        # Ensure method is never None - always use "autog-s"
        if method is None or method == "":
            method = "autog-s"
        
        # Task Configuration
        # Get default dataset and task from separate keys, fallback to first available or None
        dataset_options = list(AUTOG_CONFIG["datasets"].keys())
        default_dataset = DEFAULT_CONFIG.get("dataset")
        if not default_dataset or default_dataset not in dataset_options:
            default_dataset = dataset_options[0] if dataset_options else None
        
        # Dataset selection with default
        dataset_index = 0  # Default to first item
        if not dataset_options:
            # No options available
            dataset_options = [None]
            dataset_index = 0
        elif default_dataset and default_dataset in dataset_options:
            dataset_index = dataset_options.index(default_dataset)
        
        dataset_type = st.selectbox(
            "Dataset Type",
            options=dataset_options,
            index=dataset_index,
            help="Type of dataset for task selection",
            disabled=task_running,
            key=f"dataset_type_{widget_counter}"
        )
        
        # Task selection with default
        task_options = AUTOG_CONFIG["datasets"].get(dataset_type, []) if dataset_type else []
        task_index = 0  # Default to first item
        
        # Get default task, fallback to first available for the selected dataset or None
        default_task_name = DEFAULT_CONFIG.get("task")
        if (dataset_type == default_dataset and 
            default_task_name and 
            default_task_name in task_options):
            task_index = task_options.index(default_task_name)
        # If default task doesn't exist for current dataset, use first available or None
        elif not default_task_name or default_task_name not in task_options:
            if not task_options:
                # No options available
                task_options = [None]
                task_index = 0
            else:
                task_index = 0  # First available task for this dataset
        
        task = st.selectbox(
            "Task",
            options=task_options,
            index=task_index,
            help="Specific task within the dataset type",
            disabled=task_running,
            key=f"task_{widget_counter}"
        )
        
        # Show task description
        try:
            from prompts.task import get_task_description
            task_desc = get_task_description(dataset_type, task)
            if task_desc:
                st.info(f"📝 **Task Description:** {task_desc}")
        except:
            pass  # If task description not available, don't show anything
        
        # Custom task option
        use_custom_task = st.checkbox("Use Custom Task Description", disabled=task_running, key=f"use_custom_task_{widget_counter}")
        custom_task = None
        if use_custom_task:
            custom_task = st.text_area(
                "Custom Task Description",
                placeholder="Describe your custom task here... E.g., 'This task is to find the primary keys and foreign keys among the given tables.'",
                help="Provide a detailed description of your custom task.",
                disabled=task_running,
                key=f"custom_task_{widget_counter}"
            )
            # Ensure custom_task is None if empty string
            if not custom_task or custom_task.strip() == "":
                custom_task = None
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            cache_strategy = st.selectbox(
                "Cache Strategy",
                options=["hybrid", "memory", "disk"],
                index=0,
                help="How to handle intermediate files",
                disabled=task_running,
                key=f"cache_strategy_{widget_counter}"
            )
            
            max_rounds = st.number_input(
                "Maximum Rounds",
                min_value=1,
                max_value=50,
                value=20,
                help="Maximum number of processing rounds for AutoG-S agent",
                disabled=task_running,
                key=f"max_rounds_{widget_counter}"
            )
            
            seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=99999,
                value=DEFAULT_CONFIG["seed"],
                help="Random seed for reproducibility",
                disabled=task_running,
                key=f"seed_{widget_counter}"
            )
        
        # Final validation before creating TaskConfig
        if method is None or method == "":
            method = "autog-s"
        
        return TaskConfig(
            llm_model=llm_model,
            method=method,
            task=task,
            dataset_name=dataset_type,
            custom_task=custom_task if use_custom_task else None,
            use_custom_task=use_custom_task,
            cache_strategy=cache_strategy,
            seed=seed,
            max_rounds=max_rounds
        )

def render_file_upload():
    """Render file upload section"""
    st.markdown('<div class="section-header">📊 Data Upload</div>', unsafe_allow_html=True)
    
    # Only disable uploader during processing, not after
    task_running = st.session_state.get('task_running', False)
    uploader_disabled = task_running
    
    # Use dynamic key to reset uploader when explicitly clearing data
    uploader_key = f"file_uploader_{st.session_state.get('uploader_reset_counter', 0)}"
    
    uploaded_files = st.file_uploader(
        "Upload your data files",
        type=["csv", "tsv", "txt", "dat", "tab", "parquet", "pq", "pqt", "npy", "npz"],
        accept_multiple_files=True,
        help="Upload data files: CSV (.csv, .tsv, .txt, .dat, .tab), Parquet (.parquet, .pq, .pqt), or NumPy (.npy, .npz). Duplicate files will be automatically deduplicated." if not uploader_disabled else "Disabled during processing",
        disabled=uploader_disabled,
        key=uploader_key
    )
    
    # Always keep dataframes consistent with uploaded_files widget
    # This ensures perfect synchronization between widget display and session state
    
    current_dataframes = st.session_state.get('dataframes', {})
    
    if uploaded_files:
        # Deduplicate uploaded files by filename to handle multiple uploads of same file
        unique_files = {}
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            if filename not in unique_files:
                unique_files[filename] = uploaded_file
        
        # Check if there were duplicates and show info
        if len(uploaded_files) > len(unique_files):
            duplicate_count = len(uploaded_files) - len(unique_files)
            st.info(f"ℹ️ Note: {duplicate_count} duplicate file(s) detected in upload widget. Processing {len(unique_files)} unique file(s).")
        
        # Files are present in widget - process them and update dataframes
        new_dataframes = {}
        for uploaded_file in unique_files.values():
            df = load_file_to_dataframe(uploaded_file)
            if df is not None:
                table_name = os.path.splitext(uploaded_file.name)[0]
                new_dataframes[table_name] = df
        
        # Update session state to match widget (using deduplicated files)
        st.session_state.dataframes = new_dataframes
        st.session_state.uploaded_files = unique_files
        
        # If this is a change from current state, refresh UI
        if set(new_dataframes.keys()) != set(current_dataframes.keys()):
            st.rerun()
    else:
        # No files in widget - check if we should preserve dataframes (post-processing state)
        has_results = bool(st.session_state.get('results', None))
        rounds_completed = st.session_state.get('rounds_completed', 0)
        has_error = bool(st.session_state.get('processing_error', None))
        
        # Check if results were manually cleared (preserve data in this case)
        results_manually_cleared = st.session_state.get('results_manually_cleared', False)
        
        # Only clear dataframes if user actively removed files AND there's no processing state
        # Preserve dataframes if: has results, has completed rounds, has error, or results were manually cleared
        if current_dataframes and not has_results and rounds_completed == 0 and not has_error and not results_manually_cleared:
            # No results, no processing, no error, not manually cleared - user actively removed files, clear dataframes
            st.session_state.dataframes = {}
            st.session_state.uploaded_files = {}
            
            # Clear processing states
            keys_to_clear = ['current_round', 'rounds_completed', 'stop_requested', 'processing_started']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear log data
            st.session_state.processing_logs = []
            st.session_state.round_logs = {}
            
            st.rerun()
        # If has_results or rounds_completed > 0 or has_error, preserve dataframes (post-processing/error state)
    # Show current data status and clear button (refresh state each time)
    current_dataframes = st.session_state.get('dataframes', {})
    
    if current_dataframes:
        st.success(f"✅ {len(current_dataframes)} table(s) loaded")
        for table_name, df in current_dataframes.items():
            st.markdown(f"- **{table_name}**: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Only disable buttons during processing, not after
        task_running = st.session_state.get('task_running', False)
        buttons_disabled = task_running
        
        # Clear data button (disabled after processing)
        if st.button("🗑️ Clear All Data", help="Clear all uploaded data" if not buttons_disabled else "Disabled during processing", key="clear_data_upload_btn", disabled=buttons_disabled):
            # Clear data but preserve results (same as "before running" state)
            st.session_state.dataframes = {}
            st.session_state.uploaded_files = {}
            
            # Clear processing states but preserve results
            keys_to_clear = ['current_round', 'rounds_completed', 'task_running', 'stop_requested', 'processing_started', 'processing_error']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear log data but preserve results
            st.session_state.processing_logs = []
            st.session_state.round_logs = {}
            
            # Reset file uploader widget to clear displayed files
            st.session_state.uploader_reset_counter = st.session_state.get('uploader_reset_counter', 0) + 1
            
            st.rerun()
        
        # Data preview section (collapsible)
        with st.expander("📊 Data Preview & Statistics", expanded=False):
            render_data_preview()
    else:
        st.info("📁 No data uploaded yet")
        st.markdown("""
        **Supported file formats:**
        - 📊 **CSV/Text**: `.csv`, `.tsv`, `.txt`, `.dat`, `.tab` files
        - 🗃️ **Parquet**: `.parquet`, `.pq`, `.pqt` files  
        - 🔢 **NumPy**: `.npy`, `.npz` files
        
        💡 **Tips**: 
        - TSV files use tab separators, TXT files auto-detect delimiters
        - Parquet files support multiple extensions (.parquet, .pq, .pqt)
        - NumPy .npz files can contain multiple arrays
        """)
    
    return bool(current_dataframes)

def render_data_preview():
    """Render data preview section"""
    if not st.session_state.dataframes:
        return
        
    st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
    
    # Summary metrics
    total_rows = sum(df.shape[0] for df in st.session_state.dataframes.values())
    total_cols = sum(df.shape[1] for df in st.session_state.dataframes.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tables", len(st.session_state.dataframes))
    with col2:
        st.metric("Total Rows", f"{total_rows:,}")
    with col3:
        st.metric("Total Columns", f"{total_cols:,}")
    
    # Data preview tabs
    tabs = st.tabs(list(st.session_state.dataframes.keys()))
    
    for i, (table_name, df) in enumerate(st.session_state.dataframes.items()):
        with tabs[i]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.dataframe(df.head(10), width='stretch')
            
            with col2:
                st.markdown("**Data Info:**")
                st.markdown(f"- Shape: {df.shape[0]:,} × {df.shape[1]}")
                st.markdown(f"- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                missing_count = df.isnull().sum().sum()
                completeness = (df.size - missing_count) / df.size * 100
                st.markdown(f"- Missing: {missing_count:,} ({100-completeness:.1f}%)")
                
                # Data types summary
                dtype_counts = df.dtypes.value_counts()
                st.markdown("**Data Types:**")
                for dtype, count in dtype_counts.items():
                    st.markdown(f"- {dtype}: {count}")



def render_processing_section(config: TaskConfig):
    """Render processing section"""
    st.markdown('<div class="section-header">🚀 AutoG-S Processing Center</div>', unsafe_allow_html=True)
    
    # Configuration summary in a clean format
    with st.expander("📋 Current Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model:** {LLM_MODELS[config.llm_model]['name']}")
            st.markdown(f"**Method:** {config.method}")
            # Show "Custom Task" if custom task checkbox is checked, otherwise show the selected task
            if config.use_custom_task:
                st.markdown(f"**Task:** Custom Task")
            else:
                st.markdown(f"**Task:** {config.dataset_name}:{config.task}")
        with col2:
            st.markdown(f"**Max Rounds:** {config.max_rounds}")
            st.markdown(f"**Cache Strategy:** {config.cache_strategy}")
            st.markdown(f"**Seed:** {config.seed}")
        
        if config.use_custom_task:
            if config.custom_task and config.custom_task.strip():
                st.markdown(f"**Custom Description:** {config.custom_task}")
            else:
                st.markdown(f"**Custom Description:** *Not specified yet*")
        
        if st.session_state.dataframes:
            st.markdown(f"**Tables:** {', '.join(st.session_state.dataframes.keys())}")
        else:
            st.markdown("**Tables:** None uploaded")
    
    # Requirements check moved to right sidebar - check readiness for button
    current_dataframes_check = st.session_state.get('dataframes', {})
    has_data = bool(current_dataframes_check) and len(current_dataframes_check) > 0
    has_credentials = st.session_state.get('aws_credentials_valid', False)
    not_running = not st.session_state.get('task_running', False)
    
    # Check if custom task has description (for warning purposes only)
    custom_task_has_description = True
    if config.use_custom_task:
        custom_task_has_description = bool(config.custom_task and config.custom_task.strip())
    
    ready_to_run = has_data and has_credentials and not_running
    
    # Create a perfectly centered button with better spacing
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create detailed help text for debugging
        if ready_to_run:
            help_text = "Click to start -S table-to-graph generation"
        else:
            missing = []
            if not has_data:
                missing.append("upload data")
            if not has_credentials:
                missing.append("configure AWS credentials")
            if not not_running:
                missing.append("wait for current processing to complete")
            help_text = f"Missing: {', '.join(missing)}"
        
        run_button = st.button(
            "🚀 **RUN AUTOG-S**",
            type="primary",
            disabled=not ready_to_run,
            help=help_text,
            key="run_autogs_main_btn",
            use_container_width=True
        )
    
    # Close the styled container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show what will happen when button is clicked
    if ready_to_run:
        st.info("💡 This will analyze your data and generate a graph schema using AI.")
        # Show warning for custom task without description (but don't disable)
        if config.use_custom_task and not custom_task_has_description:
            st.warning("⚠️ Custom task is enabled but no description provided. AutoG-S will use the default task behavior.")
    elif st.session_state.get('task_running', False):
        st.info("🔄 AutoG-S is currently processing. Please wait for completion or stop the process above.")
    else:
        st.warning("⚠️ Button will be enabled when all requirements are met.")
    
    # Process AutoG-S when button is clicked
    if run_button:
        # Clear any previous run state
        if 'processing_error' in st.session_state:
            del st.session_state.processing_error
        if 'results' in st.session_state:
            del st.session_state.results
        
        # Reset processing state
        st.session_state.current_round = 0
        st.session_state.rounds_completed = 0
        st.session_state.processing_logs = []
        st.session_state.round_logs = {}
        
        # Set task_running immediately to disable configuration
        st.session_state.task_running = True
        # Force immediate rerun to update sidebar disabled state
        st.rerun()
    
    # Show processing section when processing or when we have logs/results
    task_running = st.session_state.get('task_running', False)
    has_logs = bool(st.session_state.get('round_logs'))
    has_results = bool(st.session_state.get('results'))
    has_error = bool(st.session_state.get('processing_error'))
    
    # Display error if one occurred
    if has_error and not task_running:
        error_info = st.session_state.processing_error
        
        # Show user-friendly error message
        if "already been added" in error_info['exception'] or "has already been added" in error_info['exception']:
            st.error("❌ AutoG-S encountered a system error during processing. This may be due to LLM response issues. Please try running again.")
        else:
            st.error(f"❌ {error_info['message']}")
        
        with st.expander("🔍 Error Details", expanded=True):
            st.markdown("**System Error from AutoG Pipeline**")
            st.markdown(f"**Error Type:** `{error_info['type']}`")
            st.markdown(f"**Error Message:** {error_info['exception']}")
            
            st.markdown("**Configuration:**")
            for key, value in error_info['config'].items():
                st.markdown(f"- {key}: `{value}`")
            
            st.markdown("**Full Traceback:**")
            st.code(error_info['traceback'], language='python')
            
            st.markdown("**What to do:**")
            if "already been added" in error_info['exception'] or "has already been added" in error_info['exception']:
                st.markdown("- This is a system error from the AutoG pipeline (possibly due to LLM response issues)")
                st.markdown("- **Try running again** - the LLM may produce a different result")
                st.markdown("- If the error persists, try with a different seed value or task")
            elif "credentials" in error_info['exception'].lower() or "bedrock" in error_info['exception'].lower():
                st.markdown("- Check that AWS credentials are valid and have Bedrock access")
                st.markdown("- Verify credentials are exported in your environment")
            elif "import" in error_info['exception'].lower() or "module" in error_info['exception'].lower():
                st.markdown("- Ensure you're running in the `autog-cpu` conda environment")
                st.markdown("- Verify that all AutoG-S dependencies are installed")
            else:
                st.markdown("- This appears to be a system error from the AutoG pipeline")
                st.markdown("- **Try running again** - the issue may be transient")
                st.markdown("- If the error persists, try with different configuration or data")
        
    
    if task_running or has_logs or has_results:
        
        # Processing status header (always show when processing or completed)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            round_display = st.empty()
        with col2:
            progress_bar = st.progress(0)
        with col3:
            status_indicator = st.empty()
        
        # Status text
        status_text = st.empty()
        
        # Update progress indicators based on current state
        if task_running:
            # During processing - these will be updated by run_autogs
            current_round = st.session_state.get('current_round', 0)
            rounds_completed = st.session_state.get('rounds_completed', 0)
            
            if current_round > 0:
                round_display.metric("Current Progress", f"Round {current_round}")
                progress_bar.progress(min(8 + (current_round * 10), 90))
                status_indicator.info("🔄 Running")
                status_text.text(f"🚀 AutoG-S agent: Round {current_round}")
            else:
                round_display.metric("Current Progress", "Starting...")
                progress_bar.progress(8)
                status_indicator.info("🔄 Initializing")
                status_text.text("🔄 Initializing AutoG-S...")

             # show move execution details
            round_logs = st.session_state.get('round_logs', {})
            if len(round_logs) > 0 and current_round in round_logs:
                cur_round_logs = round_logs[current_round]
                # During processing, round is not yet completed
                show_current_logs(cur_round_logs, round_completed=False)

        else:
            # After processing - show final state
            rounds_completed = st.session_state.get('rounds_completed', 0)
            if rounds_completed > 0:
                round_display.metric("Rounds Completed", rounds_completed)
                progress_bar.progress(100)
                status_indicator.success("✅ Complete")
                status_text.text("✅ Processing completed!")
        
        # Check if we should start processing (after the rerun)
        if task_running and not st.session_state.get('processing_started', False):
            # Mark that processing has started to avoid re-triggering
            st.session_state.processing_started = True
            # Store progress indicators in session state so run_autogs can update them
            st.session_state.progress_indicators = {
                'round_display': round_display,
                'progress_bar': progress_bar,
                'status_indicator': status_indicator,
                'status_text': status_text
            }
            run_autogs(config)


def show_current_logs(cur_logs, round_completed):
    round_start_logs = []
    move_logs = []
    error_logs = []

    current_move_logs = []

    for log in cur_logs:
        message = log['message']
        if "=== Starting Round" in message:
            round_start_logs.append(log)
        elif "Executing move:" in message:
            # If we have a previous move, save it
            if current_move is not None:
                move_logs.append({
                    'move': current_move,
                    'logs': current_move_logs.copy()
                })
            
            # Start new move
            current_move = message.replace("Executing move: ", "")
            current_move_logs = [log]
        elif "ERROR:" in message:
            if current_move is not None:
                current_move_logs.append(log)
            else:
                error_logs.append(log)
        else:
            if current_move is not None:
                current_move_logs.append(log)
            # else: ignore other logs since we don't display them


    # Create expandable section for each round
    current_round = st.session_state.get('current_round', 0)
    task_running = st.session_state.get('task_running', False)
    is_current = (current_round > round_completed and task_running)
    round_status = "🔄" if is_current else "✅"
    
    # Count different types of entries for better summary
    move_count = len(move_logs)
    error_count = len(error_logs)
    
    # Create a more informative title
    title_parts = [f"Round {round_num}"]
    if move_count > 0:
        title_parts.append(f"{move_count} moves")
    if error_count > 0:
        title_parts.append(f"{error_count} errors")
    
    title = f"{round_status} {' - '.join(title_parts)}"
    
    with st.expander(title, expanded=is_current):
        if logs:
            # Show round start information
            if round_start_logs:
                for log in round_start_logs:
                    st.markdown(f"**[{log['timestamp']}]** {log['message']}")
            
            # Show moves in separate expandable sections
            if move_logs:
                st.markdown("**Moves Executed:**")
                for i, move_data in enumerate(move_logs):
                    move_desc = move_data['move']
                    move_logs_list = move_data['logs']
                    
                    # Simple move title without action parsing
                    move_title = f"Move {i+1}"
                    
                    # Check if move has errors
                    has_errors = any("ERROR:" in log['message'] for log in move_logs_list)
                    move_status = "❌" if has_errors else "✅"
                    
                    with st.expander(f"{move_status} {move_title}", expanded=False):
                        # Show move details
                        st.code(move_desc, language='json')
                        
                        # Show move logs
                        if len(move_logs_list) > 1:  # More than just the "Executing move" log
                            st.markdown("**Execution Log:**")
                            for log in move_logs_list[1:]:  # Skip the first "Executing move" log
                                timestamp = log['timestamp']
                                message = log['message']
                                if "ERROR:" in message:
                                    st.error(f"[{timestamp}] {message}")
                                else:
                                    st.text(f"[{timestamp}] {message}")
            
            # Show standalone errors
            if error_logs:
                st.markdown("**Errors:**")
                for log in error_logs:
                    st.error(f"[{log['timestamp']}] {log['message']}")             

            
            # Show compact summary
            if len(logs) > 1:
                start_time = logs[0]['timestamp']
                end_time = logs[-1]['timestamp']
                st.caption(f"⏱️ {start_time} - {end_time} ({len(logs)} entries)")
        else:
            st.info("No logs available for this round")


def render_move_based_logs():
    """Render move-based logs integrated into the AutoG-S Agent Running section"""
    round_logs = st.session_state.get('round_logs', {})
    
    if not round_logs:
        return
    
    st.markdown("---")
    st.markdown("#### 📋 Round Execution Details")

    sorted_rounds = sorted(round_logs.keys())
    rounds_to_show = []
    
    # Show detailed logs for each round with move organization
    for round_num in sorted_rounds:
        logs = round_logs[round_num]
        
        # Not show the last round without moves
        if len(logs) <= 2 and round_num == sorted_rounds[-1]:
            break
        
        # Organize logs by type
        round_start_logs = []
        move_logs = []
        error_logs = []
        
        current_move = None
        current_move_logs = []
        
        for log in logs:
            message = log['message']
            if "=== Starting Round" in message:
                round_start_logs.append(log)
            elif "Executing move:" in message:
                # If we have a previous move, save it
                if current_move is not None:
                    move_logs.append({
                        'move': current_move,
                        'logs': current_move_logs.copy()
                    })
                
                # Start new move
                current_move = message.replace("Executing move: ", "")
                current_move_logs = [log]
            elif "ERROR:" in message:
                if current_move is not None:
                    current_move_logs.append(log)
                else:
                    error_logs.append(log)
            else:
                if current_move is not None:
                    current_move_logs.append(log)
                # else: ignore other logs since we don't display them
        
        # Don't forget the last move
        if current_move is not None:
            move_logs.append({
                'move': current_move,
                'logs': current_move_logs
            })
        
        # Create expandable section for each round
        current_round = st.session_state.get('current_round', 0)
        task_running = st.session_state.get('task_running', False)
        is_current = (round_num == current_round and task_running)
        round_status = "🔄" if is_current else "✅"
        
        # Count different types of entries for better summary
        move_count = len(move_logs)
        error_count = len(error_logs)
        
        # Create a more informative title
        title_parts = [f"Round {round_num}"]
        if move_count > 0:
            title_parts.append(f"{move_count} moves")
        if error_count > 0:
            title_parts.append(f"{error_count} errors")
        
        title = f"{round_status} {' - '.join(title_parts)}"
        
        with st.expander(title, expanded=is_current):
            if logs:
                # Show round start information
                if round_start_logs:
                    for log in round_start_logs:
                        st.markdown(f"**[{log['timestamp']}]** {log['message']}")
                
                # Show moves in separate expandable sections
                if move_logs:
                    st.markdown("**Moves Executed:**")
                    for i, move_data in enumerate(move_logs):
                        move_desc = move_data['move']
                        move_logs_list = move_data['logs']
                        
                        # Simple move title without action parsing
                        move_title = f"Move {i+1}"
                        
                        # Check if move has errors
                        has_errors = any("ERROR:" in log['message'] for log in move_logs_list)
                        move_status = "❌" if has_errors else "✅"
                        
                        with st.expander(f"{move_status} {move_title}", expanded=False):
                            # Show move details
                            st.code(move_desc, language='json')
                            
                            # Show move logs
                            if len(move_logs_list) > 1:  # More than just the "Executing move" log
                                st.markdown("**Execution Log:**")
                                for log in move_logs_list[1:]:  # Skip the first "Executing move" log
                                    timestamp = log['timestamp']
                                    message = log['message']
                                    if "ERROR:" in message:
                                        st.error(f"[{timestamp}] {message}")
                                    else:
                                        st.text(f"[{timestamp}] {message}")
                
                # Show standalone errors
                if error_logs:
                    st.markdown("**Errors:**")
                    for log in error_logs:
                        st.error(f"[{log['timestamp']}] {log['message']}")
                
                
                # Show compact summary
                if len(logs) > 1:
                    start_time = logs[0]['timestamp']
                    end_time = logs[-1]['timestamp']
                    st.caption(f"⏱️ {start_time} - {end_time} ({len(logs)} entries)")
            else:
                st.info("No logs available for this round")

        
def add_log_entry(message: str, round_num: int = None):
    """Add a log entry to the session state"""
    try:
        from datetime import datetime as dt
        timestamp = dt.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'message': str(message)  # Ensure message is string
        }
        
        if round_num is not None:
            # Add to round-specific logs
            if 'round_logs' not in st.session_state:
                st.session_state.round_logs = {}
            if round_num not in st.session_state.round_logs:
                st.session_state.round_logs[round_num] = []
            st.session_state.round_logs[round_num].append(log_entry)
        else:
            # Add to general processing logs
            if 'processing_logs' not in st.session_state:
                st.session_state.processing_logs = []
            st.session_state.processing_logs.append(log_entry)
    except Exception:
        # If logging fails, don't crash the app
        pass

def run_autogs(config: TaskConfig):
    """Run AutoG-S processing with real round tracking and logging"""
    # task_running is already set to True before this function is called
    st.session_state.stop_requested = False
    
    # Initialize round tracking and logs
    st.session_state.current_round = 0
    st.session_state.rounds_completed = 0
    st.session_state.processing_logs = []
    st.session_state.round_logs = {}
    
    # Clear the manually cleared flag when starting new processing
    if 'results_manually_cleared' in st.session_state:
        del st.session_state['results_manually_cleared']
    
    # Get progress indicators from session state (created in main processing section)
    progress_indicators = st.session_state.get('progress_indicators', {})
    round_display = progress_indicators.get('round_display')
    progress_bar = progress_indicators.get('progress_bar')
    status_indicator = progress_indicators.get('status_indicator')
    status_text = progress_indicators.get('status_text')
    
    try:
        add_log_entry("Starting -S processing...")
        
        # Debug: Check if autog_service exists
        if 'autog_service' not in st.session_state:
            raise AttributeError("autog_service not found in session state. SessionState.init() may not have been called.")
        
        # Update service configuration
        service = st.session_state.autog_service
        if service is None:
            raise AttributeError("autog_service is None. Service initialization may have failed.")
            
        service.cache_strategy = config.cache_strategy
        
        # Check if backend is available
        from services.simple_autog_service import BACKEND_AVAILABLE
        if not BACKEND_AVAILABLE:
            raise ImportError("AutoG-S backend modules are not available. Please check your environment setup.")
        
        # Initial setup
        add_log_entry("Initializing AutoG-S system...")
        if status_indicator:
            status_indicator.info("🔄 Initializing")
        if status_text:
            status_text.text("🔄 Initializing AutoG-S and preparing data...")
        if round_display:
            round_display.metric("Current Progress", "Setup")
        if progress_bar:
            progress_bar.progress(8)
            
        add_log_entry("Loading data and generating metadata...")
        add_log_entry(f"Configuration: Model={config.llm_model}, Method={config.method}, Max Rounds={config.max_rounds}")
        
        # Validate and fix method value
        add_log_entry(f"DEBUG: Method value = '{config.method}' (type: {type(config.method)})")
        
        # Fix method if it's invalid
        if config.method is None or config.method == "None" or str(config.method).strip() == "":
            add_log_entry(f"WARNING: Invalid method '{config.method}', using default 'autog-s'")
            config.method = "autog-s"  # Force to valid default
        
        # Ensure method is in valid list
        valid_methods = ["autog-s"]
        if config.method not in valid_methods:
            add_log_entry(f"WARNING: Unknown method '{config.method}', using 'autog-s'")
            config.method = "autog-s"
        
        # Start AutoG-S processing - show preparing state
        if status_indicator:
            status_indicator.warning("🚀 Running")
        if status_text:
            status_text.text("🚀 Starting AutoG-S agent...")
        if round_display:
            round_display.metric("Current Progress", "Starting...")
        if progress_bar:
            progress_bar.progress(8)
        
        add_log_entry("Starting AutoG-S agent execution...")
        
        # Define progress callback to update UI and logs
        def progress_callback(message):
            import re
            
            # Parse different types of AutoG-S output
            round_match = re.search(r'Round:\s*(\d+)', message)
            move_match = re.search(r'Move:\s*(.+)', message)
            error_match = re.search(r'Error:\s*(.+)', message)
                
            current_round_num = None
            
            if round_match:
                # Round indicator: "Round: 0 ..."
                current_round_num = int(round_match.group(1))
                st.session_state.current_round = current_round_num + 1  # Display as 1-indexed
                if round_display:
                    round_display.metric("Current Progress", f"Round {current_round_num + 1}")
                if status_text:
                    status_text.text(f"🚀 AutoG-S agent: Round {current_round_num + 1}")
                if progress_bar:
                    progress_bar.progress(min(8 + ((current_round_num + 1) * 10), 90))
                
                # Add round-specific log
                add_log_entry(f"=== Starting Round {current_round_num + 1} ===", current_round_num + 1)
                add_log_entry(message, current_round_num + 1)
                    
            elif move_match:
                # Move within a round: "Move: {...}"
                move_data = move_match.group(1)
                current_round_for_move = st.session_state.get('current_round', 1)
                add_log_entry(f"Executing move: {move_data}", current_round_for_move)
                if status_text:
                    status_text.text(f"🔄 Executing move in Round {current_round_for_move}")
                    
            elif error_match:
                # Error message: "Error: ..."
                error_msg = error_match.group(1)
                current_round_for_error = st.session_state.get('current_round', 1)
                add_log_entry(f"❌ ERROR: {error_msg}", current_round_for_error)
                if status_text:
                    status_text.text(f"⚠️ Error in Round {current_round_for_error}")
                
            elif "No more action can be taken" in message:
                # Completion message
                add_log_entry("✅ AutoG-S completed successfully - no more actions needed")
                if status_text:
                    status_text.text("✅ AutoG-S processing completed")
                
            elif "Too many errors" in message:
                # Error termination
                add_log_entry("❌ AutoG-S stopped due to too many errors")
                if status_text:
                    status_text.text("❌ AutoG-S stopped due to errors")
                
            else:
                # General message
                if status_text:
                    status_text.text(f"🚀 {message}")
                add_log_entry(message)
            
            return True  # Continue processing
            
        # Run the actual AutoG-S processing with progress callback
        add_log_entry("Executing AutoG-S processing pipeline...")
        
        # Final method validation before service call
        if config.method is None or config.method == "None" or str(config.method).strip() == "":
            add_log_entry(f"CRITICAL: Method is still None right before service call! Forcing to 'autog-s'")
            config.method = "autog-s"
        
        # Debug: Log all parameters being passed
        add_log_entry(f"DEBUG: Calling service.run_autog with:")
        add_log_entry(f"  - llm_name: {config.llm_model}")
        add_log_entry(f"  - method: '{config.method}'")
        # Show appropriate task information in logs
        if config.use_custom_task:
            add_log_entry(f"  - task_name: Custom Task")
            if config.custom_task and config.custom_task.strip():
                add_log_entry(f"  - custom_description: {config.custom_task}")
            else:
                add_log_entry(f"  - custom_description: Not specified")
        else:
            add_log_entry(f"  - task_name: {config.dataset_name}:{config.task}")
        add_log_entry(f"  - max_rounds: {config.max_rounds}")
        
        # Ensure method parameter is definitely not None
        method_param = config.method if config.method is not None else "autog-s"
        add_log_entry(f"DEBUG: Final method parameter: '{method_param}'")
        
        # Only pass custom task description if it's actually provided
        custom_desc = None
        if config.use_custom_task and config.custom_task and config.custom_task.strip():
            custom_desc = config.custom_task
        
        agent_history, analysis_result, output_path, generated_files = service.run_autogs(
            dataframes=st.session_state.dataframes,
            llm_name=config.llm_model,
            method=method_param,  # Use validated method parameter
            task_name=config.task,
            dataset_name=config.dataset_name,
            seed=config.seed,
            max_rounds=config.max_rounds,
            progress_callback=progress_callback,
            custom_task_description=custom_desc
        )
        
        # Parse agent history for detailed round information
        actual_rounds = st.session_state.get('current_round', 0)
        hit_max_rounds = False
        
        if agent_history:
            # Parse the agent history to extract additional information
            lines = agent_history.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Add agent history to logs (these are internal agent decisions)
                add_log_entry(f"Agent decision: {line}")
            
            # Check if hit max rounds
            if any(phrase in agent_history.lower() for phrase in ['maximum', 'max', 'limit', 'threshold']):
                hit_max_rounds = True
                add_log_entry(f"AutoG-S reached maximum rounds limit ({config.max_rounds})")
        
        # Use the current round from real-time tracking
        st.session_state.rounds_completed = actual_rounds-1 if actual_rounds > 0 else 1
        add_log_entry(f"Processing completed with {st.session_state.rounds_completed} rounds")
        
        # Update progress indicators to show completion
        if status_indicator:
            status_indicator.success("✅ Complete")
        if status_text:
            status_text.text("✅ Processing complete!")
        if round_display:
            round_display.metric("Rounds Completed", st.session_state.rounds_completed)
        if progress_bar:
            progress_bar.progress(100)  # Set progress bar to 100%
        
        add_log_entry("AutoG-S processing completed successfully!")
        add_log_entry(f"Generated {len(generated_files)} output files")
        
        # Store results in session state
        st.session_state.results = {
            'agent_history': agent_history,
            'analysis_result': analysis_result,
            'output_path': output_path,
            'generated_files': generated_files,
            'cache_info': service.get_cache_info(),
            'config': config,
            'rounds_completed': st.session_state.rounds_completed,
            'hit_max_rounds': hit_max_rounds
        }
        
        add_log_entry(f"DEBUG: Results stored in session state with {len(generated_files)} files")
        
        # Mark task as completed
        st.session_state.task_running = False
        st.session_state.processing_started = False
        # Force rerun to re-enable sidebar
        st.rerun()
            
    except Exception as e:
        if status_indicator:
            status_indicator.error("❌ Failed")
        error_msg = f"AutoG-S processing failed: {str(e)}"
        
        # Store error in session state so it persists across reruns
        st.session_state.processing_error = {
            'message': error_msg,
            'exception': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'config': {
                'llm_model': config.llm_model,
                'method': config.method,
                'task': config.task,
                'dataset': config.dataset_name,
                'seed': config.seed,
                'max_rounds': config.max_rounds,
            }
        }
        
        st.error(f"❌ {error_msg}")
        add_log_entry(f"ERROR: {error_msg}")
        
        # Write detailed error to log file
        import datetime
        error_log_path = "autogs_error.log"
        try:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"AutoG-S Error Log - {datetime.datetime.now()}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Error Message: {str(e)}\n")
                f.write(f"Error Type: {type(e).__name__}\n")
                f.write(f"\nFull Traceback:\n")
                f.write(traceback.format_exc())
                f.write(f"\nConfiguration:\n")
                f.write(f"- LLM Model: {config.llm_model}\n")
                f.write(f"- Method: {config.method}\n")
                f.write(f"- Task: {config.task}\n")
                f.write(f"- Dataset: {config.dataset_name}\n")
                f.write(f"- Seed: {config.seed}\n")
                f.write(f"- Max Rounds: {config.max_rounds}\n")
                f.write(f"- Use Custom Task: {config.use_custom_task}\n")
                if config.use_custom_task:
                    f.write(f"- Custom Task: {config.custom_task}\n")
                f.write(f"\nDataframes:\n")
                for name, df in st.session_state.dataframes.items():
                    f.write(f"- {name}: {df.shape} ({df.memory_usage(deep=True).sum() / 1024**2:.1f} MB)\n")
                f.write(f"\nPython Path:\n")
                import sys
                for i, path in enumerate(sys.path[:10]):
                    f.write(f"  {i}: {path}\n")
                f.write(f"\n{'='*80}\n")
            
            st.info(f"📝 Detailed error log written to: `{error_log_path}`")
        except Exception as log_error:
            st.warning(f"Could not write error log: {log_error}")
        
        # Logs are shown in the main processing section
        
        # Show detailed error information
        with st.expander("🔍 Error Details", expanded=True):
            st.code(traceback.format_exc())
            
            # Provide troubleshooting tips
            st.markdown("**Troubleshooting Tips:**")
            st.markdown("- Ensure you're running in the `autog-cpu` conda environment")
            st.markdown("- Check that AWS credentials are valid and have Bedrock access")
            st.markdown("- Verify that all AutoG-S dependencies are installed")
            st.markdown("- Try with a simpler dataset or different task")
            
    finally:
        st.session_state.task_running = False
        st.session_state.processing_started = False
        # Force rerun to re-enable sidebar
        st.rerun()

def render_results():
    """Render results section"""
    if not st.session_state.get('results'):
        return

    st.markdown("---")
        
    st.markdown('<div class="section-header">📈 Results</div>', unsafe_allow_html=True)
    
    # Add action buttons at the top of results
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Processing completed successfully!** 🎉")
    with col2:
        if st.button("🗑️ Clear Results", help="Clear results and logs, keep data", key="clear_results_btn", type="secondary"):
            # Clear results and logs but preserve data and configuration
            if 'results' in st.session_state:
                del st.session_state['results']
            
            # Clear processing states and logs (but preserve rounds_completed to prevent data clearing)
            keys_to_clear = ['current_round', 'task_running', 'stop_requested', 'processing_started', 'last_error', 'processing_error', 'error_traceback']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Set a flag to indicate results were manually cleared (to preserve data)
            st.session_state.results_manually_cleared = True
            
            # Clear log data
            st.session_state.processing_logs = []
            st.session_state.round_logs = {}
            
            # Show success message before rerun
            st.success("✅ Results cleared successfully! Your uploaded data is preserved.")
            
            # Force UI refresh to hide results section
            st.rerun()
    
    results = st.session_state.get('results')
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Schema Diagram", "📊 Analysis", "🤖 Agent History", "📁 Downloads"])
    
    with tab1:
        st.markdown("### Generated Schema Diagram")
        
        generated_files = results.get('generated_files', {})
        if 'schema_png' in generated_files:
            schema_path = generated_files['schema_png']
            if os.path.exists(schema_path) and os.path.getsize(schema_path) > 0:
                st.image(schema_path, caption="Database Schema Diagram", width='stretch')
                st.success("✅ Schema diagram generated successfully!")
                
                file_size = os.path.getsize(schema_path)
                st.caption(f"📊 Image size: {file_size:,} bytes")
            else:
                st.error("❌ Schema image file is empty or corrupted")
        else:
            st.info("🔄 Schema diagram not available")
    
    with tab2:
        st.markdown("### Data Analysis Results")
        st.text_area(
            "Analysis Output",
            value=results['analysis_result'],
            height=400,
            help="Automated analysis of your data"
        )
    
    with tab3:
        st.markdown("### AutoG-S Agent Processing History")
        st.text_area(
            "Agent History",
            value=results['agent_history'],
            height=400,
            help="Step-by-step processing history"
        )
    
    with tab4:
        render_downloads(results)

def render_downloads(results):
    """Render download section"""
    st.markdown("### Download Generated Files")
    
    generated_files = results.get('generated_files', {})
    service = st.session_state.autog_service
    
    if generated_files:
        st.markdown("📁 **Available Files:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text files
            for file_type, file_path in generated_files.items():
                if file_type in ['agent_history', 'information', 'metadata', 'final_metadata']:
                    try:
                        content = service.get_file_content(file_path)
                        file_name = f"{file_type}.{'yaml' if 'metadata' in file_type else 'txt'}"
                        mime_type = "application/x-yaml" if 'metadata' in file_type else "text/plain"
                        
                        st.download_button(
                            label=f"📄 Download {file_type.replace('_', ' ').title()}",
                            data=content,
                            file_name=file_name,
                            mime=mime_type,
                            help=f"Download {file_type.replace('_', ' ')}"
                        )
                    except Exception as e:
                        st.error(f"Error preparing {file_type}: {str(e)}")
        
        with col2:
            # Binary files
            for file_type, file_path in generated_files.items():
                if file_type in ['schema_png', 'schema_pdf']:
                    try:
                        content = service.get_file_content(file_path)
                        file_ext = file_type.split('_')[1]
                        mime_type = f"image/{file_ext}" if file_ext == 'png' else f"application/{file_ext}"
                        
                        st.download_button(
                            label=f"🖼️ Download Schema ({file_ext.upper()})",
                            data=content,
                            file_name=f"schema.{file_ext}",
                            mime=mime_type,
                            help=f"Download schema diagram as {file_ext.upper()}"
                        )
                    except Exception as e:
                        st.error(f"Error preparing {file_type}: {str(e)}")
        
        # File summary
        st.markdown("---")
        st.markdown("📊 **File Summary:**")
        for file_type, file_path in generated_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                size_str = f"{file_size:,} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
                st.markdown(f"- **{file_type.replace('_', ' ').title()}**: {size_str}")
    else:
        st.info("📁 No files generated yet")
    
    # Cache info
    cache_info = results.get('cache_info', {})
    if cache_info:
        st.markdown("---")
        st.markdown("**💾 Cache Information:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Strategy:** {cache_info.get('strategy', 'hybrid')}")
            st.markdown(f"**Memory Files:** {cache_info.get('memory_files', 0)}")
        with col2:
            st.markdown(f"**Memory Usage:** {cache_info.get('memory_size_mb', 0):.1f} MB")
            if cache_info.get('cached_files'):
                st.markdown(f"**Cached:** {', '.join(cache_info['cached_files'])}")

def main():
    """Main application"""
    # Initialize session state
    SessionState.init()
    
    # Header
    st.markdown('<div class="main-header">🔗 AutoG-S: Automatic Table-to-Graph Generation</div>', unsafe_allow_html=True)
    st.markdown("Transform your tabular data into graph representations using advanced LLMs.")
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Main layout: Left for AutoG-S processing and results, Right for data management
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # AutoG-S Processing section
        render_processing_section(config)
        
        # Results section
        render_results()

        # Round execution details shown below results after processing
        render_move_based_logs()
    
    with col2:
        # Data upload section with integrated preview
        has_files = render_file_upload()
        
        st.markdown("---")
        
        # Combined Status & Session Info
        st.markdown("### 📊 Status & Session Info")
        
        # Status indicators (moved to top)
        st.markdown("**Requirements Status:**")
        
        # Check requirements (refresh state every time)
        current_dataframes_status = st.session_state.get('dataframes', {})
        has_data = bool(current_dataframes_status) and len(current_dataframes_status) > 0
        has_credentials = st.session_state.get('aws_credentials_valid', False)
        not_running = not st.session_state.get('task_running', False)
        has_results = bool(st.session_state.get('results', None))
        
        # Status indicators in a compact format
        if has_data:
            st.markdown("✅ **Data:** Uploaded")
        else:
            st.markdown("❌ **Data:** Not uploaded")
        
        if has_credentials:
            st.markdown("✅ **AWS:** Credentials valid")
        else:
            st.markdown("❌ **AWS:** Credentials required")
        
        if not_running:
            if has_results:
                st.markdown("✅ **Status:** Processing complete")
            elif has_data and has_credentials:
                st.markdown("✅ **Status:** Ready to run")
            else:
                st.markdown("⚠️ **Status:** Not ready")
        else:
            st.markdown("🔄 **Status:** Processing...")
        
        # Overall readiness
        ready_to_run = has_data and has_credentials and not_running
        
        if ready_to_run:
            if has_results:
                st.success("✅ Processing completed successfully. You can run again with different settings.")
            else:
                st.success("🎉 All requirements met! Ready to start AutoG-S processing.")
        else:
            missing_items = []
            if not has_data:
                missing_items.append("upload data")
            if not has_credentials:
                missing_items.append("configure AWS credentials")
            if not not_running:
                missing_items.append("wait for current processing to complete")
            
            if missing_items:
                st.warning(f"⚠️ Please {' and '.join(missing_items)} before running AutoG-S.")
        
        st.markdown("---")
        
        # Session details (moved to bottom)
        st.markdown("**Session Details:**")
        st.markdown(f"- **ID:** `{st.session_state.session_id}`")
        st.markdown(f"- **Tables:** {len(st.session_state.get('dataframes', {}))}")
        if st.session_state.get('results'):
            st.markdown(f"- **Output:** `{st.session_state.get('results')['output_path']}`")

if __name__ == "__main__":
    main()