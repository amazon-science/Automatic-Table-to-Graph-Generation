# Setup environment and paths first
import sys
import os
from pathlib import Path

# Get directories
current_file = Path(__file__).resolve()
web_app_dir = current_file.parent
root_dir = web_app_dir.parent

# Add paths in correct order
paths_to_add = [
    str(web_app_dir),  # web_app directory first for local imports
    str(root_dir),     # root directory for backend imports
    str(root_dir / "multi-table-benchmark"),
    str(root_dir / "dbinfer"),
    str(root_dir / "models"),
    str(root_dir / "prompts"),
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"✅ Added paths to sys.path")
print(f"📁 Web app dir: {web_app_dir}")
print(f"📁 Root dir: {root_dir}")

import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback
from dotenv import load_dotenv

# Import web_app specific modules
try:
    from utils.config import LLM_MODELS, AUTOG_CONFIG
    print("✅ Imported utils.config")
except ImportError as e:
    print(f"❌ Failed to import utils.config: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for: {web_app_dir / 'utils' / 'config.py'}")
    print(f"File exists: {(web_app_dir / 'utils' / 'config.py').exists()}")
    raise

try:
    from utils.test_data import SAMPLE_DATASETS
    print("✅ Imported utils.test_data")
except ImportError as e:
    print(f"❌ Failed to import utils.test_data: {e}")
    raise

try:
    from services.autog_service import InMemoryAutoGService
    print("✅ Imported services.autog_service")
except ImportError as e:
    print(f"❌ Failed to import services.autog_service: {e}")
    raise

# Load environment variables
load_dotenv()

# Set page title and layout
st.set_page_config(
    page_title="AutoG2 Web Demo",
    layout="wide",
    page_icon="🔗"
)

# Initialize session state
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'autog_service' not in st.session_state:
    st.session_state.autog_service = InMemoryAutoGService()
if 'results' not in st.session_state:
    st.session_state.results = None

# Load model configuration
model_config = LLM_MODELS

# Helper function to load file into DataFrame
def load_file_to_dataframe(uploaded_file):
    """Load uploaded file into pandas DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        elif uploaded_file.name.endswith(('.npy', '.npz')):
            # Handle numpy files
            bytes_data = uploaded_file.read()
            arr = np.load(io.BytesIO(bytes_data), allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):
                data = {}
                for k in arr.files:
                    if k == 'feat':
                        data[k] = list(arr[k])
                    else:
                        data[k] = arr[k]
                return pd.DataFrame(data)
            else:
                return pd.DataFrame(arr)
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.title("🔗 AutoG2 Web Demo")
    st.markdown("---")
    
    # AWS Credentials Section
    st.subheader("AWS Credentials")
    aws_access_key = st.text_input("AWS Access Key ID", type="password")
    aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
    aws_session_token = st.text_input("AWS Session Token", type="password")
    
    if aws_access_key and aws_secret_key:
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        if aws_session_token:
            os.environ['AWS_SESSION_TOKEN'] = aws_session_token
        st.success("✅ AWS credentials set")
    else:
        st.warning("⚠️ Please provide AWS credentials")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("LLM Model Selection")
    model_select = st.selectbox(
        "Choose an LLM Model",
        list(model_config.keys()),
        help="Select the language model for AutoG2 processing"
    )
    selected_model_config = model_config[model_select]
    st.info(selected_model_config["description"])
    
    st.markdown("---")
    
    # AutoG Configuration
    st.subheader("AutoG Configuration")
    method = st.selectbox(
        "Method",
        AUTOG_CONFIG["methods"],
        help="AutoG processing method"
    )
    
    dataset_type = st.selectbox(
        "Dataset Type",
        list(AUTOG_CONFIG["datasets"].keys()),
        help="Type of dataset for task selection"
    )
    
    task = st.selectbox(
        "Task",
        AUTOG_CONFIG["datasets"][dataset_type],
        help="Specific task within the dataset type"
    )
    
    task_name = f"{dataset_type}:{task}"
    
    # Cache Strategy
    st.subheader("Performance Settings")
    cache_strategy = st.selectbox(
        "Cache Strategy",
        ["hybrid", "memory", "disk"],
        index=0,
        help="How to handle intermediate files: hybrid (recommended), memory (faster), disk (lower memory)"
    )
    
    # Update service cache strategy
    if st.session_state.autog_service.cache_strategy != cache_strategy:
        st.session_state.autog_service.cache_strategy = cache_strategy
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader("Data Upload")
    
    # Sample data option
    use_sample_data = st.checkbox("Use Sample Data", help="Load pre-generated sample datasets for testing")
    
    if use_sample_data:
        sample_dataset = st.selectbox(
            "Choose Sample Dataset",
            list(SAMPLE_DATASETS.keys()),
            format_func=lambda x: SAMPLE_DATASETS[x]['name'],
            help="Select a sample dataset to test AutoG2 functionality"
        )
        
        # Show dataset description
        st.info(SAMPLE_DATASETS[sample_dataset]['description'])
        
        if st.button("Load Sample Data"):
            try:
                with st.spinner("Loading sample data..."):
                    sample_data = SAMPLE_DATASETS[sample_dataset]['generator']()
                    st.session_state.dataframes = sample_data
                    
                    # Show success message with data info
                    total_rows = sum(df.shape[0] for df in sample_data.values())
                    st.success(f"✅ Loaded {SAMPLE_DATASETS[sample_dataset]['name']}")
                    st.info(f"📊 {len(sample_data)} tables, {total_rows} total rows")
                    
            except Exception as e:
                st.error(f"❌ Failed to load sample data: {str(e)}")
                if "not found" in str(e).lower():
                    st.warning("💡 Make sure you're running from the correct directory with access to the data files.")
    else:
        uploaded_files = st.file_uploader(
            "Upload Data Files",
            type=["csv", "parquet", "npy", "npz"],
            accept_multiple_files=True,
            help="Upload CSV, Parquet, or NumPy files containing your tabular data"
        )

# Main content area
st.title("🔗 AutoG2: Automatic Table-to-Graph Generation")
st.markdown("Upload your tabular data and generate graph representations automatically using advanced LLMs.")

# File processing section
if uploaded_files and not use_sample_data:
    st.subheader("📊 Uploaded Data")
    
    # Process uploaded files
    new_dataframes = {}
    for uploaded_file in uploaded_files:
        df = load_file_to_dataframe(uploaded_file)
        if df is not None:
            table_name = os.path.splitext(uploaded_file.name)[0]
            new_dataframes[table_name] = df
    
    # Update session state
    st.session_state.dataframes = new_dataframes
    


# Data preview section
if st.session_state.dataframes:
    st.subheader("📊 Current Data")
    
    # Show data summary
    total_rows = sum(df.shape[0] for df in st.session_state.dataframes.values())
    total_cols = sum(df.shape[1] for df in st.session_state.dataframes.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tables", len(st.session_state.dataframes))
    with col2:
        st.metric("Total Rows", f"{total_rows:,}")
    with col3:
        st.metric("Total Columns", f"{total_cols:,}")
    
    # Display data preview
    for table_name, df in st.session_state.dataframes.items():
        with st.expander(f"📋 {table_name} ({df.shape[0]:,} rows, {df.shape[1]} columns)"):
            # Show first few rows
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Types:**")
                dtype_info = df.dtypes.value_counts().to_dict()
                for dtype, count in dtype_info.items():
                    st.write(f"- {dtype}: {count} columns")
            
            with col2:
                st.write("**Data Quality:**")
                missing_count = df.isnull().sum().sum()
                st.write(f"- Missing values: {missing_count:,}")
                st.write(f"- Completeness: {((df.size - missing_count) / df.size * 100):.1f}%")
                
                # Show columns with missing values
                missing_cols = df.isnull().sum()
                missing_cols = missing_cols[missing_cols > 0]
                if len(missing_cols) > 0:
                    st.write("**Columns with missing values:**")
                    for col, count in missing_cols.items():
                        st.write(f"- {col}: {count}")

# Processing section
if st.session_state.dataframes:
    st.markdown("---")
    st.subheader("🚀 Run AutoG2 Processing")
    
    # Configuration summary
    config_col, button_col = st.columns([3, 1])
    
    with config_col:
        st.write("**Current Configuration:**")
        st.write(f"- **Model:** {selected_model_config['name']}")
        st.write(f"- **Method:** {method}")
        st.write(f"- **Task:** {task_name}")
        st.write(f"- **Tables:** {', '.join(st.session_state.dataframes.keys())}")
    
    with button_col:
        # Clear data button
        if st.button("🗑️ Clear Data", help="Clear all loaded data"):
            st.session_state.dataframes = {}
            st.session_state.results = None
            st.rerun()
        
        # Run AutoG2 button
        run_button = st.button(
            "🔄 Run AutoG2",
            type="primary",
            disabled=not (aws_access_key and aws_secret_key),
            help="Process the uploaded data with AutoG2"
        )
    
    # Run AutoG2 processing
    if run_button:
        if not st.session_state.dataframes:
            st.error("Please upload data files first!")
        else:
            with st.spinner("🔄 Running AutoG2 processing... This may take several minutes."):
                try:
                    # Run AutoG2
                    agent_history, analysis_result, output_path, generated_files = st.session_state.autog_service.run_autog2(
                        dataframes=st.session_state.dataframes,
                        llm_name=model_select,
                        method=method,
                        task_name=task_name,
                        dataset_name="webapp_dataset"
                    )
                    
                    # Get cache information
                    cache_info = st.session_state.autog_service.get_cache_info()
                    
                    # Store results
                    st.session_state.results = {
                        'agent_history': agent_history,
                        'analysis_result': analysis_result,
                        'output_path': output_path,
                        'generated_files': generated_files,
                        'cache_info': cache_info
                    }
                    
                    st.success("✅ AutoG2 processing completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ AutoG2 processing failed: {str(e)}")
                    st.error("**Error Details:**")
                    st.code(traceback.format_exc())

# Results section
if st.session_state.results:
    st.markdown("---")
    st.subheader("📈 Results")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Schema Diagram", "📊 Data Analysis", "🤖 Agent History", "📁 Downloads"])
    
    with tab1:
        st.subheader("Generated Schema Diagram")
        
        # Display schema image if available
        generated_files = st.session_state.results.get('generated_files', {})
        if 'schema_png' in generated_files:
            try:
                # Check if file exists and is readable
                schema_path = generated_files['schema_png']
                if os.path.exists(schema_path) and os.path.getsize(schema_path) > 0:
                    st.image(schema_path, caption="Database Schema Diagram", use_container_width=True)
                    st.success("✅ Schema diagram generated successfully!")
                    
                    # Show image info
                    file_size = os.path.getsize(schema_path)
                    st.caption(f"📊 Image size: {file_size:,} bytes")
                else:
                    st.error("❌ Schema image file is empty or corrupted")
            except Exception as e:
                st.error(f"❌ Error displaying schema image: {str(e)}")
                st.caption("💡 The schema file may be corrupted or in an unsupported format")
        else:
            st.info("🔄 Schema diagram will appear here after processing completes.")
            st.caption("The schema visualization shows the relationships between your data tables")
            
        # Show schema generation status
        if generated_files:
            if 'schema_png' in generated_files:
                st.write("📊 **Schema Files Generated:**")
                if 'schema_png' in generated_files:
                    st.write("- ✅ PNG format (displayed above)")
                if 'schema_pdf' in generated_files:
                    st.write("- ✅ PDF format (available for download)")
            else:
                st.warning("⚠️ Schema diagram not found. This may indicate an issue during processing.")
    
    with tab2:
        st.subheader("Data Analysis Results")
        st.text_area(
            "Analysis Output",
            value=st.session_state.results['analysis_result'],
            height=400,
            help="Automated analysis of your uploaded data"
        )
    
    with tab3:
        st.subheader("AutoG Agent Processing History")
        st.text_area(
            "Agent History",
            value=st.session_state.results['agent_history'],
            height=400,
            help="Step-by-step processing history from the AutoG agent"
        )
    
    with tab4:
        st.subheader("Download Generated Files")
        
        generated_files = st.session_state.results.get('generated_files', {})
        
        if generated_files:
            st.write("📁 **Available Files:**")
            
            # Create download buttons for each file
            col1, col2 = st.columns(2)
            
            with col1:
                # Agent History
                if 'agent_history' in generated_files:
                    try:
                        file_content = st.session_state.autog_service.get_file_content(generated_files['agent_history'])
                        st.download_button(
                            label="📄 Download Agent History",
                            data=file_content,
                            file_name="agent_history.txt",
                            mime="text/plain",
                            help="Download the complete agent processing history"
                        )
                    except Exception as e:
                        st.error(f"Error preparing agent history: {str(e)}")
                
                # Metadata
                if 'metadata' in generated_files:
                    try:
                        file_content = st.session_state.autog_service.get_file_content(generated_files['metadata'])
                        st.download_button(
                            label="📋 Download Metadata (YAML)",
                            data=file_content,
                            file_name="metadata.yaml",
                            mime="application/x-yaml",
                            help="Download the dataset metadata configuration"
                        )
                    except Exception as e:
                        st.error(f"Error preparing metadata: {str(e)}")
                
                # Information file
                if 'information' in generated_files:
                    try:
                        file_content = st.session_state.autog_service.get_file_content(generated_files['information'])
                        st.download_button(
                            label="📊 Download Data Analysis",
                            data=file_content,
                            file_name="information.txt",
                            mime="text/plain",
                            help="Download the detailed data analysis report"
                        )
                    except Exception as e:
                        st.error(f"Error preparing information file: {str(e)}")
            
            with col2:
                # Schema PNG
                if 'schema_png' in generated_files:
                    try:
                        file_content = st.session_state.autog_service.get_file_content(generated_files['schema_png'])
                        st.download_button(
                            label="🖼️ Download Schema (PNG)",
                            data=file_content,
                            file_name="schema.png",
                            mime="image/png",
                            help="Download the schema diagram as PNG image"
                        )
                    except Exception as e:
                        st.error(f"Error preparing schema PNG: {str(e)}")
                
                # Schema PDF
                if 'schema_pdf' in generated_files:
                    try:
                        file_content = st.session_state.autog_service.get_file_content(generated_files['schema_pdf'])
                        st.download_button(
                            label="📄 Download Schema (PDF)",
                            data=file_content,
                            file_name="schema.pdf",
                            mime="application/pdf",
                            help="Download the schema diagram as PDF document"
                        )
                    except Exception as e:
                        st.error(f"Error preparing schema PDF: {str(e)}")
                
                # Final metadata (if different from initial)
                if 'final_metadata' in generated_files:
                    try:
                        file_content = st.session_state.autog_service.get_file_content(generated_files['final_metadata'])
                        st.download_button(
                            label="📋 Download Final Metadata",
                            data=file_content,
                            file_name="final_metadata.yaml",
                            mime="application/x-yaml",
                            help="Download the final processed metadata"
                        )
                    except Exception as e:
                        st.error(f"Error preparing final metadata: {str(e)}")
            
            # File summary
            st.markdown("---")
            st.write("📊 **File Summary:**")
            for file_type, file_path in generated_files.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    size_str = f"{file_size:,} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
                    st.write(f"- **{file_type.replace('_', ' ').title()}**: {size_str}")
                else:
                    st.write(f"- **{file_type.replace('_', ' ').title()}**: ❌ File not found")
        else:
            st.info("📁 Generated files will appear here after processing completes.")
            
        # Cache and storage info
        cache_info = st.session_state.results.get('cache_info', {})
        output_path = st.session_state.results['output_path']
        
        st.markdown("---")
        st.write("**💾 Storage Information:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Output Directory:** `{output_path}`")
            st.write(f"**Cache Strategy:** {cache_info.get('strategy', 'hybrid')}")
        
        with col2:
            if cache_info:
                st.write(f"**Memory Cache:** {cache_info.get('memory_files', 0)} files ({cache_info.get('memory_size_mb', 0):.1f} MB)")
                if cache_info.get('cached_files'):
                    st.write(f"**Cached Files:** {', '.join(cache_info['cached_files'])}")
        
        st.caption("📝 Small files (text, metadata) are cached in memory for faster access. Large files (models, backups) remain on disk.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>AutoG2 Web Demo | Powered by Streamlit & AWS Bedrock</p>
    </div>
    """,
    unsafe_allow_html=True
)

