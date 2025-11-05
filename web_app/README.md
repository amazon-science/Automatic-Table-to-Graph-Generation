
# AutoG Web Demo

A Streamlit web application for running AutoG (Automatic Table-to-Graph Generation) with in-memory data processing.

📄 **Paper**: [AutoG: Towards automatic graph construction from tabular data](https://arxiv.org/abs/2501.15282)

## Setup

### Prerequisites

1. **Conda Environment**: Make sure you have the `autog-cpu` environment set up:
```bash
# From the root directory (Automatic-Table-to-Graph-Generation)
bash multi-table-benchmark/conda/create_conda_env.sh -c -p 3.9 -t 2.1
conda activate autog-cpu
export PYTHONPATH=$(pwd)/multi-table-benchmark
```

**Note**: The PYTHONPATH export is required for the backend imports to work correctly.

2. **AWS Credentials**: You'll need AWS Bedrock access for LLM models:
   - Set environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
   - **Detailed setup guide**: See [AWS_SETUP.md](AWS_SETUP.md) for comprehensive instructions

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```


### Running the Web App

#### Option 1: Using the startup script (Recommended)
```bash
cd web_app
python run_autogs_webapp.py
```

#### Option 2: Direct Streamlit command
If you prefer to run Streamlit directly, you'll need to set the PYTHONPATH for all required modulesst:
```bash
# From the root directory (Automatic-Table-to-Graph-Generation)
export PYTHONPATH=$(pwd):$(pwd)/multi-table-benchmark:$(pwd)/dbinfer:$(pwd)/models:$(pwd)/prompts:$(pwd)/web_app
cd web_app
streamlit run AutoGS_WebApp.py
```

## Usage Steps

1. **Choose Data Source**: 
   - **Upload your own files** - Supports multiple file formats:
     - **CSV Files**: `.csv` (comma-separated values)
     - **TSV Files**: `.tsv`, `.tab` (tab-separated values)
     - **Text Files**: `.txt`, `.dat` (auto-detects delimiter, fallback to comma)
     - **Parquet Files**: `.parquet`, `.pq`, `.pqt` (Apache Parquet format)
     - **NumPy Files**: `.npy`, `.npz` (NumPy array format)
   - **Multiple files supported**: Upload multiple tables at once
   - **Automatic deduplication**: Duplicate filenames are automatically handled
   - **File size**: Large files are supported, memory usage is displayed for each dataset
2. **Configure Processing**:
   - Select LLM model (Sonnet 4 recommended)
   - Choose AutoG method (autog-s)
   - Select dataset type and task, or use custom task description
3. **Process**: Click "Run AutoG" button and wait for completion
4. **View Results**: 
   - **Schema Diagram**: Visualization of your data relationships
   - **Data Analysis**: Detailed analysis of your dataset
   - **Agent History**: All actions taken during processing
   - **Downloads**: Get metadata.yaml, agent_history.txt, and schema files
5. **Check Run Execution Details**:
   - Check each move/action taken in each round


## Troubleshooting

1. **Import Errors**: Make sure you're in the `autog-cpu` conda environment and PYTHONPATH is set correctly
   - **Required**: From root directory, run `export PYTHONPATH=$(pwd)/multi-table-benchmark`
2. **AWS Errors**: Verify your AWS credentials and Bedrock access
3. **Memory Issues**: Large datasets may require more RAM
4. **Path Issues**: Ensure you're running from the correct directory (web_app/)
5. **Sample Data Not Loading**: Make sure you're running from the AutoG root directory
6. **File Upload Issues**: Ensure your files are in supported formats:
   - **Supported**: CSV (.csv), TSV (.tsv, .tab), Text (.txt, .dat), Parquet (.parquet, .pq, .pqt), NumPy (.npy, .npz)


## Generated Files

After processing, you can download:
- **📄 agent_history.txt**: Complete step-by-step processing log
- **📋 metadata.yaml**: Dataset structure and configuration
- **📊 information.txt**: Detailed data analysis report  
- **🖼️ schema.png**: Database schema diagram (PNG format)
- **📄 schema.pdf**: Database schema diagram (PDF format)