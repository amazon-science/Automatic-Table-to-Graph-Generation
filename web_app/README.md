
# AutoG2 Web Demo

A Streamlit web application for running AutoG2 (Automatic Table-to-Graph Generation) with in-memory data processing.

## Features

- 📊 **Multiple Data Sources**: Upload CSV, Parquet, or NumPy files OR use built-in example datasets
- 🏪 **Real Example Data**: Includes real ADSP (Amazon Digital Services Platform) data with 288K+ rows
- 🔗 **Graph Generation**: Process tabular data into graph representations using AutoG2
- 🖼️ **Schema Visualization**: Automatic generation and display of database schema diagrams
- 📁 **File Downloads**: Download generated metadata, agent history, and schema diagrams
- 🤖 **Multiple LLM Models**: Support for Claude Sonnet 4, Sonnet 3, and Haiku
- 📈 **Real-time Analysis**: Live data preview, quality metrics, and processing status
- 💾 **In-memory Processing**: No file system dependencies for user data
- 🎯 **Easy Testing**: Built-in sample datasets for immediate experimentation

## Setup

### Prerequisites

1. **Conda Environment**: Make sure you have the `autog-cpu` environment set up:
```bash
# From the root directory
bash multi-table-benchmark/conda/create_conda_env.sh -c -p 3.9 -t 2.1
conda activate autog-cpu
```

2. **Install Dependencies**:
```bash
cd web_app
pip install -r requirements.txt
```

3. **System Dependencies**:
```bash
# For schema diagram generation
sudo apt-get install graphviz
```

4. **AWS Credentials**: You'll need AWS Bedrock access for LLM models. You can either:
   - Set environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
   - Or enter them in the web app interface

### Running the Web App

#### Option 1: Using the startup script (Recommended)
```bash
cd web_app
python run_webapp.py
```

#### Option 2: Direct Streamlit command
```bash
cd web_app
streamlit run app.py
```

The web app will be available at: http://localhost:8501

## Quick Start

### Option A: Use Example Data (Fastest)
1. Open the web app: `python web_app/run_webapp.py`
2. In the sidebar, check "Use Sample Data"
3. Select "🏪 ADSP Real Data" (288K+ rows of real Amazon data)
4. Click "Load Sample Data"
5. Enter your AWS credentials
6. Click "🔄 Run AutoG2"

### Option B: Upload Your Own Data
1. Open the web app
2. Upload your CSV/Parquet/NumPy files
3. Configure LLM model and task
4. Enter AWS credentials
5. Run AutoG2 processing

## Available Example Datasets

- **🏪 ADSP Real Data**: Real Amazon Digital Services Platform data (288K+ rows)
  - Product nodes, hierarchical paths, purchase transactions
- **📚 Academic Papers**: Sample MAG-style academic data
- **🛒 E-commerce Data**: Sample user-product-transaction data  
- **📱 Social Network**: Sample user-post-connection data

## Usage Steps

1. **Choose Data Source**: 
   - Use sample data (checkbox in sidebar) OR upload your own files
2. **Configure Processing**:
   - Select LLM model (Sonnet 4 recommended)
   - Choose AutoG method (autog-s for faster processing)
   - Select dataset type and task
3. **Set Credentials**: Enter AWS Bedrock credentials
4. **Process**: Click "Run AutoG2" and wait for completion (5-15 minutes)
5. **View Results**: 
   - **Schema Diagram**: Interactive visualization of your data relationships
   - **Data Analysis**: Detailed analysis of your dataset structure
   - **Agent History**: Step-by-step processing log
   - **Downloads**: Get metadata.yaml, agent_history.txt, and schema files

## Supported File Formats

- **CSV**: Standard comma-separated values
- **Parquet**: Apache Parquet format
- **NumPy**: `.npy` and `.npz` files

## Configuration Options

- **LLM Models**: Claude Sonnet 4, Sonnet 3, Haiku
- **Methods**: autog-s, autog-m, autog-l
- **Tasks**: Various predefined tasks for different dataset types

## Architecture

The web app uses:
- **Streamlit**: Web interface
- **InMemoryAutoGService**: Custom service for processing DataFrames
- **Temporary Workspaces**: Creates temporary directories for AutoG2 processing
- **AWS Bedrock**: LLM inference

### Project Structure
```
web_app/
├── app.py                    # Main Streamlit application
├── services/
│   ├── autog_service.py      # Core AutoG2 integration service
│   └── __init__.py
├── utils/
│   ├── config.py             # LLM and AutoG configurations
│   ├── test_data.py          # Real + sample data loaders
│   └── __init__.py
├── unittests/
│   ├── test_data_loading.py  # Data loading functionality tests
│   ├── test_schema_generation.py # Schema generation tests
│   ├── run_all_tests.py      # Comprehensive test runner
│   ├── README.md             # Test documentation
│   └── __init__.py
├── setup_env.py              # Environment and PYTHONPATH setup
├── run_webapp.py             # Application startup script
├── requirements.txt          # Python dependencies
└── README.md                 # Main documentation
```

## Testing

### Run All Tests
```bash
python web_app/unittests/run_all_tests.py
```

### Individual Tests
```bash
# Test data loading functionality
python web_app/unittests/test_data_loading.py

# Test schema generation
python web_app/unittests/test_schema_generation.py

# Test caching functionality
python web_app/unittests/test_caching.py
```

These tests will verify that all sample datasets load correctly, show data statistics, test schema generation capabilities, and validate the intelligent caching system.

## Troubleshooting

1. **Import Errors**: Make sure you're in the `autog-cpu` conda environment
2. **AWS Errors**: Verify your AWS credentials and Bedrock access
3. **Memory Issues**: Large datasets may require more RAM
4. **Path Issues**: The startup script handles PYTHONPATH automatically
5. **Sample Data Not Loading**: Make sure you're running from the AutoG root directory
6. **File Upload Issues**: Check file format (CSV, Parquet, NPY, NPZ) and size limits

## Performance Tips

### Cache Strategy Selection
- **Hybrid (Default)**: Best balance of speed and memory usage
- **Memory**: Fastest access, use for small datasets (<100MB)
- **Disk**: Most memory-efficient, use for large datasets (>1GB)

### Processing Optimization
- **For Testing**: Use sample data or smaller datasets first
- **Model Selection**: Sonnet 4 for best quality, Haiku for faster processing
- **Method Selection**: autog-s for faster results, autog-l for more comprehensive analysis
- **Memory**: Close other applications when processing large datasets

### File Management
- **Small files** (text, metadata) are automatically cached in memory for instant downloads
- **Large files** (models, backups) remain on disk to preserve memory
- **Cache information** is displayed in the Downloads tab after processing

## Example Workflow

1. **Start**: `python web_app/run_webapp.py`
2. **Load ADSP Data**: Check "Use Sample Data" → Select ADSP → Load
3. **Configure**: Sonnet 4 + autog-s + custom:repeater
4. **Credentials**: Enter AWS keys in sidebar
5. **Process**: Click "Run AutoG2" (takes 5-15 minutes)
6. **Results**: 
   - View the generated **schema diagram** showing table relationships
   - Download **metadata.yaml** with the processed data structure
   - Download **agent_history.txt** with complete processing log
   - Download **schema.png/pdf** for presentations or documentation

## Generated Files

After processing, you can download:
- **📄 agent_history.txt**: Complete step-by-step processing log
- **📋 metadata.yaml**: Dataset structure and configuration
- **📊 information.txt**: Detailed data analysis report  
- **🖼️ schema.png**: Database schema diagram (PNG format)
- **📄 schema.pdf**: Database schema diagram (PDF format)