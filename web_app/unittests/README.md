# AutoG2 Web App Unit Tests

This directory contains unit tests for the AutoG2 web application.

## Available Tests

### `test_data_loading.py`
Tests the data loading functionality including:
- Real ADSP data loading from `data/datasets/adsp/data/`
- Sample data generation (MAG, e-commerce, social network)
- Data availability checking
- Error handling for missing files

### `test_schema_generation.py`
Tests the schema generation functionality including:
- DBB dataset creation from DataFrames
- Schema plot generation (PNG and PDF)
- Graphviz integration
- File output validation

## Running Tests

### Run All Tests
```bash
python run_all_tests.py
```

### Run Individual Tests
```bash
python test_data_loading.py
python test_schema_generation.py
```

## Test Requirements

- Must be run from the AutoG root directory
- Requires `autog-cpu` conda environment
- Graphviz must be installed for schema generation tests
- Real data tests require `data/datasets/adsp/data/` to exist

## Expected Output

### Data Loading Test
- Checks availability of real datasets (ADSP, AVS, MAG)
- Tests 4 sample datasets with detailed statistics
- Reports table counts, row counts, and column counts

### Schema Generation Test
- Creates sample dataset
- Generates schema diagrams
- Reports file sizes and generation status
- Cleans up temporary files

## Troubleshooting

- **Import Errors**: Ensure you're in the correct conda environment
- **Path Errors**: Run from the AutoG root directory
- **Graphviz Errors**: Install with `sudo apt-get install graphviz`
- **Data Not Found**: Verify the data directory structure exists