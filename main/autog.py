import os
import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import typer
from rich import traceback

from models.autog.agent_improved import AutoG_Agent, load_dbb_dataset_from_cfg_path_no_name
from prompts.task import get_task_description
from utils.misc import seed_everything

# Ensure NLTK punkt_tab is available
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    pass  # NLTK not required if not used


# ============================================================================
# Configuration and Types
# ============================================================================

class AutoGMethod(str, Enum):
    """Available AutoG execution methods."""
    SELECTIVE = "autog-s"  # no refinement
    AUGMENTED = "autog-a"  # AutoG-Augmented (with feedback)


@dataclass
class AutoGConfig:
    """Configuration for AutoG execution."""
    dataset: str
    schema_path: Path
    method: AutoGMethod
    task_name: str
    column_types_file: str = "type.txt"
    seed: int = 0
    lm_path: Optional[Path] = None
    use_cache: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        self.schema_path = Path(self.schema_path)
        if self.lm_path:
            self.lm_path = Path(self.lm_path)
        # Convert string to enum if needed
        if isinstance(self.method, str):
            self.method = AutoGMethod(self.method)


@dataclass
class AutoGResult:
    """Result of AutoG execution."""
    final_schema: Dict[str, Any]
    history: List[str]
    output_path: Path
    dataset: str
    task_name: str
    method: str


# ============================================================================
# Helper Functions
# ============================================================================

def retrieve_input_schema(full_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract input schema containing only dataset name and tables.

    Args:
        full_schema: Complete schema dictionary

    Returns:
        Filtered schema with only dataset_name and tables
    """
    return {
        key: value for key, value in full_schema.items()
        if key in ["dataset_name", "tables"]
    }


def _find_table_meta(meta_dict: Dict[str, Any], table_name: str) -> Optional[Dict[str, Any]]:
    """Find table metadata with case-insensitive lookup.

    Args:
        meta_dict: Metadata dictionary
        table_name: Table name to find

    Returns:
        Table metadata dict or None if not found
    """
    # Try exact match first
    if table_name in meta_dict:
        return meta_dict[table_name]

    # Try case-insensitive match
    table_name_lower = table_name.lower()
    for key in meta_dict:
        if key.lower() == table_name_lower:
            return meta_dict[key]

    return None


def is_column_numerical(column_value: np.ndarray) -> bool:
    """Check if a column contains numerical data.

    Args:
        column_value: Column data array

    Returns:
        True if column is numerical
    """
    if column_value.dtype in ['int64', 'float64']:
        return True
    if column_value.dtype == 'object':
        try:
            column_value.astype(int)
            return True
        except (ValueError, TypeError):
            return False
    return False


def infer_primary_key(column_value: np.ndarray) -> bool:
    """Infer if a column should be a primary key.

    Primary key criteria: numerical and all unique values.

    Args:
        column_value: Column data array

    Returns:
        True if column should be a primary key
    """
    return (is_column_numerical(column_value) and
            np.unique(column_value).size == column_value.size)


def build_column_schema(column_name: str, column_value: np.ndarray,
                       meta_dict: Dict[str, Any], table_name: str) -> Optional[Dict[str, Any]]:
    """Build schema for a single column.

    Args:
        column_name: Name of the column
        column_value: Column data array
        meta_dict: Metadata dictionary for all tables
        table_name: Name of the parent table

    Returns:
        Column schema dictionary or None if column should be skipped
    """
    # Find table in meta_dict (case-insensitive)
    table_meta = _find_table_meta(meta_dict, table_name)
    if table_meta is None:
        return None

    # Skip if not present in metadata
    if column_name not in table_meta:
        return None

    # Check if this is a primary key
    if infer_primary_key(column_value):
        return {
            'name': column_name,
            'dtype': 'primary_key',
            'description': table_meta[column_name][1]
        }

    # Regular column
    return {
        'name': column_name,
        'dtype': table_meta[column_name][0],
        'description': table_meta[column_name][1]
    }


def build_table_schema(table_name: str, table_data: Dict[str, np.ndarray],
                      meta_dict: Dict[str, Any], data_meta_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build schema for a single table.

    Args:
        table_name: Name of the table
        table_data: Dictionary mapping column names to data arrays
        meta_dict: Metadata dictionary for all tables
        data_meta_dict: Dataset metadata dictionary

    Returns:
        Table schema dictionary
    """
    table_schema = {
        'name': table_name,
        'columns': [],
        'format': data_meta_dict[table_name].format.value,
        'source': data_meta_dict[table_name].source
    }

    for column_name, column_value in table_data.items():
        column_schema = build_column_schema(column_name, column_value, meta_dict, table_name)
        if column_schema is not None:
            table_schema['columns'].append(column_schema)

    return table_schema


def build_task_schema(task, meta_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build schema for a single task.

    Args:
        task: Task metadata object
        meta_dict: Metadata dictionary for all tables

    Returns:
        Task schema dictionary
    """
    task_schema = {
        'name': task.name,
        'task_type': task.task_type,
        'target_column': task.target_column,
        'target_table': task.target_table,
        'evaluation_metric': task.evaluation_metric,
        'format': task.format,
        'source': task.source,
        'columns': []
    }

    # Find target table metadata (case-insensitive)
    target_table_meta = _find_table_meta(meta_dict, task.target_table)

    for column in task.columns:
        # Determine if this is a special column
        is_special = (
            column.name == task.target_column or
            column.dtype == 'datetime'
        )

        # Skip non-special columns not in metadata
        if not is_special:
            if target_table_meta is None or column.name not in target_table_meta:
                continue

        column_info = {
            'name': column.name,
            'dtype': (
                column.dtype if is_special
                else target_table_meta[column.name][0]
            )
        }
        task_schema['columns'].append(column_info)

    return task_schema


def generate_training_metainfo(data, meta_dict: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """Generate complete metadata for training.

    Args:
        data: Dataset object containing tables and metadata
        meta_dict: Metadata dictionary mapping table->column->type info
        task_name: Name of the task to generate metadata for

    Returns:
        Dictionary containing dataset metadata including tables and tasks
    """
    # Convert data metadata list to dict for easier lookup
    data_meta_dict = {table.name: table for table in data.metadata.tables}

    # Build table schemas
    tables = []
    for table_name in data.tables:
        table_data = data.tables[table_name]
        table_schema = build_table_schema(table_name, table_data, meta_dict, data_meta_dict)
        tables.append(table_schema)

    # Build task schemas
    tasks = []
    for task in data.metadata.tasks:
        if task.name == task_name:
            task_schema = build_task_schema(task, meta_dict)
            tasks.append(task_schema)

    return {
        'dataset_name': data.dataset_name,
        'tables': tables,
        'tasks': tasks
    }


def read_txt_dict(file_path: Path) -> Dict[str, Any]:
    """Read dictionary from a text file.

    Args:
        file_path: Path to the text file

    Returns:
        Dictionary parsed from file contents
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return ast.literal_eval(content)


def find_metadata_yaml(dataset_path: Path) -> Path:
    """Find metadata.yaml file with clear fallback logic.

    Searches for metadata.yaml in the dataset directory, falling back to
    the 'old' subdirectory if not found at the root level.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Path to the directory containing metadata.yaml

    Raises:
        FileNotFoundError: If metadata.yaml not found in expected locations
    """
    # Check root level first
    if (dataset_path / "metadata.yaml").exists():
        return dataset_path

    # Check 'old' subdirectory
    old_path = dataset_path / "old"
    if (old_path / "metadata.yaml").exists():
        return old_path

    # Not found in either location
    raise FileNotFoundError(
        f"metadata.yaml not found in {dataset_path} or {old_path}. "
        f"Please ensure dataset is properly structured."
    )


def capitalize_first_alpha_concise(text: str) -> str:
    """Capitalize the first alphabetic character in text.

    Args:
        text: Input string

    Returns:
        String with first alpha character capitalized
    """
    for i, char in enumerate(text):
        if char.isalpha():
            return text[:i] + text[i:].replace(char, char.upper(), 1)
    return text


# ============================================================================
# Validation Functions
# ============================================================================

def check_graphviz_installed() -> bool:
    """Check if Graphviz dot executable is available.

    Returns:
        True if graphviz is installed, False otherwise
    """
    import shutil
    return shutil.which("dot") is not None


def validate_config(config: AutoGConfig) -> None:
    """Validate AutoG configuration.

    Args:
        config: AutoG configuration object

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required files/directories don't exist
    """
    # Validate schema path exists
    if not config.schema_path.exists():
        raise FileNotFoundError(
            f"Schema path does not exist: {config.schema_path}"
        )

    # Validate dataset directory exists
    dataset_path = config.schema_path / config.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_path}"
        )

    # Validate column types file exists
    types_file = dataset_path / config.column_types_file
    if not types_file.exists():
        raise FileNotFoundError(
            f"Column types file does not exist: {types_file}. "
            f"Expected at {types_file}"
        )

    # Validate information.txt exists
    info_file = dataset_path / "information.txt"
    if not info_file.exists():
        raise FileNotFoundError(
            f"Dataset information file does not exist: {info_file}"
        )

    # Validate metadata.yaml can be found
    try:
        find_metadata_yaml(dataset_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cannot find metadata.yaml for dataset '{config.dataset}': {e}"
        )

    # Validate LM path if provided
    if config.lm_path and not config.lm_path.exists():
        raise FileNotFoundError(
            f"Language model path does not exist: {config.lm_path}"
        )

    # Validate dataset name is not empty
    if not config.dataset.strip():
        raise ValueError("Dataset name cannot be empty")

    # Validate task name is not empty
    if not config.task_name.strip():
        raise ValueError("Task name cannot be empty")

    # Check for Graphviz (optional but recommended)
    if not check_graphviz_installed():
        import warnings
        warnings.warn(
            "Graphviz 'dot' executable not found in PATH. "
            "Graph visualizations will fail. Install with:\n"
            "  Ubuntu/Debian: sudo apt-get install graphviz\n"
            "  macOS: brew install graphviz\n"
            "  Or skip visualization features.",
            UserWarning
        )


# ============================================================================
# Core Execution Functions
# ============================================================================

def run_autog(config: AutoGConfig) -> AutoGResult:
    """Execute AutoG with given configuration.

    Args:
        config: AutoG configuration

    Returns:
        AutoGResult containing final schema and execution history

    Raises:
        Various exceptions if execution fails
    """
    # Set random seed
    seed_everything(config.seed)

    # Build paths
    dataset_path = config.schema_path / config.dataset
    metainfo_path = dataset_path / config.column_types_file
    information_path = dataset_path / "information.txt"

    # Load metadata
    metainfo = read_txt_dict(metainfo_path)

    # Load dataset information
    with open(information_path, 'r') as file:
        information = file.read()

    # Find and load data configuration
    data_config_path = find_metadata_yaml(dataset_path)
    multi_tabular_data = load_dbb_dataset_from_cfg_path_no_name(data_config_path)

    # Get task description
    task_description = get_task_description(config.dataset, config.task_name)

    # Generate schema input
    schema_input = generate_training_metainfo(
        multi_tabular_data,
        metainfo,
        task_name=config.task_name
    )

    # Initialize AutoG agent
    agent = AutoG_Agent(
        initial_schema=schema_input,
        mode=config.method.value,
        oracle=None,
        path_to_file=str(data_config_path),
        use_cache=config.use_cache,
        threshold=10,
        task_description=task_description,
        dataset=config.dataset,
        task_name=config.task_name,
        schema_info=information,
        lm_path=str(config.lm_path) if config.lm_path else None,
        data_type_file=config.column_types_file
    )

    # Run augmentation
    agent.augment()

    # Build result
    return AutoGResult(
        final_schema=schema_input,  # Agent modifies this in-place
        history=agent.history,
        output_path=data_config_path,
        dataset=config.dataset,
        task_name=config.task_name,
        method=config.method.value
    )


def main(
    dataset: str = typer.Argument(
        ...,
        help="Dataset name (e.g., 'movielens', 'mag', 'ieee-cis')"
    ),
    schema_path: str = typer.Argument(
        ...,
        help="Root path containing dataset directories"
    ),
    method: str = typer.Argument(
        AutoGMethod.SELECTIVE.value,
        help=f"AutoG method: {AutoGMethod.SELECTIVE.value} (final schema) or {AutoGMethod.AUGMENTED.value} (all rounds)"
    ),
    task_name: str = typer.Argument(
        ...,
        help="Task name (e.g., 'ratings', 'venue', 'fraud')"
    ),
    column_types_file: str = typer.Option(
        "type.txt",
        "--types-file",
        help="Column type annotations file name"
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Random seed for reproducibility"
    ),
    lm_path: Optional[str] = typer.Option(
        "deepjoin/all-mpnet-base-v2",
        "--lm-path",
        help="Path to DeepJoin language model"
    ),
    use_cache: bool = typer.Option(
        True,
        "--use-cache/--no-cache",
        help="Use cached LLM responses from test/ directory (for testing)"
    )
):
    """Run AutoG agent for automatic graph schema generation from tabular data.

    AutoG uses LLMs to iteratively refine database schemas into optimized
    graph structures for Graph Machine Learning tasks.

    Examples:
        # MovieLens ratings with AutoG-Selective
        python -m main.autog movielens datasets autog-s ratings

        # MAG venue prediction
        python -m main.autog mag datasets autog-s venue --seed 42

        # IEEE-CIS fraud with custom types file
        python -m main.autog ieee-cis datasets autog-s fraud --types-file custom_types.txt
    """
    typer.echo("AutoG: Automatic Graph Schema Generation")
    typer.echo(f"Dataset: {dataset}, Task: {task_name}, Method: {method}")

    # Create configuration
    config = AutoGConfig(
        dataset=dataset,
        schema_path=Path(schema_path),
        method=method,  # Will be converted to enum in __post_init__
        task_name=task_name,
        column_types_file=column_types_file,
        seed=seed,
        lm_path=Path(lm_path) if lm_path else None,
        use_cache=use_cache
    )

    # Validate configuration
    try:
        validate_config(config)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1)

    # Run AutoG
    try:
        result = run_autog(config)
    except Exception as e:
        typer.echo(f"AutoG execution failed: {e}", err=True)
        raise

    # Display results
    typer.echo(f"\nAutoG completed successfully!")
    typer.echo(f"Output path: {result.output_path}")
    typer.echo(f"Schema generated for {len(result.final_schema.get('tables', []))} tables")
    typer.echo(f"\nAugmentation history ({len(result.history)} steps):")
    for step in result.history:
        typer.echo(f"  - {step}")


if __name__ == '__main__':
    traceback.install(show_locals=True)
    typer.run(main)