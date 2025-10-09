"""
Improved AutoG Agent implementation with better code organization and type safety.

Key improvements:
- Better class organization with separation of concerns
- Type hints throughout
- Improved error handling
- Extracted helper classes for different responsibilities
- Reduced code duplication
- Better documentation
"""

import os
import ast
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy
from dataclasses import dataclass, field

import typer
import joblib

from models.autog.action_new import (
    get_autog_actions,
    pack_function_introduction_prompt,
    turn_dbb_into_a_lookup_table
)
from prompts.mautog import (
    get_multi_round_action_selection_prompt,
    get_single_round_multi_step_prompt
)
from models.llm.gconstruct import (
    extract_between_tags,
    analyze_dataframes,
    dummy_llm_interaction
)
from models.autog.deepjoin import join_discovery, load_pretrain_jtd_lm
from utils.plot import plot_rdb_dataset_schema
from dbinfer_bench.dataset_meta import DBBColumnSchema
from dbinfer_bench.rdb_dataset import DBBRDBDataset


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class AutoGConfig:
    """Configuration for AutoG agent."""

    # Core settings
    mode: str = "autog-s"  # "autog-a" or "autog-s"
    threshold: int = 10  # Max rounds
    llm_sleep: float = 0.5  # Sleep between API calls
    use_cache: bool = False

    # Paths
    path_to_file: str = ""
    lm_path: str = ""  # DeepJoin model path
    data_type_file: str = ""

    # Dataset/Task info
    dataset: str = "mag"
    task_name: str = "venue"
    task_description: str = "autog"
    schema_info: str = ""

    # DeepJoin settings
    jtd_k: int = 20  # Top-k similar columns
    recalculate: bool = True

    # Other
    update_task: bool = False
    oracle: Optional[Any] = None


# ============================================================================
# Helper Classes
# ============================================================================

class DeepJoinHelper:
    """Helper for DeepJoin similarity calculations."""

    def __init__(self, config: AutoGConfig):
        self.config = config
        self._model = None

    def load_model(self):
        """Lazy load DeepJoin model."""
        if self._model is None:
            self._model = load_pretrain_jtd_lm(self.config.lm_path)
        return self._model

    def calculate_similarities(self, dbb, round_num: int) -> str:
        """Calculate column similarities using DeepJoin."""
        if self.config.jtd_k == 0:
            return ""

        cache_dir = Path(self.config.path_to_file) / f'round_{round_num}'
        cache_file = cache_dir / 'deepjoin.pkl'
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Try to load from cache
        if cache_file.exists() and not self.config.recalculate:
            typer.echo("Loading DeepJoin from cache")
            result = joblib.load(cache_file)
        elif self.config.recalculate and round_num > 0:
            # For rounds > 0, try to load round 0 cache
            round_0_cache = Path(self.config.path_to_file) / 'round_0' / 'deepjoin.pkl'
            if round_0_cache.exists():
                typer.echo("Loading DeepJoin from round 0 cache")
                result = joblib.load(round_0_cache)
            else:
                result = self._compute_similarities(dbb)
        else:
            result = self._compute_similarities(dbb)
            # Cache first round only
            if round_num == 0:
                joblib.dump(result, cache_file)

        return self._format_top_k_similarities(dbb, result)

    def _compute_similarities(self, dbb) -> Dict:
        """Compute similarities using model."""
        typer.echo("Computing DeepJoin similarities")
        model = self.load_model()
        return join_discovery(dbb, model)

    def _format_top_k_similarities(self, dbb, similarity_dict: Dict[Tuple[str, str, str, str], float]) -> str:
        """Format top-k most similar column pairs."""
        lookup = turn_dbb_into_a_lookup_table(dbb)
        sorted_sims = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

        output_lines = []
        valid_pair_count = 0

        for (table1, col1, table2, col2), similarity in sorted_sims:
            if valid_pair_count >= self.config.jtd_k:
                break

            # Skip if columns are missing
            if not lookup.get((table1, col1)) or not lookup.get((table2, col2)):
                continue

            # Skip if already connected
            if self._are_columns_connected(lookup, table1, col1, table2, col2):
                continue

            # Skip float columns
            if lookup[(table1, col1)].dtype == 'float' or lookup[(table2, col2)].dtype == 'float':
                continue

            valid_pair_count += 1
            ordinal = self._get_ordinal(valid_pair_count)
            output_lines.append(
                f'The pair with the {ordinal} highest similarity is column '
                f'"{col1}" from Table "{table1}" and column "{col2}" from Table "{table2}" '
                f'with similarity {similarity:.3f}'
            )

        return '\n'.join(output_lines)

    @staticmethod
    def _are_columns_connected(lookup: Dict, t1: str, c1: str, t2: str, c2: str) -> bool:
        """Check if two columns are already connected."""
        col1, col2 = lookup[(t1, c1)], lookup[(t2, c2)]

        # Direct FK relationship
        if hasattr(col1, 'link_to') and col1.link_to == f"{t2}.{c2}":
            return True
        if hasattr(col2, 'link_to') and col2.link_to == f"{t1}.{c1}":
            return True

        # Both point to same target
        if (hasattr(col1, 'link_to') and hasattr(col2, 'link_to') and
            col1.link_to == col2.link_to):
            return True

        # PK-FK relationship
        if (col1.dtype == 'primary_key' and col2.dtype == 'foreign_key' and
            hasattr(col2, 'link_to') and col2.link_to == f"{t1}.{c1}"):
            return True
        if (col2.dtype == 'primary_key' and col1.dtype == 'foreign_key' and
            hasattr(col1, 'link_to') and col1.link_to == f"{t2}.{c2}"):
            return True

        return False

    @staticmethod
    def _get_ordinal(n: int) -> str:
        """Get ordinal string (1st, 2nd, 3rd, etc.)."""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"


class PromptBuilder:
    """Builder for AutoG prompts."""

    def __init__(self, config: AutoGConfig, deepjoin_helper: DeepJoinHelper):
        self.config = config
        self.deepjoin_helper = deepjoin_helper
        self.action_list = get_autog_actions()
        self.icl_demonstrations = self._load_demonstrations()

    def _load_demonstrations(self) -> List[str]:
        """Load in-context learning demonstrations."""
        return get_single_round_multi_step_prompt()

    def build_prompt(self, dbb, history: List[str], round_num: int) -> str:
        """Build complete prompt for LLM."""
        # Examples
        example_str = "\n".join(self.icl_demonstrations)

        # Action descriptions
        action_strs = [
            pack_function_introduction_prompt(func)
            for func in self.action_list.values()
        ]
        action_description = "\n\n".join(action_strs)

        # History
        history_str = (
            "\n\n".join(history) if history
            else "First iteration, no history yet\n\n"
        )

        # Schema
        schema = dbb.metadata.json()

        # Statistics
        stats = self._get_statistics(dbb)

        # DeepJoin prior
        deepjoin_prior = self.deepjoin_helper.calculate_similarities(dbb, round_num)

        return get_multi_round_action_selection_prompt(
            action_description, example_str, history_str,
            schema, stats, self.config.task_description, deepjoin_prior
        )

    def _get_statistics(self, dbb) -> str:
        """Get dataset statistics."""
        if not self.config.recalculate:
            return self.config.schema_info

        table_meta_dict = {
            f'Table {table_name}': table
            for table_name, table in dbb.tables.items()
        }

        if self.config.dataset == 'stackexchange':
            return analyze_dataframes(table_meta_dict, dbb=dbb)
        else:
            return analyze_dataframes(table_meta_dict)


class TaskUpdater:
    """Handles task metadata updates."""

    def __init__(self, config: AutoGConfig):
        self.config = config

    def update_task(self, dbb) -> 'DBBRDBDataset':
        """Update task type information."""
        if self.config.dataset == 'diginetica':
            return dbb

        # Update table metadata
        self._update_table_metadata(dbb)

        # Update task metadata
        if dbb.metadata.tasks:
            self._update_task_metadata(dbb)

        return dbb

    def _update_table_metadata(self, dbb):
        """Update table metadata - convert multi_category to category."""
        for table in dbb.metadata.tables:
            time_column = None
            for column in table.columns:
                if column.dtype == 'multi_category':
                    column.dtype = 'category'
                elif column.dtype == 'datetime':
                    time_column = column.name
            table.time_column = time_column

    def _update_task_metadata(self, dbb):
        """Update task column metadata."""
        task = dbb.metadata.tasks[0]
        target_table = task.target_table

        # Find target table
        target_table_schema = next(
            (t for t in dbb.metadata.tables if t.name == target_table),
            None
        )
        if not target_table_schema:
            return

        # Build mappings
        column_type_map = {col.name: col.dtype for col in target_table_schema.columns}
        link_to_map = {
            col.name: col.link_to
            for col in target_table_schema.columns
            if hasattr(col, 'link_to')
        }
        valid_columns = {col.name for col in target_table_schema.columns}

        # Update task columns
        updated_columns = []
        columns_to_remove = []

        for col in task.columns:
            # Keep target and datetime columns
            if col.name == task.target_column or col.dtype == 'datetime':
                updated_columns.append(col)
                continue

            # Remove columns not in target table
            if col.name not in valid_columns:
                columns_to_remove.append(col.name)
                continue

            # Update column type
            if col.name in column_type_map:
                col.dtype = column_type_map[col.name]

            # Update link_to for FKs
            if col.dtype == 'foreign_key' and col.name in link_to_map:
                col.link_to = link_to_map[col.name]

            updated_columns.append(col)

        # Remove columns from task data
        for col_name in columns_to_remove:
            for split in ['train_set', 'validation_set', 'test_set']:
                dataset = getattr(dbb.tasks[0], split)
                if col_name in dataset:
                    dataset.pop(col_name)

        task.columns = updated_columns
        dbb.metadata.tasks[0] = task


class DatasetPostProcessor:
    """Handles dataset-specific post-processing."""

    DATASET_PROCESSORS = {}

    def __init__(self, config: AutoGConfig):
        self.config = config
        self._register_processors()

    def _register_processors(self):
        """Register dataset-specific processors."""
        self.DATASET_PROCESSORS = {
            'outbrain': self._process_outbrain,
            'avs': self._process_avs,
            'diginetica': self._process_diginetica,
        }

    def process(self, dbb) -> 'DBBRDBDataset':
        """Apply dataset-specific post-processing."""
        processor = self.DATASET_PROCESSORS.get(self.config.dataset)
        if processor:
            return processor(dbb)
        return dbb

    def _process_outbrain(self, dbb):
        """Post-process Outbrain dataset."""
        time_column_map = {
            'Event': 'timestamp',
            'PageView': 'timestamp',
            'Click': 'timestamp',
            'DocumentsMeta': 'publish_time'
        }

        for table in dbb.metadata.tables:
            if table.name in time_column_map:
                table.time_column = time_column_map[table.name]

        if dbb.metadata.tasks:
            dbb.metadata.tasks[0].time_column = 'timestamp'

        return dbb

    def _process_avs(self, dbb):
        """Post-process AVS dataset."""
        # Set time columns
        time_column_map = {
            'History': 'offerdate',
            'Transaction': 'date'
        }

        for table in dbb.metadata.tables:
            if table.name in time_column_map:
                table.time_column = time_column_map[table.name]

        # Remove repeater column from History table to prevent leakage
        for i, table in enumerate(dbb.metadata.tables):
            if table.name == 'History':
                for j, col in enumerate(table.columns):
                    if col.name == 'repeater':
                        dbb.metadata.tables[i].columns.pop(j)
                        break

        # Update task
        if dbb.tasks:
            dbb.tasks[0].metadata.columns.append(
                DBBColumnSchema(name='timestamp', dtype='datetime')
            )
            dbb.tasks[0].time_column = 'timestamp'

        if dbb.metadata.tasks:
            dbb.metadata.tasks[0].time_column = 'timestamp'

        return dbb

    def _process_diginetica(self, dbb):
        """Post-process Diginetica dataset."""
        # Set time columns
        time_tables = ['QueryResult', 'Click', 'View', 'Purchase', 'Query']
        for table in dbb.metadata.tables:
            if table.name in time_tables:
                table.time_column = 'timestamp'

        if not dbb.metadata.tasks:
            return dbb

        # FIXED: Sync task metadata columns with actual task data columns
        # Instead of hardcoding, build column schemas from actual data
        for task in dbb.tasks:
            # Get actual columns from task data (train_set is guaranteed to exist)
            actual_columns = set(task.train_set.keys())

            # Build column schema list matching actual data
            column_schemas = []

            # Build lookup of existing schemas
            existing_schemas = {col.name: col for col in task.metadata.columns}

            for col_name in sorted(actual_columns):  # Sort for consistency
                if col_name in existing_schemas:
                    # Reuse existing schema
                    column_schemas.append(existing_schemas[col_name])
                else:
                    # Infer schema for new columns
                    # Check if it's a known column type
                    if col_name == 'timestamp':
                        column_schemas.append(DBBColumnSchema(name='timestamp', dtype='datetime'))
                    elif col_name.endswith('Id') or col_name.endswith('_id'):
                        # Try to find FK relationship from main tables
                        fk_link = None
                        for tbl in dbb.metadata.tables:
                            for tcol in tbl.columns:
                                if tcol.name == col_name and tcol.dtype == 'foreign_key':
                                    fk_link = getattr(tcol, 'link_to', None)
                                    break
                            if fk_link:
                                break
                        if fk_link:
                            column_schemas.append(DBBColumnSchema(
                                name=col_name,
                                dtype='foreign_key',
                                link_to=fk_link
                            ))
                        else:
                            # Default to category if no FK found
                            column_schemas.append(DBBColumnSchema(name=col_name, dtype='category'))
                    else:
                        # Default to category for unknown columns
                        column_schemas.append(DBBColumnSchema(name=col_name, dtype='category'))

            # Update task metadata
            task.metadata.columns = column_schemas
            if 'timestamp' in actual_columns:
                task.metadata.time_column = 'timestamp'

        # Also update metadata.tasks if present
        if dbb.metadata.tasks and dbb.tasks:
            for i, meta_task in enumerate(dbb.metadata.tasks):
                if i < len(dbb.tasks):
                    meta_task.columns = dbb.tasks[i].metadata.columns
                    if 'timestamp' in dbb.tasks[i].train_set:
                        meta_task.time_column = 'timestamp'

        return dbb


# ============================================================================
# Main Agent Class
# ============================================================================

class AutoG_Agent:
    """
    Main AutoG agent for automatic graph schema generation.

    The agent iteratively refines database schemas by proposing and executing
    actions (add edges, remove columns, etc.) based on LLM guidance and
    DeepJoin similarity analysis.
    """

    def __init__(
        self,
        initial_schema: Dict,
        mode: str = "autog-s",
        oracle: Optional[Any] = None,
        path_to_file: str = "",
        use_cache: bool = False,
        threshold: int = 10,
        llm_sleep: float = 0.5,
        task_description: str = 'autog',
        dataset: str = 'mag',
        task_name: str = 'venue',
        schema_info: str = "",
        lm_path: str = "",
        jtd_k: int = 20,
        recalculate: bool = True,
        data_type_file: str = "",
        update_task: bool = False
    ):
        """Initialize AutoG agent with configuration."""

        # Create configuration
        self.config = AutoGConfig(
            mode=mode,
            threshold=threshold,
            llm_sleep=llm_sleep,
            use_cache=use_cache,
            path_to_file=path_to_file,
            lm_path=lm_path,
            data_type_file=data_type_file,
            dataset=dataset,
            task_name=task_name,
            task_description=task_description,
            schema_info=schema_info,
            jtd_k=jtd_k,
            recalculate=recalculate,
            update_task=update_task,
            oracle=oracle
        )

        # Initialize helpers
        self.deepjoin_helper = DeepJoinHelper(self.config)
        self.prompt_builder = PromptBuilder(self.config, self.deepjoin_helper)
        self.task_updater = TaskUpdater(self.config)
        self.post_processor = DatasetPostProcessor(self.config)

        # State
        self.action_list = get_autog_actions()
        self.state = initial_schema
        self.original_state = deepcopy(initial_schema)
        self.history: List[str] = []
        self.round = 0

        # Statistics
        self.error_count = 0
        self.success_count = 0

    def augment(self):
        """Run the schema augmentation process."""
        typer.echo("Starting schema augmentation")

        # Load initial database
        dbb = DBBRDBDataset(self.config.path_to_file)

        # Iterative refinement
        for round_num in range(self.config.threshold):
            typer.echo(f"Round {round_num}")

            dbb, should_continue = self._execute_round(dbb, round_num)
            self.round += 1

            if not should_continue:
                self._finalize_schema(dbb)
                return

        typer.echo(f"Reached maximum rounds ({self.config.threshold})")
        self._finalize_schema(dbb)

    def _execute_round(self, dbb, round_num: int) -> Tuple['DBBRDBDataset', bool]:
        """Execute a single refinement round."""
        # Create round directory
        round_dir = Path(self.config.path_to_file) / f'{self.config.task_name}_round_{round_num}'
        round_dir.mkdir(parents=True, exist_ok=True)

        # Build and save prompt
        prompt = self.prompt_builder.build_prompt(dbb, self.history, round_num)
        query_file = round_dir / 'query.txt'
        response_file = round_dir / 'response.txt'

        # Get LLM response (with cache support)
        response = dummy_llm_interaction(
            query_text=prompt,
            query_filepath=str(query_file),
            response_filepath=str(response_file),
            use_cache=self.config.use_cache,
            dataset=self.config.dataset,
            task=self.config.task_name,
            round_num=round_num
        )

        # Parse response
        try:
            selection = extract_between_tags(response, "selection")[0].strip()
        except (IndexError, AttributeError) as e:
            typer.echo(f"Failed to extract selection: {e}")
            return dbb, False

        if selection == "None":
            typer.echo("No more actions selected")
            return dbb, False

        # Execute actions
        try:
            actions = json.loads(selection)
        except json.JSONDecodeError as e:
            typer.echo(f"Failed to parse actions: {e}")
            return dbb, False

        dbb = self._execute_actions(dbb, actions)

        # For autog-s mode, stop after one round
        return dbb, False

    def _execute_actions(self, dbb, actions: List[Dict]) -> 'DBBRDBDataset':
        """Execute a list of actions on the database."""
        for action_spec in actions:
            typer.echo(f"Executing: {action_spec.get('action', 'unknown')}")
            dbb = self._execute_single_action(dbb, action_spec)

        return dbb

    def _execute_single_action(self, dbb, action_spec: Dict) -> 'DBBRDBDataset':
        """Execute a single action with error handling."""
        action_name = action_spec.get('action')
        # IMPORTANT: Make a copy to avoid mutating the original action_spec
        parameters = action_spec.get('parameters', {}).copy()
        explanation = action_spec.get('explanation', '')

        if action_name not in self.action_list:
            typer.echo(f"Unknown action: {action_name}")
            self.error_count += 1
            return dbb

        # Backup state
        last_valid_dbb = deepcopy(dbb)

        try:
            # Execute action
            action_func = self.action_list[action_name]
            parameters['dbb'] = dbb
            dbb = action_func(**parameters)

            # Record success
            parameters_without_dbb = {k: v for k, v in parameters.items() if k != 'dbb'}
            action_record = {
                'action': action_name,
                'parameters': parameters_without_dbb,
                'explanation': explanation
            }
            self.history.append(json.dumps(action_record))
            self.success_count += 1

            # Set method
            dbb.method = 'r2n'

            # Backup for autog-a mode
            if self.config.mode == 'autog-a':
                if self.config.update_task:
                    dbb = self.task_updater.update_task(dbb)
                self._backup_state(dbb)

            # Update task if needed
            if self.config.update_task:
                dbb = self.task_updater.update_task(dbb)

        except Exception as e:
            typer.echo(f"Action failed: {e}")
            # Remove dbb from action_spec before JSON serialization
            action_spec_without_dbb = {
                'action': action_spec.get('action'),
                'parameters': {k: v for k, v in action_spec.get('parameters', {}).items() if k != 'dbb'},
                'explanation': action_spec.get('explanation', '')
            }
            self.history.append(f"Error: {str(e)} | Action: {json.dumps(action_spec_without_dbb)}")
            self.error_count += 1
            dbb = last_valid_dbb

            # Stop if too many errors
            if self.error_count >= 3:
                typer.echo("Too many errors, halting")

        return dbb

    def _backup_state(self, dbb):
        """Backup current state for autog-a mode."""
        backup_path = Path(self.config.path_to_file) / f"backup_{len(self.history)}"
        dbb.save(str(backup_path))

    def _finalize_schema(self, dbb):
        """Finalize and save the schema."""
        # Apply post-processing
        dbb = self.post_processor.process(dbb)

        # Save
        final_path = Path(self.config.path_to_file) / 'final'
        dbb.save(str(final_path))

        # Plot schema
        plot_rdb_dataset_schema(dbb, str(final_path / 'schema'))

        typer.echo(f"Schema saved to {final_path}")
        typer.echo(f"Statistics: {self.success_count} successes, {self.error_count} errors")

    def get_current_state(self) -> Dict:
        """Get current schema state."""
        return self.state


# ============================================================================
# Utility Functions
# ============================================================================

def load_dbb_dataset_from_cfg_path_no_name(cfg_path: str) -> DBBRDBDataset:
    """Load DBB dataset from configuration path."""
    return DBBRDBDataset(cfg_path)
