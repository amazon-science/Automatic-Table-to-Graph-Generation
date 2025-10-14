import inspect
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import numpy as np
import pandas as pd
import duckdb
from dbinfer_bench.dataset_meta import DBBColumnSchema, DBBTableSchema, DBBColumnDType


# ============================================================================
# Helper Functions
# ============================================================================

def _quote_identifier(name: str) -> str:
    """Quote SQL identifier to prevent injection and handle reserved words."""
    return '"' + name.replace('"', '""') + '"'


def _is_null(value: Any) -> bool:
    """Check if value is null (None or NaN)."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _coerce_dtype(dt: Union[str, DBBColumnDType]) -> DBBColumnDType:
    """
    Coerce dtype to DBBColumnDType enum.

    Handles both string and enum inputs to ensure type consistency.
    """
    if isinstance(dt, DBBColumnDType):
        return dt
    # Try by value first, then by attribute name
    try:
        return DBBColumnDType(dt)
    except (ValueError, KeyError):
        try:
            return getattr(DBBColumnDType, dt)
        except AttributeError:
            raise ValueError(f"Cannot coerce '{dt}' to DBBColumnDType")


# ============================================================================
# Schema Helper Class
# ============================================================================

class SchemaHelper:
    """Helper class for schema lookup and modification operations."""

    def __init__(self, dbb):
        self.dbb = dbb
        self._build_lookups()

    def _build_lookups(self):
        """Build lookup dictionaries for fast schema access."""
        self.column_lookup: Dict[Tuple[str, str], DBBColumnSchema] = {}
        self.table_lookup: Dict[str, DBBTableSchema] = {}

        for table in self.dbb.metadata.tables:
            self.table_lookup[table.name] = table
            for column in table.columns:
                self.column_lookup[(table.name, column.name)] = column

    def get_column(self, table_name: str, col_name: str) -> Optional[DBBColumnSchema]:
        """Get column schema safely."""
        return self.column_lookup.get((table_name, col_name))

    def get_table(self, table_name: str) -> Optional[DBBTableSchema]:
        """Get table schema safely."""
        return self.table_lookup.get(table_name)

    def get_column_type(self, table_name: str, col_name: str) -> Optional[DBBColumnDType]:
        """Get column dtype (FIXED: return enum not string)."""
        col = self.get_column(table_name, col_name)
        return col.dtype if col else None

    def find_primary_key(self, table_name: str) -> Optional[str]:
        """Find primary key column name for a table."""
        table = self.get_table(table_name)
        if not table:
            return None
        for col in table.columns:
            if col.dtype == DBBColumnDType.primary_key:
                return col.name
        return None

    def count_primary_keys(self) -> int:
        """Count total number of primary keys in database."""
        count = 0
        for table in self.dbb.metadata.tables:
            for col in table.columns:
                if col.dtype == DBBColumnDType.primary_key:
                    count += 1
        return count

    def remove_column_from_table(self, table_name: str, col_name: str):
        """Remove column from table schema."""
        for table in self.dbb.metadata.tables:
            if table.name == table_name:
                table.columns = [col for col in table.columns if col.name != col_name]
                break
        self._build_lookups()

    def add_column_to_table(self, table_name: str, column_schema: DBBColumnSchema):
        """Add column to table schema (FIXED: prevent duplicates)."""
        for table in self.dbb.metadata.tables:
            if table.name == table_name:
                # Check if column already exists
                existing_names = {col.name for col in table.columns}
                if column_schema.name in existing_names:
                    print(f"Warning: Column '{column_schema.name}' already exists in table '{table_name}', skipping")
                    return
                table.columns.append(column_schema)
                break
        self._build_lookups()

    def add_table(self, table_schema: DBBTableSchema):
        """Add new table to database schema (FIXED: validate no duplicate columns)."""
        # Validate no duplicate column names within the table
        col_names = [col.name for col in table_schema.columns]
        if len(col_names) != len(set(col_names)):
            duplicates = [name for name in col_names if col_names.count(name) > 1]
            raise ValueError(f"Cannot create table '{table_schema.name}' with duplicate column names: {set(duplicates)}")

        self.dbb.metadata.tables.append(table_schema)
        self._build_lookups()

    def update_column_type(self, table_name: str, col_name: str,
                          new_dtype: DBBColumnDType, link_to: Optional[str] = None):
        """Update column dtype and optionally link_to (FIXED: use None not delattr)."""
        for table in self.dbb.metadata.tables:
            if table.name == table_name:
                for col in table.columns:
                    if col.name == col_name:
                        col.dtype = new_dtype
                        # FIXED: Set to None instead of delattr for dataclass compatibility
                        if new_dtype == DBBColumnDType.foreign_key:
                            if link_to:
                                col.link_to = link_to
                        else:
                            # Not FK - clear link_to by setting to None
                            if hasattr(col, 'link_to'):
                                col.link_to = None
                        break
                break
        self._build_lookups()

    def update_all_foreign_keys(self, old_link_to: str, new_link_to: str):
        """
        Update all foreign keys pointing to old_link_to to point to new_link_to.

        WARNING: This is a global operation that affects ALL FKs with the same target.
        Use with caution - intended for multi-column-point-to-one scenarios.
        """
        for table in self.dbb.metadata.tables:
            for col in table.columns:
                if col.dtype == DBBColumnDType.foreign_key and hasattr(col, 'link_to'):
                    if col.link_to == old_link_to:
                        col.link_to = new_link_to
        self._build_lookups()


# ============================================================================
# DuckDB Helper Class
# ============================================================================

class DuckDBHelper:
    """Helper class for data transformations using DuckDB."""

    def __init__(self, dbb):
        self.dbb = dbb
        self.conn = duckdb.connect(':memory:')

    def register_table(self, table_name: str):
        """
        Register a table from dbb into DuckDB.

        FIXED: Validate column lengths and skip multi-dimensional columns
        (embeddings, arrays) that can't be processed by DuckDB/pandas.
        """
        data = self.dbb.tables[table_name]

        # Filter out multi-dimensional columns (embeddings, nested arrays)
        filtered_data = {}
        for col_name, col_data in data.items():
            if isinstance(col_data, np.ndarray):
                if col_data.ndim > 1:
                    # Skip multi-dimensional arrays (embeddings, etc.)
                    print(f"Skipping multi-dimensional column '{col_name}' in table '{table_name}' (shape={col_data.shape})")
                    continue
                # Also skip object arrays that contain lists/arrays
                if col_data.dtype == object and len(col_data) > 0:
                    first_elem = col_data[0]
                    if isinstance(first_elem, (list, np.ndarray)):
                        print(f"Skipping nested array column '{col_name}' in table '{table_name}'")
                        continue
            filtered_data[col_name] = col_data

        if not filtered_data:
            raise ValueError(f"Table '{table_name}' has no valid 1D columns to register")

        # Validate all columns have same length
        lens = {len(v) for v in filtered_data.values()}
        if len(lens) > 1:
            raise ValueError(
                f"Inconsistent column lengths in '{table_name}': {lens}. "
                f"All columns must have the same number of rows."
            )

        df = pd.DataFrame(filtered_data)
        self.conn.register(table_name, df)

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return result as DataFrame."""
        return self.conn.execute(query).df()

    def get_unique_values(self, table_name: str, col_name: str) -> np.ndarray:
        """Get unique values from a column (FIXED: quoted identifiers)."""
        self.register_table(table_name)
        t_quoted = _quote_identifier(table_name)
        c_quoted = _quote_identifier(col_name)
        query = f"SELECT DISTINCT {c_quoted} FROM {t_quoted} WHERE {c_quoted} IS NOT NULL"
        result = self.execute_query(query)
        return result[col_name].to_numpy()

    def get_union_values(self, table1: str, col1: str, table2: str, col2: str) -> np.ndarray:
        """Get union of unique values from two columns (FIXED: no ORDER BY for mixed types)."""
        self.register_table(table1)
        self.register_table(table2)

        t1_q = _quote_identifier(table1)
        c1_q = _quote_identifier(col1)
        t2_q = _quote_identifier(table2)
        c2_q = _quote_identifier(col2)

        # FIXED: Removed ORDER BY to handle mixed-type columns (int/string)
        # UNION already deduplicates, DISTINCT is redundant but kept for clarity
        query = f"""
            SELECT DISTINCT value FROM (
                SELECT {c1_q} as value FROM {t1_q}
                UNION
                SELECT {c2_q} as value FROM {t2_q}
            ) WHERE value IS NOT NULL
        """
        result = self.execute_query(query)
        return result['value'].to_numpy()

    def create_mapping_table(self, values: np.ndarray, pk_name: str,
                            value_col_name: str) -> Dict[str, np.ndarray]:
        """Create a lookup table with primary key and values."""
        return {
            pk_name: np.arange(len(values)),
            value_col_name: values
        }

    def map_values_to_keys(self, values: np.ndarray,
                          mapping: Dict[Any, int]) -> List[int]:
        """Map values to their corresponding keys using a mapping dict."""
        return [mapping.get(v, -1) for v in values]

    def explode_column(self, table_name: str, pk_col: str,
                      multi_col: str) -> pd.DataFrame:
        """Explode multi-value column into separate rows."""
        self.register_table(table_name)
        df = self.conn.execute(
            f"SELECT {_quote_identifier(pk_col)}, {_quote_identifier(multi_col)} FROM {_quote_identifier(table_name)}"
        ).df()
        return df.explode(multi_col).reset_index(drop=True)

    def normalize_columns(self, table_name: str, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract columns and create normalized table with unique combinations.

        FIXED: Handle NaN values with collision-proof null token.
        """
        self.register_table(table_name)

        # Get distinct combinations
        cols_quoted = [_quote_identifier(c) for c in cols]
        cols_str = ', '.join(cols_quoted)
        query = f"SELECT DISTINCT {cols_str} FROM {_quote_identifier(table_name)}"
        unique_df = self.execute_query(query).reset_index(drop=True)
        unique_df = unique_df.reset_index().rename(columns={'index': 'new_id'})

        # FIXED: Use collision-proof null token to avoid merging legitimate -1 with nulls
        # Pandas treats NaN != NaN in joins, causing rows with nulls to get NaN FKs
        NULL_TOKEN = "\0NULL\0"  # Special token that can't collide with real data
        original_df = self.conn.execute(f"SELECT * FROM {_quote_identifier(table_name)}").df()

        # Make copies for key column creation
        orig = original_df.copy()
        uniq = unique_df.copy()

        # Create temporary join keys with null tokens
        for c in cols:
            orig[f"__key__{c}"] = orig[c].astype(object).where(orig[c].notna(), NULL_TOKEN)
            uniq[f"__key__{c}"] = uniq[c].astype(object).where(uniq[c].notna(), NULL_TOKEN)

        # Merge on the temporary key columns
        key_cols = [f"__key__{c}" for c in cols]
        merged_df = orig.merge(uniq[['new_id'] + key_cols], on=key_cols, how='left')

        # Drop temporary key columns from merged result
        merged_df = merged_df.drop(columns=key_cols)

        return unique_df, merged_df

    def close(self):
        """Close DuckDB connection."""
        self.conn.close()


# ============================================================================
# Action Functions
# ============================================================================

def remove_primary_key(dbb, base_table_name: str, col_name: str):
    """
    Remove a primary key constraint from a column in the original table.

    If the column is just an index, then the column will be removed from both
    schema and data. This is useful when a table acts as an edge (with only FKs)
    and the PK prevents proper edge detection.

    Args:
        dbb: The database object (DBBRDBDataset)
        base_table_name: The name of the table
        col_name: The name of the primary key column to remove

    Returns:
        dbb: Modified database object

    Example:
        Table with schema: [id (PK), user (FK), book (FK)]
        After removal: [user (FK), book (FK)]
    """
    helper = SchemaHelper(dbb)

    # Verify column exists and is a primary key
    col = helper.get_column(base_table_name, col_name)
    if not col:
        print(f"Warning: Column {col_name} not found in table {base_table_name}")
        return dbb

    if col.dtype != DBBColumnDType.primary_key:
        print(f"Warning: Column {col_name} is not a primary key")
        return dbb

    # Remove from schema
    helper.remove_column_from_table(base_table_name, col_name)

    # Remove from data
    if base_table_name in dbb.tables and col_name in dbb.tables[base_table_name]:
        del dbb.tables[base_table_name][col_name]

    return dbb


def add_primary_key(dbb, base_table_name: str, col_name: str):
    """
    Add a primary key column to a table.

    Creates an auto-incrementing integer column starting from 0.
    If the table already has a primary key, no changes are made.

    Args:
        dbb: The database object
        base_table_name: The name of the table
        col_name: The name for the new primary key column

    Returns:
        dbb: Modified database object
    """
    helper = SchemaHelper(dbb)

    # Check if table already has a primary key
    existing_pk = helper.find_primary_key(base_table_name)
    if existing_pk:
        print(f"Table {base_table_name} already has primary key: {existing_pk}")
        return dbb

    # Add to schema
    pk_schema = DBBColumnSchema(name=col_name, dtype=DBBColumnDType.primary_key)
    helper.add_column_to_table(base_table_name, pk_schema)

    # Add to data - create sequential IDs (FIXED: handle empty tables)
    n_rows = 0
    if base_table_name in dbb.tables and dbb.tables[base_table_name]:
        # Get any column to determine row count
        any_col = next(iter(dbb.tables[base_table_name].values()), None)
        if any_col is not None:
            n_rows = len(any_col)

    dbb.tables.setdefault(base_table_name, {})[col_name] = np.arange(n_rows, dtype=int)

    return dbb


def generate_or_connect_dummy_table(dbb, base_table_name: str, orig_col_name: str,
                                    new_table_name: str, new_col_name: str, **kwargs):
    """
    Convert a category column to a foreign key pointing to a dummy table.

    Following DBInfer convention: dummy tables (with only one value column) are NOT
    added to the schema - they're referenced implicitly via link_to (e.g., Brand.id).
    DBInfer creates them automatically from distinct values.

    The data is still created in dbb.tables for internal consistency, but no
    table schema is added to dbb.metadata.tables.
    """
    helper = SchemaHelper(dbb)
    db_helper = DuckDBHelper(dbb)

    try:
        # Validate source column
        col = helper.get_column(base_table_name, orig_col_name)
        if not col:
            print(f"Warning: Column {orig_col_name} not found in table {base_table_name}")
            return dbb

        # Check if target dummy table already exists in schema
        target = helper.get_table(new_table_name)

        # Determine PK name (convention: use new_col_name, or {Table}ID if ambiguous)
        # FIXED: Always use table-specific PK name to avoid conflicts
        pk_name = f"{new_table_name}ID"

        if target is None:
            # Create data for the dummy table (but NO schema entry per DBInfer convention)
            distinct_vals = db_helper.get_unique_values(base_table_name, orig_col_name)

            # Create data in dbb.tables (for internal use)
            dbb.tables[new_table_name] = {
                pk_name: np.arange(len(distinct_vals), dtype=int),
                orig_col_name: distinct_vals  # Keep original column name for value
            }

            # Build mapping for remapping source column
            mapping = {val: idx for idx, val in enumerate(distinct_vals) if not _is_null(val)}
        else:
            # Target table exists - find PK and value columns
            pk_candidates = [c.name for c in target.columns if c.dtype == DBBColumnDType.primary_key]
            if pk_candidates:
                pk_name = pk_candidates[0]

            # Find value column
            existing_cols = dbb.tables.get(new_table_name, {})
            value_col = orig_col_name if orig_col_name in existing_cols else next(
                (name for name in existing_cols if name != pk_name), orig_col_name
            )

            # Build mapping from existing data
            tgt_data = dbb.tables[new_table_name]
            mapping = {
                label: idx
                for label, idx in zip(tgt_data[value_col], tgt_data[pk_name])
                if not _is_null(label)
            }

        # Remap the base column to integer IDs
        src_arr = dbb.tables[base_table_name][orig_col_name]
        dbb.tables[base_table_name][orig_col_name] = np.array(
            [(-1 if _is_null(v) else mapping.get(v, -1)) for v in src_arr], dtype=int
        )

        # Update schema: change column to FK pointing to dummy table
        # DBInfer convention: link_to = "{TableName}.{pk_column_name}"
        link_to = f"{new_table_name}.{pk_name}"
        helper.update_column_type(base_table_name, orig_col_name,
                                  DBBColumnDType.foreign_key, link_to=link_to)

        return dbb
    finally:
        db_helper.close()



def connect_two_columns(dbb, table_1_name: str, table_1_col_name: str,
                       table_2_name: str, table_2_col_name: str, **kwargs):
    """
    Connect two columns, creating relationships based on their types.

    FIXED: Proper logic flow that doesn't block key connections.

    This function handles multiple scenarios:
    1. category + category: Create new surrogate table
    2. category + primary_key: Direct FK connection (if values are subset)
    3. foreign_key + foreign_key (same target): No change
    4. foreign_key + primary_key: Update FK to point to PK
    5. Two non-key columns of same type: Create surrogate key table

    Args:
        dbb: The database object
        table_1_name: Name of first table
        table_1_col_name: Name of column in first table
        table_2_name: Name of second table
        table_2_col_name: Name of column in second table

    Returns:
        dbb: Modified database object
    """
    helper = SchemaHelper(dbb)
    db_helper = DuckDBHelper(dbb)

    try:
        # Get column types
        col1 = helper.get_column(table_1_name, table_1_col_name)
        col2 = helper.get_column(table_2_name, table_2_col_name)

        # If columns don't exist, try to create dummy connection
        if not col1 or not col2:
            return generate_or_connect_dummy_table(dbb, table_1_name, table_1_col_name,
                                                  table_2_name, table_2_col_name)

        type1, type2 = col1.dtype, col2.dtype

        # Case 1: Both FK pointing to same target - no change needed
        if (type1 == DBBColumnDType.foreign_key and type2 == DBBColumnDType.foreign_key):
            if hasattr(col1, 'link_to') and hasattr(col2, 'link_to'):
                if col1.link_to == col2.link_to:
                    return dbb

        # FIXED: Orient so table_1 is the source and table_2 is the target
        # This must happen BEFORE type checking to allow category→PK connections
        if type1 == DBBColumnDType.primary_key:
            table_1_name, table_2_name = table_2_name, table_1_name
            table_1_col_name, table_2_col_name = table_2_col_name, table_1_col_name
            col1, col2 = col2, col1
            type1, type2 = type2, type1

        if type1 == DBBColumnDType.foreign_key and type2 == DBBColumnDType.category_t:
            table_1_name, table_2_name = table_2_name, table_1_name
            table_1_col_name, table_2_col_name = table_2_col_name, table_1_col_name
            col1, col2 = col2, col1
            type1, type2 = type2, type1

        # Handle FK update case
        update_fk = (type1 == DBBColumnDType.foreign_key)

        # Case 2: category + category → build surrogate table
        if type1 == DBBColumnDType.category_t and type2 == DBBColumnDType.category_t:
            return _connect_non_key_columns(dbb, helper, db_helper,
                                            table_1_name, table_1_col_name,
                                            table_2_name, table_2_col_name,
                                            DBBColumnDType.category_t)

        # Case 3: (category|FK) + PK → direct FK if values are subset, else surrogate
        if type2 == DBBColumnDType.primary_key:
            pk_values = db_helper.get_unique_values(table_2_name, table_2_col_name)
            fk_like = db_helper.get_unique_values(table_1_name, table_1_col_name)
            fk_like = fk_like[~pd.isna(fk_like)]

            n_pks = helper.count_primary_keys()
            dataset_name = dbb.metadata.dataset_name

            # Direct connection if FK values are subset of PK
            if np.all(np.isin(fk_like, pk_values)):
                helper.update_column_type(table_1_name, table_1_col_name,
                                          DBBColumnDType.foreign_key,
                                          link_to=f"{table_2_name}.{table_2_col_name}")
                return dbb

            # FIXED: Use bridge table if not subset to avoid demoting PK!
            # CRITICAL: _connect_non_key_columns would convert the PK to FK,
            # breaking all existing FK relationships pointing to it.
            ## we have to hardcode diginetica here because this dataset has some problems and it's convenient here to have it as an exception
            if n_pks > 1 and dataset_name != 'diginetica':
                return _bridge_category_to_pk(dbb, helper, db_helper,
                                              table_1_name, table_1_col_name,
                                              table_2_name, table_2_col_name)
            else:
                # Force connection even if values don't match (edge case for single PK)
                helper.update_column_type(table_1_name, table_1_col_name,
                                          DBBColumnDType.foreign_key,
                                          link_to=f"{table_2_name}.{table_2_col_name}")
                return dbb

        # Case 4: FK → FK (different targets)
        # FIXED: Make cascade opt-in via kwargs (was always True in this branch!)
        if type1 == DBBColumnDType.foreign_key and type2 == DBBColumnDType.foreign_key:
            target = getattr(col2, 'link_to', None)
            if target:
                old_target = getattr(col1, 'link_to', None)
                # FIXED: Cascade is opt-in via kwargs, defaults to False
                cascade = bool(kwargs.get("cascade", False))
                if cascade and old_target:
                    # Global retarget: update ALL FKs pointing to old target
                    helper.update_all_foreign_keys(old_target, target)
                else:
                    # Local retarget: update only this column
                    helper.update_column_type(table_1_name, table_1_col_name,
                                              DBBColumnDType.foreign_key,
                                              link_to=target)
            return dbb

        # Case 5: Non-key same-type → surrogate table
        if (type1 not in {DBBColumnDType.category_t, DBBColumnDType.primary_key, DBBColumnDType.foreign_key}
            and type1 == type2):
            return _connect_non_key_columns(dbb, helper, db_helper,
                                            table_1_name, table_1_col_name,
                                            table_2_name, table_2_col_name, type1)

        # Different types (non-key) - no operation
        return dbb

    finally:
        # FIXED: Always close connection to prevent leaks
        db_helper.close()


def _bridge_category_to_pk(dbb, helper: SchemaHelper, db_helper: DuckDBHelper,
                           src_table: str, src_col: str,
                           pk_table: str, pk_col: str,
                           bridge_name: Optional[str] = None):
    """
    Create a bridge table connecting a category/FK column to a PK WITHOUT demoting the PK.

    CRITICAL FIX: When FK values are not a subset of PK values, we cannot directly
    connect them. This function creates a bridge/dimension table that:
    1. Leaves the PK column UNCHANGED (preserves existing FK relationships)
    2. Only rewires the source column to point to the bridge
    3. Bridge has a nullable FK to the PK for matching values

    This prevents the critical bug where _connect_non_key_columns would convert
    the PK column to FK, breaking all existing references to it.

    Args:
        dbb: The database object
        helper: SchemaHelper instance
        db_helper: DuckDBHelper instance
        src_table: Source table name (has category/FK column)
        src_col: Source column name
        pk_table: Target table name (has PK column)
        pk_col: Target PK column name
        bridge_name: Optional custom name for bridge table

    Returns:
        dbb: Modified database object
    """
    # Get distinct source values
    src_vals = db_helper.get_unique_values(src_table, src_col)

    # Get PK domain
    pk_vals = db_helper.get_unique_values(pk_table, pk_col)
    pk_set = set(pk_vals.tolist())

    # Build bridge table
    bridge_name = bridge_name or f"lkp_{src_table}_{src_col}"
    bridge_pk = f"{bridge_name}ID"

    # Row layout: [bridge_id (PK), value (category), pk_ref (FK to pk_table.pk_col, nullable)]
    value_to_id = {}
    bridge_values = []
    bridge_pk_ref = []

    # FIXED: Use running length as ID, not enumerate index (nulls cause gaps)
    for v in src_vals:
        if _is_null(v):
            continue  # Skip null values
        new_id = len(bridge_values)  # Current length = next ID
        value_to_id[v] = new_id
        bridge_values.append(v)
        # If value exists in PK set, link it; otherwise NULL
        bridge_pk_ref.append(v if v in pk_set else None)

    # 1) Create bridge schema
    # FIXED: Avoid duplicate column names - use different names for value vs FK columns
    value_col_name = f"{src_col}_value" if src_col == pk_col else src_col
    fk_col_name = f"{pk_col}_fk" if src_col == pk_col else pk_col

    bridge_schema = DBBTableSchema(
        name=bridge_name,
        columns=[
            DBBColumnSchema(name=bridge_pk, dtype=DBBColumnDType.primary_key),
            DBBColumnSchema(name=value_col_name, dtype=DBBColumnDType.category_t),
            DBBColumnSchema(name=fk_col_name, dtype=DBBColumnDType.foreign_key,
                           link_to=f"{pk_table}.{pk_col}")
        ],
        format="parquet",
        source=f"data/{bridge_name}.parquet"
    )
    helper.add_table(bridge_schema)

    # 2) Create bridge data
    dbb.tables[bridge_name] = {
        bridge_pk: np.arange(len(bridge_values)),
        value_col_name: np.array(bridge_values, dtype=object),
        fk_col_name: np.array(bridge_pk_ref, dtype=object)  # nullable
    }

    # 3) Rewire ONLY the source column to FK -> bridge PK
    link_to = f"{bridge_name}.{bridge_pk}"
    helper.update_column_type(src_table, src_col, DBBColumnDType.foreign_key, link_to=link_to)

    # Map source data values to bridge IDs
    src_data = dbb.tables[src_table][src_col]

    def to_id(v):
        if _is_null(v):
            return -1
        return value_to_id.get(v, -1)

    dbb.tables[src_table][src_col] = np.array([to_id(v) for v in src_data], dtype=int)

    return dbb


def _connect_non_key_columns(dbb, helper: SchemaHelper, db_helper: DuckDBHelper,
                             table_1_name: str, table_1_col_name: str,
                             table_2_name: str, table_2_col_name: str, col_type: DBBColumnDType):
    """
    Helper function to connect two non-key columns using surrogate key approach.
    FIXED: Null-safe value mapping and dtype coercion.
    """
    # Get union of values (excludes NULLs)
    union_values = db_helper.get_union_values(table_1_name, table_1_col_name,
                                             table_2_name, table_2_col_name)

    # New surrogate table name and columns
    new_table_name = table_1_col_name[0].upper() + table_1_col_name[1:]
    new_col_name = table_1_col_name
    new_pk_name = f"{new_table_name}ID"

    # --- NEW: coerce dtype so we never hand a raw string into the schema ---
    try:
        coerced_col_type = _coerce_dtype(col_type)
    except Exception:
        # Fallback: treat unknowns as category-like text
        coerced_col_type = getattr(DBBColumnDType, "text", DBBColumnDType.category_t)

    value_to_id = {v: i for i, v in enumerate(union_values)}

    def safe_map(arr):
        result = []
        for v in arr:
            if _is_null(v):
                result.append(-1)
            else:
                result.append(value_to_id.get(v, -1))
        return np.asarray(result, dtype=int)

    link_to = f"{new_table_name}.{new_pk_name}"
    helper.update_column_type(table_1_name, table_1_col_name,
                              DBBColumnDType.foreign_key, link_to=link_to)
    helper.update_column_type(table_2_name, table_2_col_name,
                              DBBColumnDType.foreign_key, link_to=link_to)

    new_table = DBBTableSchema(
        name=new_table_name,
        columns=[
            DBBColumnSchema(name=new_pk_name, dtype=DBBColumnDType.primary_key),
            DBBColumnSchema(name=new_col_name, dtype=coerced_col_type)  # <-- coerced
        ],
        format='parquet',
        source=f'data/{new_table_name}.parquet'
    )
    helper.add_table(new_table)

    dbb.tables[new_table_name] = {
        new_pk_name: np.arange(len(union_values)),
        new_col_name: union_values
    }

    col1_data = dbb.tables[table_1_name][table_1_col_name]
    dbb.tables[table_1_name][table_1_col_name] = safe_map(col1_data)
    col2_data = dbb.tables[table_2_name][table_2_col_name]
    dbb.tables[table_2_name][table_2_col_name] = safe_map(col2_data)

    return dbb



def explode_multi_category_column(dbb, original_table: str, multi_cat_col: str,
                                  primary_key_column: str, new_table_name: str,
                                  new_col_name: str, dtype: Union[str, DBBColumnDType], **kwargs):
    """
    Explode a multi-category column (array/list) into a separate table.

    FIXED: When dtype='foreign_key', auto-create the target dimension table if missing,
    and map exploded labels to that table's IDs.
    """
    helper = SchemaHelper(dbb)

    if original_table not in dbb.tables:
        print(f"Table {original_table} not found")
        return dbb

    data = dbb.tables[original_table]
    if multi_cat_col not in data:
        print(f"Column {multi_cat_col} not found in table {original_table}")
        return dbb

    if not isinstance(data[multi_cat_col], np.ndarray):
        data[multi_cat_col] = np.array(data[multi_cat_col])

    if data[multi_cat_col].dtype != object:
        print(f"Column {multi_cat_col} is not object type, cannot explode")
        return dbb

    pk_col = helper.get_column(original_table, primary_key_column)
    if not pk_col:
        print(f"Primary key column {primary_key_column} not found")
        return dbb

    if pk_col.dtype == DBBColumnDType.foreign_key:
        pk_table, pk_column_name = pk_col.link_to.split('.')
    elif pk_col.dtype == DBBColumnDType.primary_key:
        pk_table = original_table
        pk_column_name = primary_key_column
    else:
        print(f"Invalid primary key column type: {pk_col.dtype}")
        return dbb

    # Remove column from original table schema
    helper.remove_column_from_table(original_table, multi_cat_col)

    is_fk = (dtype == DBBColumnDType.foreign_key) or (dtype == "foreign_key")

    # Explode the data (renaming to new_col_name here)
    df = pd.DataFrame({
        pk_column_name: data[primary_key_column],
        multi_cat_col: data[multi_cat_col]
    })
    exploded_df = df.explode(multi_cat_col).reset_index(drop=True)
    exploded_df = exploded_df.rename(columns={multi_cat_col: new_col_name})

    # Decide schema for the new exploded table
    if not is_fk:
        col_dt = _coerce_dtype(dtype)
        new_cols = [
            DBBColumnSchema(name=pk_column_name, dtype=DBBColumnDType.foreign_key,
                            link_to=f"{pk_table}.{pk_column_name}"),
            DBBColumnSchema(name=new_col_name, dtype=col_dt),
            DBBColumnSchema(name=f"{new_table_name}ID", dtype=DBBColumnDType.primary_key)
        ]
    else:
        # Infer target dimension table name if not provided
        target_table_name = kwargs.get("target_table") or kwargs.get("target_table_name")
        if not target_table_name:
            base = new_col_name[:-2] if new_col_name.lower().endswith("id") else new_col_name
            target_table_name = base[:1].upper() + base[1:]

        target = helper.get_table(target_table_name)
        # FIXED: Always use {TableName}ID for PK to avoid conflicts with value column
        target_pk_name = f"{target_table_name}ID"
        # Value column uses lowercase table name or original column name
        value_col_name = target_table_name[:1].lower() + target_table_name[1:] if target_table_name else new_col_name

        # Ensure PK and value column names are different
        if value_col_name == target_pk_name:
            value_col_name = f"{value_col_name}_value"

        # Create the target dimension if missing
        if target is None:
            # unique label values
            unique_vals = exploded_df[new_col_name].dropna().unique()
            target_schema = DBBTableSchema(
                name=target_table_name,
                columns=[
                    DBBColumnSchema(name=target_pk_name, dtype=DBBColumnDType.primary_key),
                    DBBColumnSchema(name=value_col_name, dtype=DBBColumnDType.category_t),
                ],
                format='parquet',
                source=f"data/{target_table_name}.parquet"
            )
            helper.add_table(target_schema)
            dbb.tables[target_table_name] = {
                target_pk_name: np.arange(len(unique_vals), dtype=int),
                value_col_name: np.array(unique_vals, dtype=object)
            }
        else:
            # discover pk + pick a value column to map labels
            pk_candidates = [c.name for c in target.columns if c.dtype == DBBColumnDType.primary_key]
            if target_pk_name not in pk_candidates and pk_candidates:
                target_pk_name = pk_candidates[0]
            # prefer a semantic/textual column
            prefer = [value_col_name] + [c.name for c in target.columns
                                         if c.name != target_pk_name and c.dtype in {
                                             getattr(DBBColumnDType, "category_t", None),
                                             getattr(DBBColumnDType, "text", None)}]
            tbl = dbb.tables[target_table_name]
            value_col_name = next((n for n in prefer if n in tbl), prefer[-1] if prefer else value_col_name)

        # Map exploded labels to IDs
        target_data = dbb.tables[target_table_name]
        target_ids = np.asarray(target_data[target_pk_name])
        id_set = set(target_ids.tolist())

        labels = exploded_df[new_col_name].to_numpy()

        # FIXED: Safely detect if labels are already numeric IDs vs raw strings
        def _is_numeric_id(v):
            """Check if value is a numeric ID that can be safely converted to int."""
            if _is_null(v):
                return False
            # Handle various numeric types
            if isinstance(v, (int, np.integer)):
                return True
            if isinstance(v, (float, np.floating)):
                # Check if it's a whole number (not NaN)
                return not np.isnan(v) and v == int(v)
            # For strings/other types, try conversion
            if isinstance(v, str):
                try:
                    float_v = float(v)
                    return float_v == int(float_v)
                except (ValueError, OverflowError):
                    return False
            return False

        # Check if all non-null values are numeric IDs that exist in target
        all_numeric_ids = all((_is_null(v) or (_is_numeric_id(v) and (int(float(v)) if isinstance(v, str) else int(v)) in id_set)) for v in labels)

        if all_numeric_ids and any(_is_numeric_id(v) for v in labels):
            # Labels are already IDs - convert safely
            def safe_int(v):
                if _is_null(v):
                    return -1
                try:
                    if isinstance(v, str):
                        return int(float(v))
                    return int(v)
                except (ValueError, OverflowError, TypeError):
                    return -1
            mapped = np.array([safe_int(v) for v in labels], dtype=int)
        else:
            # Labels are raw values - need to map via value column
            mapping = {
                lab: id_
                for lab, id_ in zip(np.asarray(target_data[value_col_name], dtype=object), target_ids)
                if not _is_null(lab)
            }
            mapped = np.array([(-1 if _is_null(v) else mapping.get(v, -1)) for v in labels], dtype=int)

        exploded_df[new_col_name] = mapped

        new_cols = [
            DBBColumnSchema(name=pk_column_name, dtype=DBBColumnDType.foreign_key,
                            link_to=f"{pk_table}.{pk_column_name}"),
            DBBColumnSchema(name=new_col_name, dtype=DBBColumnDType.foreign_key,
                            link_to=f"{target_table_name}.{target_pk_name}")
        ]

    # Create exploded table schema (no extra PK when FK branch)
    new_table = DBBTableSchema(
        name=new_table_name,
        columns=new_cols,
        format='parquet',
        source=f'data/{new_table_name.lower()}.parquet'
    )
    helper.add_table(new_table)

    # Populate exploded table
    if not is_fk:
        exploded_df[f"{new_table_name}ID"] = np.arange(len(exploded_df))
    del data[multi_cat_col]
    dbb.tables[original_table] = data
    dbb.tables[new_table_name] = {col: exploded_df[col].to_numpy() for col in exploded_df.columns}

    return dbb



def generate_non_dummy_table(dbb, base_table_name: str, cols: List[str],
                             new_table_name: str, **kwargs):
    """
    Extract columns from a table into a new normalized table.

    Creates a new table with unique combinations of the specified columns,
    adds a PK to it, and replaces the columns in the original table with
    a FK to the new table.

    Args:
        dbb: The database object
        base_table_name: Name of the source table
        cols: List of column names to extract
        new_table_name: Name for the new normalized table

    Returns:
        dbb: Modified database object
    """
    helper = SchemaHelper(dbb)
    db_helper = DuckDBHelper(dbb)

    try:
        if base_table_name not in dbb.tables:
            print(f"Table {base_table_name} not found")
            return dbb

        # Get column schemas for columns to extract
        col_schemas = []
        for col_name in cols:
            col = helper.get_column(base_table_name, col_name)
            if not col:
                print(f"Column {col_name} not found in {base_table_name}")
                return dbb

            if hasattr(col, 'link_to'):
                col_schemas.append(DBBColumnSchema(
                    name=col.name,
                    dtype=DBBColumnDType.foreign_key,
                    link_to=col.link_to
                ))
            else:
                col_schemas.append(DBBColumnSchema(
                    name=col.name,
                    dtype=col.dtype
                ))

        # Remove columns from base table and add FK to new table
        for col_name in cols:
            helper.remove_column_from_table(base_table_name, col_name)

        new_fk_name = f"{new_table_name}ID"
        helper.add_column_to_table(
            base_table_name,
            DBBColumnSchema(name=new_fk_name, dtype=DBBColumnDType.foreign_key,
                           link_to=f"{new_table_name}.{new_fk_name}")
        )

        # Create new table schema (FIXED: consistent extension)
        col_schemas.append(DBBColumnSchema(name=new_fk_name,
                                           dtype=DBBColumnDType.primary_key))

        new_table = DBBTableSchema(
            name=new_table_name,
            columns=col_schemas,
            format='parquet',
            source=f'data/{new_table_name}.parquet'
        )
        helper.add_table(new_table)

        # Normalize data using DuckDB (now handles NaN properly)
        unique_df, merged_df = db_helper.normalize_columns(base_table_name, cols)
        unique_df = unique_df.rename(columns={'new_id': new_fk_name})

        # FIXED: Ensure FK column is integer array without NaN
        # The updated normalize_columns uses sentinels to avoid NaN in merges
        base_data = dbb.tables[base_table_name]
        base_data = {k: v for k, v in base_data.items() if k not in cols}
        # Convert to Int64 (nullable integer), fill any remaining NaN with -1, then to int
        base_data[new_fk_name] = merged_df['new_id'].astype('Int64').fillna(-1).astype(int).to_numpy()
        dbb.tables[base_table_name] = base_data

        # Create new table data
        new_table_data = {col: unique_df[col].to_numpy() for col in unique_df.columns}
        dbb.tables[new_table_name] = new_table_data

        return dbb
    finally:
        db_helper.close()


# ============================================================================
# Utility Function (for compatibility with old code)
# ============================================================================

def turn_dbb_into_a_lookup_table(dbb):
    """Turns the dbb object into a lookup table for easier access.

    Args:
        dbb: The dbb object to be turned into a lookup table.

    Returns:
        A lookup table with the dbb object.
    """
    lookup_table = {}
    for table in dbb.metadata.tables:
        for column in table.columns:
            lookup_table[(table.name, column.name)] = column
    return lookup_table


# ============================================================================
# Public API
# ============================================================================

def get_autog_actions() -> Dict[str, Callable[..., Any]]:
    """Get dictionary of all available AutoG actions (FIXED: proper type hint)."""
    return {
        "generate_or_connect_dummy_table": generate_or_connect_dummy_table,
        "connect_two_columns": connect_two_columns,
        "explode_multi_category_column": explode_multi_category_column,
        "generate_non_dummy_table": generate_non_dummy_table,
        "remove_primary_key": remove_primary_key,
        "add_primary_key": add_primary_key
    }


def apply_simulated_actions(function_name: str, params: Dict[str, Any]):
    """Apply an action by name with given parameters."""
    actions = get_autog_actions()
    if function_name not in actions:
        raise ValueError(f"Function {function_name} not found. "
                        f"Available: {list(actions.keys())}")
    func = actions[function_name]
    return func(**params)


def pack_function_introduction_prompt(func: Callable[..., Any]) -> str:
    """Generate prompt text describing a function (FIXED: handle None docstring)."""
    doc_string = inspect.getdoc(func) or "(no docstring provided)"
    intro = f"Here is the introduction of the function {func.__name__}:\n"
    intro += doc_string
    return intro


class SimulatedActions:
    """Container class for AutoG actions."""

    def __init__(self):
        self.actions = get_autog_actions()

    def __getitem__(self, key: str):
        return self.actions[key]

    def __iter__(self):
        return iter(self.actions.items())

    def __len__(self):
        return len(self.actions)

    def __repr__(self):
        return f"SimulatedActions({list(self.actions.keys())})"
