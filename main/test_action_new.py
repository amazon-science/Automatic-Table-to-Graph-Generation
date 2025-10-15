"""
Comprehensive test suite for action_new.py

Tests all 6 AutoG actions with various scenarios including:
- Basic functionality
- Edge cases
- Data integrity
- Schema consistency
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict
from dbinfer_bench.dataset_meta import (
    DBBColumnSchema, DBBTableSchema, DBBRDBDatasetMeta,
    DBBColumnDType, DBBTableDataFormat
)


# Mock DBBRDBDataset class for testing
class MockDBBRDBDataset:
    """Mock database object for testing."""

    def __init__(self, metadata: Dict, tables: Dict):
        self.metadata = DBBRDBDatasetMeta.parse_obj(metadata)
        self.tables = tables


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_database():
    """Create a simple database with Users and Products tables."""
    metadata = {
        'dataset_name': 'test_db',
        'tables': [
            {
                'name': 'Users',
                'source': 'data/users.parquet',
                'format': 'parquet',
                'columns': [
                    {'name': 'userID', 'dtype': 'primary_key'},
                    {'name': 'name', 'dtype': 'text'},
                    {'name': 'age', 'dtype': 'float'},
                    {'name': 'city', 'dtype': 'category'}
                ]
            },
            {
                'name': 'Products',
                'source': 'data/products.parquet',
                'format': 'parquet',
                'columns': [
                    {'name': 'productID', 'dtype': 'primary_key'},
                    {'name': 'name', 'dtype': 'text'},
                    {'name': 'category', 'dtype': 'category'},
                    {'name': 'price', 'dtype': 'float'}
                ]
            }
        ],
        'tasks': [],
        'method': 'r2n'
    }

    tables = {
        'Users': {
            'userID': np.array([0, 1, 2, 3, 4]),
            'name': np.array(['Alice', 'Bob', 'Charlie', 'David', 'Eve']),
            'age': np.array([25.0, 30.0, 35.0, 28.0, 32.0]),
            'city': np.array(['NYC', 'LA', 'NYC', 'SF', 'LA'])
        },
        'Products': {
            'productID': np.array([0, 1, 2, 3]),
            'name': np.array(['Laptop', 'Phone', 'Tablet', 'Monitor']),
            'category': np.array(['Electronics', 'Electronics', 'Electronics', 'Electronics']),
            'price': np.array([1000.0, 800.0, 500.0, 300.0])
        }
    }

    return MockDBBRDBDataset(metadata, tables)


@pytest.fixture
def database_with_fk():
    """Create a database with foreign key relationships."""
    metadata = {
        'dataset_name': 'test_db_fk',
        'tables': [
            {
                'name': 'Users',
                'source': 'data/users.parquet',
                'format': 'parquet',
                'columns': [
                    {'name': 'userID', 'dtype': 'primary_key'},
                    {'name': 'name', 'dtype': 'text'}
                ]
            },
            {
                'name': 'Orders',
                'source': 'data/orders.parquet',
                'format': 'parquet',
                'columns': [
                    {'name': 'orderID', 'dtype': 'primary_key'},
                    {'name': 'userID', 'dtype': 'foreign_key', 'link_to': 'Users.userID'},
                    {'name': 'amount', 'dtype': 'float'}
                ]
            }
        ],
        'tasks': [],
        'method': 'r2n'
    }

    tables = {
        'Users': {
            'userID': np.array([0, 1, 2]),
            'name': np.array(['Alice', 'Bob', 'Charlie'])
        },
        'Orders': {
            'orderID': np.array([0, 1, 2, 3]),
            'userID': np.array([0, 0, 1, 2]),
            'amount': np.array([100.0, 150.0, 200.0, 75.0])
        }
    }

    return MockDBBRDBDataset(metadata, tables)


@pytest.fixture
def database_with_multi_category():
    """Create a database with multi-category columns."""
    metadata = {
        'dataset_name': 'test_db_multi',
        'tables': [
            {
                'name': 'Movies',
                'source': 'data/movies.parquet',
                'format': 'parquet',
                'columns': [
                    {'name': 'movieID', 'dtype': 'primary_key'},
                    {'name': 'title', 'dtype': 'text'},
                    {'name': 'genres', 'dtype': 'multi_category'}
                ]
            }
        ],
        'tasks': [],
        'method': 'r2n'
    }

    tables = {
        'Movies': {
            'movieID': np.array([0, 1, 2]),
            'title': np.array(['Movie A', 'Movie B', 'Movie C']),
            'genres': np.array([
                ['Action', 'Thriller'],
                ['Comedy', 'Romance'],
                ['Action', 'Sci-Fi', 'Adventure']
            ], dtype=object)
        }
    }

    return MockDBBRDBDataset(metadata, tables)


# ============================================================================
# Test remove_primary_key
# ============================================================================

def test_remove_primary_key_basic(simple_database):
    """Test basic primary key removal."""
    from models.autog.action_new import remove_primary_key

    dbb = simple_database
    initial_cols = len(dbb.metadata.tables[0].columns)
    initial_data_cols = len(dbb.tables['Users'])

    # Remove primary key
    dbb = remove_primary_key(dbb, 'Users', 'userID')

    # Check schema
    assert len(dbb.metadata.tables[0].columns) == initial_cols - 1
    assert 'userID' not in [col.name for col in dbb.metadata.tables[0].columns]

    # Check data
    assert len(dbb.tables['Users']) == initial_data_cols - 1
    assert 'userID' not in dbb.tables['Users']


def test_remove_primary_key_nonexistent(simple_database):
    """Test removing non-existent primary key."""
    from models.autog.action_new import remove_primary_key

    dbb = simple_database
    initial_cols = len(dbb.metadata.tables[0].columns)

    # Try to remove non-existent column
    dbb = remove_primary_key(dbb, 'Users', 'nonexistent')

    # Should remain unchanged
    assert len(dbb.metadata.tables[0].columns) == initial_cols


def test_remove_primary_key_wrong_type(simple_database):
    """Test removing column that's not a primary key."""
    from models.autog.action_new import remove_primary_key

    dbb = simple_database
    initial_cols = len(dbb.metadata.tables[0].columns)

    # Try to remove non-PK column
    dbb = remove_primary_key(dbb, 'Users', 'name')

    # Should remain unchanged
    assert len(dbb.metadata.tables[0].columns) == initial_cols


# ============================================================================
# Test add_primary_key
# ============================================================================

def test_add_primary_key_basic(simple_database):
    """Test adding primary key to table without one."""
    from models.autog.action_new import add_primary_key, remove_primary_key

    dbb = simple_database

    # First remove existing PK
    dbb = remove_primary_key(dbb, 'Users', 'userID')

    # Add new PK
    dbb = add_primary_key(dbb, 'Users', 'newID')

    # Check schema
    pk_cols = [col for col in dbb.metadata.tables[0].columns
               if col.dtype == DBBColumnDType.primary_key]
    assert len(pk_cols) == 1
    assert pk_cols[0].name == 'newID'

    # Check data
    assert 'newID' in dbb.tables['Users']
    assert len(dbb.tables['Users']['newID']) == 5
    assert np.array_equal(dbb.tables['Users']['newID'], np.array([0, 1, 2, 3, 4]))


def test_add_primary_key_already_exists(simple_database):
    """Test adding PK when one already exists."""
    from models.autog.action_new import add_primary_key

    dbb = simple_database
    initial_cols = len(dbb.metadata.tables[0].columns)

    # Try to add PK when one exists
    dbb = add_primary_key(dbb, 'Users', 'newID')

    # Should remain unchanged
    assert len(dbb.metadata.tables[0].columns) == initial_cols
    assert 'newID' not in dbb.tables['Users']


# ============================================================================
# Test generate_or_connect_dummy_table
# ============================================================================



# ============================================================================
# Test connect_two_columns
# ============================================================================

def test_connect_two_columns_category_to_pk(database_with_fk):
    """Test connecting category column to primary key."""
    from models.autog.action_new import connect_two_columns, add_primary_key

    dbb = database_with_fk

    # Add category column to Orders
    dbb.metadata.tables[1].columns.append(
        DBBColumnSchema(name='status', dtype=DBBColumnDType.category_t)
    )
    dbb.tables['Orders']['status'] = np.array([0, 1, 0, 1])

    # Add Status table
    dbb.metadata.tables.append(DBBTableSchema(
        name='Status',
        source='data/status.parquet',
        format=DBBTableDataFormat.PARQUET,
        columns=[
            DBBColumnSchema(name='statusID', dtype=DBBColumnDType.primary_key),
            DBBColumnSchema(name='name', dtype=DBBColumnDType.text_t)
        ]
    ))
    dbb.tables['Status'] = {
        'statusID': np.array([0, 1]),
        'name': np.array(['pending', 'completed'])
    }

    # Connect status to Status.statusID
    dbb = connect_two_columns(dbb, 'Orders', 'status', 'Status', 'statusID')

    # Check schema
    status_col = None
    for col in dbb.metadata.tables[1].columns:
        if col.name == 'status':
            status_col = col
            break

    assert status_col is not None
    assert status_col.dtype == DBBColumnDType.foreign_key
    assert status_col.link_to == 'Status.statusID'


def test_connect_two_columns_same_fk(database_with_fk):
    """Test connecting two columns already pointing to same FK."""
    from models.autog.action_new import connect_two_columns

    dbb = database_with_fk

    # Add another FK to Users in Orders table
    dbb.metadata.tables[1].columns.append(
        DBBColumnSchema(name='assignedTo', dtype=DBBColumnDType.foreign_key,
                       link_to='Users.userID')
    )
    dbb.tables['Orders']['assignedTo'] = np.array([1, 2, 0, 1])

    initial_link = dbb.metadata.tables[1].columns[-1].link_to

    # Connect - should be no-op
    dbb = connect_two_columns(dbb, 'Orders', 'userID', 'Orders', 'assignedTo')

    # Should remain unchanged
    assert dbb.metadata.tables[1].columns[-1].link_to == initial_link


def test_connect_two_columns_different_types(simple_database):
    """Test connecting columns with different types."""
    from models.autog.action_new import connect_two_columns

    dbb = simple_database
    initial_age_type = None
    for col in dbb.metadata.tables[0].columns:
        if col.name == 'age':
            initial_age_type = col.dtype
            break

    # Try to connect float to category
    dbb = connect_two_columns(dbb, 'Users', 'age', 'Users', 'city')

    # Should remain unchanged
    for col in dbb.metadata.tables[0].columns:
        if col.name == 'age':
            assert col.dtype == initial_age_type


def test_connect_non_key_columns(simple_database):
    """Test connecting two non-key columns of same type."""
    from models.autog.action_new import connect_two_columns

    dbb = simple_database

    # Add another text column to Products
    dbb.metadata.tables[1].columns.append(
        DBBColumnSchema(name='brand', dtype=DBBColumnDType.text_t)
    )
    dbb.tables['Products']['brand'] = np.array(['BrandA', 'BrandB', 'BrandA', 'BrandC'])

    # Connect name columns (both text)
    dbb = connect_two_columns(dbb, 'Users', 'name', 'Products', 'name')

    # Should create new table
    table_names = [t.name for t in dbb.metadata.tables]
    assert 'Name' in table_names

    # Check that columns are now FKs
    users_name_col = None
    for col in dbb.metadata.tables[0].columns:
        if col.name == 'name':
            users_name_col = col
            break

    assert users_name_col.dtype == DBBColumnDType.foreign_key
    assert 'Name.NameID' in users_name_col.link_to


# ============================================================================
# Test explode_multi_category_column
# ============================================================================

def test_explode_multi_category_basic(database_with_multi_category):
    """Test basic multi-category column explosion."""
    from models.autog.action_new import explode_multi_category_column

    dbb = database_with_multi_category

    # Explode genres
    dbb = explode_multi_category_column(
        dbb, 'Movies', 'genres', 'movieID', 'MovieGenres', 'genre', 'category'
    )

    # Check schema - genres should be removed from Movies
    movies_cols = [col.name for col in dbb.metadata.tables[0].columns]
    assert 'genres' not in movies_cols

    # Check new table exists
    table_names = [t.name for t in dbb.metadata.tables]
    assert 'MovieGenres' in table_names

    # Check new table data
    assert 'MovieGenres' in dbb.tables
    assert 'movieID' in dbb.tables['MovieGenres']
    assert 'genre' in dbb.tables['MovieGenres']
    assert 'MovieGenresID' in dbb.tables['MovieGenres']

    # Should have 7 rows (2 + 2 + 3)
    assert len(dbb.tables['MovieGenres']['movieID']) == 7

    # Check data integrity - should have correct mappings
    movie_genres = dbb.tables['MovieGenres']
    assert np.sum(movie_genres['movieID'] == 0) == 2  # Movie 0 has 2 genres
    assert np.sum(movie_genres['movieID'] == 1) == 2  # Movie 1 has 2 genres
    assert np.sum(movie_genres['movieID'] == 2) == 3  # Movie 2 has 3 genres


def test_explode_multi_category_as_fk(database_with_multi_category):
    """Test exploding multi-category as foreign keys only."""
    from models.autog.action_new import explode_multi_category_column

    dbb = database_with_multi_category

    # Explode as FK
    dbb = explode_multi_category_column(
        dbb, 'Movies', 'genres', 'movieID', 'MovieGenres', 'genreID', 'foreign_key'
    )

    # Check schema
    movie_genres_table = None
    for table in dbb.metadata.tables:
        if table.name == 'MovieGenres':
            movie_genres_table = table
            break

    assert movie_genres_table is not None
    col_names = [col.name for col in movie_genres_table.columns]
    assert 'MovieGenresID' not in col_names  # No PK when dtype is FK
    assert 'movieID' in col_names
    assert 'genreID' in col_names


def test_explode_non_object_column(simple_database):
    """Test exploding non-object column (should fail gracefully)."""
    from models.autog.action_new import explode_multi_category_column

    dbb = simple_database
    initial_table_count = len(dbb.metadata.tables)

    # Try to explode float column
    dbb = explode_multi_category_column(
        dbb, 'Users', 'age', 'userID', 'Ages', 'value', 'float'
    )

    # Should not create new table
    assert len(dbb.metadata.tables) == initial_table_count


# ============================================================================
# Test generate_non_dummy_table
# ============================================================================

def test_generate_non_dummy_table_basic(simple_database):
    """Test extracting columns into new table."""
    from models.autog.action_new import generate_non_dummy_table

    dbb = simple_database

    # Extract name and city into new table
    dbb = generate_non_dummy_table(dbb, 'Users', ['name', 'city'], 'UserProfile')

    # Check schema - columns should be removed from Users
    users_cols = [col.name for col in dbb.metadata.tables[0].columns]
    assert 'name' not in users_cols
    assert 'city' not in users_cols
    assert 'UserProfileID' in users_cols

    # Check new table exists
    table_names = [t.name for t in dbb.metadata.tables]
    assert 'UserProfile' in table_names

    # Check new table schema
    profile_table = None
    for table in dbb.metadata.tables:
        if table.name == 'UserProfile':
            profile_table = table
            break

    profile_cols = [col.name for col in profile_table.columns]
    assert 'name' in profile_cols
    assert 'city' in profile_cols
    assert 'UserProfileID' in profile_cols

    # Check data
    assert 'UserProfile' in dbb.tables
    assert 'name' in dbb.tables['UserProfile']
    assert 'city' in dbb.tables['UserProfile']
    assert 'UserProfileID' in dbb.tables['UserProfile']

    # Check deduplication - should have fewer rows than original
    # Original: 5 users, but with duplicates in name+city combinations
    assert len(dbb.tables['UserProfile']['UserProfileID']) <= 5

    # Check FK in Users table
    assert 'UserProfileID' in dbb.tables['Users']
    assert len(dbb.tables['Users']['UserProfileID']) == 5


def test_generate_non_dummy_table_with_fk(database_with_fk):
    """Test extracting column that is already a foreign key."""
    from models.autog.action_new import generate_non_dummy_table

    dbb = database_with_fk

    # Add another column to Orders
    dbb.metadata.tables[1].columns.append(
        DBBColumnSchema(name='status', dtype=DBBColumnDType.category_t)
    )
    dbb.tables['Orders']['status'] = np.array(['pending', 'completed', 'pending', 'shipped'])

    # Extract userID and status
    dbb = generate_non_dummy_table(dbb, 'Orders', ['userID', 'status'], 'OrderInfo')

    # Check that UserID in OrderInfo is still a foreign key
    order_info_table = None
    for table in dbb.metadata.tables:
        if table.name == 'OrderInfo':
            order_info_table = table
            break

    userid_col = None
    for col in order_info_table.columns:
        if col.name == 'userID':
            userid_col = col
            break

    assert userid_col is not None
    assert userid_col.dtype == DBBColumnDType.foreign_key
    assert userid_col.link_to == 'Users.userID'


def test_generate_non_dummy_table_nonexistent_columns(simple_database):
    """Test extracting non-existent columns."""
    from models.autog.action_new import generate_non_dummy_table

    dbb = simple_database
    initial_table_count = len(dbb.metadata.tables)

    # Try to extract non-existent column
    dbb = generate_non_dummy_table(dbb, 'Users', ['nonexistent'], 'NewTable')

    # Should not create new table
    assert len(dbb.metadata.tables) == initial_table_count




def test_data_integrity_after_actions(simple_database):
    """Test that data integrity is maintained after actions."""
    from models.autog.action_new import generate_non_dummy_table

    dbb = simple_database

    # Get initial row count
    initial_user_count = len(dbb.tables['Users']['userID'])

    # Extract columns
    dbb = generate_non_dummy_table(dbb, 'Users', ['name', 'city'], 'NameCity')

    # Verify row count unchanged in Users
    assert len(dbb.tables['Users']['userID']) == initial_user_count

    # Verify FK relationships are valid
    profile_ids = dbb.tables['Users']['NameCityID']
    unique_profile_ids = np.unique(profile_ids)

    # All IDs should be in the range of the new table
    assert np.all(profile_ids >= 0)
    assert np.all(profile_ids < len(dbb.tables['NameCity']['NameCityID']))


def test_schema_consistency_after_actions(database_with_fk):
    """Test that schema remains consistent after actions."""
    from models.autog.action_new import connect_two_columns, add_primary_key

    dbb = database_with_fk

    # Perform actions
    dbb = connect_two_columns(dbb, 'Orders', 'userID', 'Users', 'userID')

    # Verify all tables have proper structure
    for table in dbb.metadata.tables:
        assert table.name
        assert table.source
        assert table.format
        assert len(table.columns) > 0

        # Verify all columns have names and types
        for col in table.columns:
            assert col.name
            assert col.dtype

            # Verify FK columns have link_to
            if col.dtype == DBBColumnDType.foreign_key:
                assert hasattr(col, 'link_to')
                assert '.' in col.link_to


# ============================================================================
# API Tests
# ============================================================================

def test_get_autog_actions():
    """Test that all actions are available."""
    from models.autog.action_new import get_autog_actions

    actions = get_autog_actions()

    expected_actions = [
        'remove_primary_key',
        'add_primary_key',
        'generate_or_connect_dummy_table',
        'connect_two_columns',
        'explode_multi_category_column',
        'generate_non_dummy_table'
    ]

    for action in expected_actions:
        assert action in actions
        assert callable(actions[action])


def test_apply_simulated_actions(simple_database):
    """Test applying actions via the apply_simulated_actions function."""
    from models.autog.action_new import apply_simulated_actions

    dbb = simple_database

    # Apply action via string name
    result = apply_simulated_actions('add_primary_key', {
        'dbb': dbb,
        'base_table_name': 'Products',
        'col_name': 'testID'
    })

    # Should fail since Products already has a PK
    assert 'testID' not in result.tables['Products']


def test_simulated_actions_class():
    """Test SimulatedActions container class."""
    from models.autog.action_new import SimulatedActions

    actions = SimulatedActions()

    # Test __len__
    assert len(actions) == 6

    # Test __getitem__
    assert callable(actions['add_primary_key'])

    # Test __iter__
    action_names = [name for name, _ in actions]
    assert 'remove_primary_key' in action_names

    # Test __repr__
    repr_str = repr(actions)
    assert 'SimulatedActions' in repr_str


def test_pack_function_introduction_prompt():
    """Test prompt generation for functions."""
    from models.autog.action_new import pack_function_introduction_prompt, add_primary_key

    prompt = pack_function_introduction_prompt(add_primary_key)

    assert 'add_primary_key' in prompt
    assert 'primary key' in prompt.lower()
    assert len(prompt) > 50  # Should have substantial documentation


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_table_operations(simple_database):
    """Test operations on tables with no data."""
    from models.autog.action_new import add_primary_key

    dbb = simple_database

    # Create empty table
    dbb.metadata.tables.append(DBBTableSchema(
        name='EmptyTable',
        source='data/empty.parquet',
        format=DBBTableDataFormat.PARQUET,
        columns=[
            DBBColumnSchema(name='col1', dtype=DBBColumnDType.text_t)
        ]
    ))
    dbb.tables['EmptyTable'] = {'col1': np.array([])}

    # Add PK to empty table
    dbb = add_primary_key(dbb, 'EmptyTable', 'id')

    # Should work
    assert 'id' in dbb.tables['EmptyTable']
    assert len(dbb.tables['EmptyTable']['id']) == 0


def test_invalid_action_name():
    """Test calling non-existent action."""
    from models.autog.action_new import apply_simulated_actions

    with pytest.raises(ValueError, match="not found"):
        apply_simulated_actions('nonexistent_action', {})


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
