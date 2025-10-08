"""
    Update in the camera ready version:
    Instead of loading the file from the disk, we directly pass dbb as a parameter and change everything
    there, it will be much faster since the number of disk read/write will be reduced.
    After this change, we don't need to hard_code function as string but can directly call them. 
"""
import inspect
import numpy as np
import pandas as pd
from dbinfer_bench.dataset_meta import DBBColumnSchema, DBBTableSchema

def apply_simulated_actions(function_name, params):
    actions = SimulatedActions()
    if function_name in actions:
        func = actions[function_name]
        return func(**params)
    else:
        raise ValueError(f"Function {function_name} not found.")

class SimulatedActions():
    def __init__(self):
        self.actions = get_autog_actions()

    def __getitem__(self, key):
        return self.actions[key]

    def __iter__(self):
        return iter(self.actions.items())

    def __len__(self):
        return len(self.actions)

    def __repr__(self):
        return f"SimulatedActions({self.actions})"


def get_autog_actions():
    G_DUMMY_T = generate_or_connect_dummy_table
    C_TWO_C = connect_two_columns
    E_MULTI_C = explode_multi_category_column
    G_NON_DUMMY_T = generate_non_dummy_table
    RM_PK = remove_primary_key
    ADD_PK = add_primary_key

    actions = {
        "explode_multi_category_column": E_MULTI_C,
        "generate_non_dummy_table": G_NON_DUMMY_T, 
        "remove_primary_key": RM_PK,
        "add_primary_key": ADD_PK,
        "generate_or_connect_dummy_table": G_DUMMY_T,
        "connect_two_columns": C_TWO_C
    }
    return actions


def pack_function_introduction_prompt(func):
    doc_string = inspect.getdoc(func)
    intro = f"Here is the introduction of the function {func.__name__}:\n"
    intro += doc_string
    # intro += "The function has the following parameters:\n"
    # for param in parameters:
    #     intro += f"{param}: {parameters[param]}\n"
    return intro 

## here we generate code, the first return for schema, the second return for data

def remove_primary_key(dbb, base_table_name, col_name):
    """
        Description:
        Remove a primary key constraint from a column in the original table
        If the column is just an index, then the column will be removed from the table.
        For example, if the schema is like {
            {"name": "id", "dtype": "primary_key"},
            {"name": "user", "dtype": "foreign_key", "link_to": "user.userID"},
            {"name": "book", "dtype": "foreign_key", "link_to": "book.bookID"},
        }
        In such case, it's clear that this table represents the role of an edge, while the presence of primary key prevents heuristic to turn this table into an edge. Primary key is not needed in this case.
        In such case, we will remove the primary key constraint from the column.
        Parameters:
        dbb: the database object
        base_table_name: the name of the original table
        col_name: the name of the column in the original table
    """
    tables=dbb.metadata.tables
    del_col_idx = 0
    del_table_idx = 0
    for i, table in enumerate(tables):
        if table.name == f'{base_table_name}':
            for j, column in enumerate(table.columns):
                if column.name == f'{col_name}':
                    del_col_idx = j
            del_table_idx = i
    tables[del_table_idx].columns.pop(del_col_idx)
    dbb.metadata.tables = tables
    return dbb

def add_primary_key(dbb, base_table_name, col_name):
    """
        Description:
        Add a primary key column to the original table
        Parameters:
        dbb: the database object
        base_table_name: the name of the original table
        col_name: the name of the newly added primary key column
    """

    tables=dbb.metadata.tables
    for table in tables:
        if table.name == base_table_name:
            for column in table.columns:
                if column.dtype == 'primary_key':
                    print("Already has a primary key, will not add a new one")
                    return dbb
            table.columns.append(DBBColumnSchema(name=col_name, dtype='primary_key'))
            break
    dbb.metadata.tables = tables  

    data = dbb.tables[base_table_name]
    first_key = list(data.keys())[0]
    ndata = len(data[first_key])
    new_column = np.arange(ndata)
    data[f'{col_name}'] = new_column
    dbb.tables[base_table_name] = data
    return dbb


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

def generate_or_connect_dummy_table(dbb, base_table_name, orig_col_name, new_table_name, new_col_name, **kwargs):
    """
        Description:
        This function can be used in two ways:
        1. Generate a dummy table with only one primary key
        2. Turn an existing column with categorical type to an existing dummy table
        "orig_col_name" must be a column with category type
        Parameters:
        dbb: the database object
        base_table_name: the name of the original table
        orig_col_name: the name of the original column in the original table, this should be a column with category type
        new_table_name: the name of the new table to be created/connected
        new_col_name: the name of the new column to be created/connected
    """
    tables = dbb.metadata.tables
    for i, table in enumerate(tables):
        if table.name == f'{base_table_name}':
            for j, column in enumerate(table.columns):
                if column.name == f'{orig_col_name}':
                    table.columns[j].dtype = 'foreign_key'
                    table.columns[j].link_to = f'{new_table_name}.{new_col_name}'
                    break
            tables[i] = table
            break
    dbb.metadata.tables = tables
    return dbb

def number_of_pks(dbb):
    """
        Description:
        Count the number of primary keys in the database
        Parameters:
        dbb: the database object
    """
    number_of_pks = 0
    for table in dbb.metadata.tables:
        for column in table.columns:
            if column.dtype == 'primary_key':
                number_of_pks += 1
    return number_of_pks

def connect_two_columns_with_non_key_type(dbb, table_1_name, table_1_col_name, table_2_name, table_2_col_name, col_type, **kwargs):
    """
        Description:
        A helper function to connect two columns with non-key type
        Parameters:
        dbb: the database object
        table_1_name: the name of the first table
        table_1_col_name: the name of the column in the first table
        table_2_name: the name of the second table
        table_2_col_name: the name of the column in the second table
    """
    tables = dbb.metadata.tables
    ## dtype of the two columns
    
    ## get the value of the first column
    col_1_data = dbb.tables[table_1_name][table_1_col_name]
    col_2_data = dbb.tables[table_2_name][table_2_col_name]
    ## get the unique values of the first column
    unique_col_1_data = np.unique(col_1_data)
    ## get the unique values of the second column
    unique_col_2_data = np.unique(col_2_data)
    ## get the union of the two columns (the value of the new column)
    union_col_data = np.union1d(unique_col_1_data, unique_col_2_data)
    ## create a new table (name is the upper of col_name)
    new_table_name = table_1_col_name[0].upper() + table_1_col_name[1:]
    ## new_col_name 
    new_col_name = table_1_col_name
    new_table_pk_col_name = f'{new_table_name}ID'
    ## change the dtype of two columns to be a foreign key to the primary key
    ## of the new table
    for table in tables:
        if table.name == f'{table_1_name}' or table.name == f'{table_2_name}':
            for column in table.columns:
                if column.name == f'{table_1_col_name}' or column.name == f'{table_2_col_name}':
                    column.dtype = 'foreign_key'
                    column.link_to = f'{new_table_name}.{new_table_pk_col_name}'
    ## create a new table
    new_table = DBBTableSchema(
        name=new_table_name,
        columns=[
            DBBColumnSchema(name=new_table_pk_col_name, dtype='primary_key'),
            DBBColumnSchema(name=new_col_name, dtype=col_type)
        ],
        format = 'parquet', 
        source = f'data/{new_table_name}.parquet'
    ) 
    dbb.metadata.tables.append(new_table)
    ## add the data to the new table
    new_table_data = {new_table_pk_col_name: np.arange(len(union_col_data)), new_col_name: union_col_data}
    dbb.tables[new_table_name] = new_table_data
    ## change the value of two old tables, need to change them into corresponding foreign key
    ## foreign key of table 1
    value_to_key_mapping = {v: k for k, v in enumerate(union_col_data)}
    ## change the value of table 1
    dbb.tables[table_1_name][table_1_col_name] = [value_to_key_mapping[v] for v in col_1_data]
    ## change the value of table 2
    dbb.tables[table_2_name][table_2_col_name] = [value_to_key_mapping[v] for v in col_2_data]
    return dbb

    
    




def connect_two_columns(dbb, table_1_name, table_1_col_name, table_2_name, table_2_col_name, **kwargs):
    """
        Description:
        Connect two columns, this function can be used for the following case. Always put the column with category type in table 1.
        1. A category column in table 1 is connected to a category column in table 2, in this case, a new dummy table will be created
        2. A category column in table 1 is connected to a primary key column in table 2, in this case, the column in table 1 will be turned into a foreign key column. In case 2, table_2_col_name must be a primary key column
        3. A category column in table 1 is connected to a non-category and non-primary key column in table 2, in this case, we will use a trick called Surrogate Key. 
        4. If the column in table 1 is already a foreign key, then in this case it's probably a multi-column-point-to-one case, we need to update other fk columns too.
        Parameters:
        dbb: the database object
        table_1_name: the name of the first table, 
        table_1_col_name: the name of the column in the first table, this should be a column with category type
        table_2_name: the name of the second table
        table_2_col_name: the name of the column in the second table, this should be a column with category type
    """
    tables=dbb.metadata.tables

    

    ## find the primary key for table 2     
    pk = [column for table in tables if table.name == f'{table_2_name}' for column in table.columns if column.dtype == 'primary_key']
    metatype_dict = {f'{table.name}.{column.name}': column.dtype for table in tables for column in table.columns}
    columninfo_dict = {f'{table.name}.{column.name}': column for table in tables for column in table.columns}
    if f'{table_1_name}.{table_1_col_name}' not in metatype_dict or f'{table_2_name}.{table_2_col_name}' not in metatype_dict:
        # import ipdb; ipdb.set_trace()
        return generate_or_connect_dummy_table(dbb, table_1_name, table_1_col_name, table_2_name, table_2_col_name)
    type_of_col1 = metatype_dict[f'{table_1_name}.{table_1_col_name}']
    type_of_col2 = metatype_dict[f'{table_2_name}.{table_2_col_name}']
    ## trivial case, both fk and point to the same, directly return
    # import ipdb; ipdb.set_trace()
    if type_of_col1 == 'foreign_key' and type_of_col2 == 'foreign_key' and columninfo_dict[f'{table_1_name}.{table_1_col_name}'].link_to == columninfo_dict[f'{table_2_name}.{table_2_col_name}'].link_to:
        return dbb
    if type_of_col1 != type_of_col2:
        return dbb 
    
    ## two non key types with the same type 
    if type_of_col1 not in ['category', 'primary_key', 'foreign_key'] and type_of_col2 not in ['category', 'primary_key', 'foreign_key']:
        return connect_two_columns_with_non_key_type(dbb, table_1_name, table_1_col_name, table_2_name, table_2_col_name, col_type=type_of_col1)

    # add_fk = False
    ## ensure that table 2 is the target 
    if type_of_col1 == 'foreign_key' and type_of_col2 == 'category':
        table_1_name, table_2_name = table_2_name, table_1_name
        table_1_col_name, table_2_col_name = table_2_col_name, table_1_col_name
        type_of_col1, type_of_col2 = type_of_col2, type_of_col1
    if type_of_col1 == 'primary_key':
        table_1_name, table_2_name = table_2_name, table_1_name
        table_1_col_name, table_2_col_name = table_2_col_name, table_1_col_name
        type_of_col1, type_of_col2 = type_of_col2, type_of_col1
    ## case 3, both foreign key, but one link_to primary key
    if type_of_col1 == 'foreign_key' and type_of_col2 == 'foreign_key' and columninfo_dict[columninfo_dict[f'{table_1_name}.{table_1_col_name}'].link_to].dtype == 'primary_key':
        table_1_name, table_2_name = table_2_name, table_1_name
        table_1_col_name, table_2_col_name = table_2_col_name, table_1_col_name
        type_of_col1, type_of_col2 = type_of_col2, type_of_col1
    ## a special case, when there's a primary key while it can not cover the ids in another column, create a new table
    if type_of_col1 == 'foreign_key':
        update_fk = True
    else:
        update_fk = False
    ## determine new table and new column name
    if type_of_col1 == 'category' and type_of_col2 == 'category':
        new_table_name = f'{table_1_name}_{table_2_name}'
        new_table_col_name = f'{table_2_col_name}'
    elif type_of_col2 == 'primary_key':
        pk_data_range = dbb.tables[table_2_name][table_2_col_name]
        pk_data_range = np.unique(pk_data_range)
        fk_data_range = dbb.tables[table_1_name][table_1_col_name]
        fk_data_range = np.unique(fk_data_range)
        fk_data_range = fk_data_range[~np.isnan(fk_data_range)].astype(pk_data_range.dtype)
        n_pk = number_of_pks(dbb)
        if not np.all(np.isin(fk_data_range, pk_data_range)) and n_pk > 1 and dbb.metadata.dataset_name != 'diginetica':
            ## we need to create a new table
            ## this is a special case
            ## the latter condition is to say that we must ensure there are still primary keys in the table
            ## diginetica has some bugs, the fk and pk doesn't follow strict constraints, here is a work around to not update the pk to fk.
            new_table_name = f'{table_1_name}_{table_2_name}'
            new_table_col_name = f'{table_2_col_name}'
        else:
            new_table_name = ""
            new_table_col_name = ""
    else:
        new_table_name = ""
        new_table_col_name = ""
    old_link_to = ""
    new_link_to = ""
    direct_connect = False
    save_table_i = -1
    save_column_j = -1
    # import ipdb; ipdb.set_trace()
    for i, table in enumerate(tables):
        if table.name == f'{table_1_name}':
            for j, column in enumerate(table.columns):
                if column.name == f'{table_1_col_name}':
                    # if column.dtype != 'category': raise ValueError("The column must be a category type")
                    table.columns[j].dtype = 'foreign_key'
                    if new_table_name != "":
                        table.columns[j].link_to = f'{new_table_name}.{new_table_col_name}'
                        new_link_to = f'{new_table_name}.{new_table_col_name}'
                    else:
                        ## we don't create new dummay table, two possibilities
                        ## 1. update the fk key
                        ## 2. connect column 1 to column 2 (cat to fk, cat to pk)
                        if update_fk:
                            old_link_to = column.link_to
                        else:
                            if type_of_col2 == 'primary_key':
                                table.columns[j].link_to = f'{table_2_name}.{table_2_col_name}'
                                if not update_fk:
                                    direct_connect = True
                            elif type_of_col2 == 'foreign_key':
                                save_table_i = i
                                save_column_j = j
                            # new_link_to = f'{table_2_name}.{table_2_col_name}'
                    break
            tables[i] = table
    if direct_connect:
        dbb.metadata.tables = tables
        return dbb
    for i, table in enumerate(tables):
        if table.name == f'{table_2_name}':
            add_pk = False
            for j, column in enumerate(table.columns):
                if column.name == f'{table_2_col_name}':
                    # if column.dtype != 'category' and new_table_name != '': raise ValueError("The column must be a category type")
                    if column.dtype == 'category' and new_table_name == '': raise ValueError("In this case you must provide a new table name")
                    if new_table_name != '':
                        table.columns[j].dtype = 'foreign_key'
                        table.columns[j].link_to = f'{new_table_name}.{new_table_col_name}'
                    
                    if save_column_j >= 0:
                        ## must be foreign key
                        tables[save_table_i].columns[save_column_j].link_to = column.link_to

                    if update_fk:
                        if column.dtype == 'primary_key':
                            new_link_to = f'{table_2_name}.{table_2_col_name}'
                        else:
                            new_link_to = column.link_to
                    # elif column.dtype != 'category' and column.dtype != 'foreign_key' and column.dtype != 'primary_key' and new_table_name == '':
                    #     ## Surrogate Key
                    #     if len(pk) == 0:
                    #         ## add a primary key for table 2 
                    #         add_pk = True
                    #     add_fk = True
                    # else:
                    break
            # if add_pk:
            #     table.columns.append(DBBColumnSchema(name=f'{table_2_col_name}ID', dtype='primary_key'))
            #     ## open table 2
            #     data = dbb.tables[table_2_name]
            #     data[f'{table_2_col_name}ID'] = np.arange(len(data[f'{table_2_col_name}']))
            # import ipdb; ipdb.set_trace()
            tables[i] = table
    ## update fk in multi fk case
    # import ipdb; ipdb.set_trace()
    if update_fk:
        for i, table in enumerate(tables):
            for j, column in enumerate(table.columns):
                if column.dtype == 'foreign_key' and column.link_to == old_link_to:
                    table.columns[j].link_to = new_link_to
        dbb.metadata.tables = tables
        return dbb
    # if add_fk:
    #     ## mapping between table_2_col_name and pk of table 2
    #     table2_pk_name = pk[0]['name'] if len(pk) > 0 else f"{table_2_col_name}ID"
    #     ## change the schema
    #     for i, table in enumerate(tables):
    #         if table.name == f'{table_1_name}':
    #             for j, column in enumerate(table.columns):
    #                 if column.name == f'{table_1_col_name}':
    #                     table.columns[j].dtype = 'foreign_key'
    #                     table.columns[j].link_to = f'{table_2_name}.{table2_pk_name}'
    #                     break
    #     data = dbb.tables[table_2_name]
    #     ## mapping between table_2_col_name and pk of table 2
    #     table2_pk_name = pk[0]['name'] if len(pk) > 0 else f"{table_2_col_name}ID"
    #     table2_pk = data[table2_pk_name]
    #     table2_col = data[f'{table_2_col_name}']
    #     mapping = {k:v for k, v in zip(table2_col, table2_pk)}
    #     ## map into table 1
    #     data = dbb.tables[table_1_name]
    #     data[f'{table_1_col_name}'] = [mapping[val] for val in data[f'{table_1_col_name}']]
    #     dbb.tables[table_1_name] = data
    return dbb

def explode_multi_category_column(dbb, original_table, multi_cat_col, primary_key_column, new_table_name, new_col_name, dtype, **kwargs):
    """
        Description:
        Explode a multi-category column into multiple columns. You should determine whether to use this function. If you don't explode a multi-category column, it will be treated as a single category column automatically.
        Parameters:
        dbb: the database object
        original_table: name of the original table where the multi-category column is located
        multi_cat_col: the name of the multi-category column
        primary_key_column: the name of the primary key column in the original table
        new_table_name: the name of the new table to be created
        new_col_name: the name of the new column to be created
        dtype: the data type of the new column, if set to "foreign_key", this table will contain only "foreign_keys". In this case, it means you only want to use this column's relaion. If set to other types, this table will contain the original column's values, and a primary key will be added, this means you want to use this column's values.
    """
    data = dbb.tables[original_table]
    tables = dbb.metadata.tables
    columninfo_dict = {f'{table.name}.{column.name}': column for table in tables for column in table.columns}
    ## if primary key column is not a real primary, find primary key
    if columninfo_dict[f'{original_table}.{primary_key_column}'].dtype != 'primary_key':
        ## foreign key 
        if columninfo_dict[f'{original_table}.{primary_key_column}'].dtype == 'foreign_key':
            primary_key_table, primary_key_column = columninfo_dict[f'{original_table}.{primary_key_column}'].link_to.split('.')
        else:
            print("Not valid explode parameters, quit")
            return dbb
    else:
        primary_key_table = original_table
    rel_cols = {k:v for k, v in data.items() if k in [f'{primary_key_column}', f'{multi_cat_col}']}
    df = pd.DataFrame(rel_cols)
    if df[f'{multi_cat_col}'].dtype != 'object':
        print(f"Warning: The column {multi_cat_col} is not of type object, will halt")
        return dbb
    if dtype != "foreign_key":
        tables = dbb.metadata.tables
        del_col_idx = 0
        del_table_idx = 0
        for i, table in enumerate(tables):
            if table.name == f'{original_table}':
                for j, column in enumerate(table.columns):
                    if column.name == f'{multi_cat_col}':
                        del_col_idx = j
                del_table_idx = i
        tables[del_table_idx].columns.pop(del_col_idx)
        ## use dbbtables schema and dbbcolumns schema
        c_new_table_name = new_table_name.lower()
        new_table = DBBTableSchema(
            name=new_table_name,
            columns=[
                DBBColumnSchema(name=primary_key_column, dtype='foreign_key', link_to=f'{primary_key_table}.{primary_key_column}'),
                DBBColumnSchema(name=new_col_name, dtype=dtype),
                DBBColumnSchema(name=f'{new_table_name}ID', dtype='primary_key')
            ],
            source=f'data/{c_new_table_name}.pqt',
            format='parquet'
        )
        tables.append(new_table)
        dbb.metadata.tables = tables
    else:
        tables = dbb.metadata.tables
        del_col_idx = 0
        del_table_idx = 0
        for i, table in enumerate(tables):
            if table.name == f'{original_table}':
                for j, column in enumerate(table.columns):
                    if column.name == f'{multi_cat_col}':
                        del_col_idx = j
                del_table_idx = i
        tables[del_table_idx].columns.pop(del_col_idx)
        c_new_table_name = new_table_name.lower()
        new_table = DBBTableSchema(
            name=new_table_name,
            columns=[
                DBBColumnSchema(name=primary_key_column, dtype='foreign_key', link_to=f'{primary_key_table}.{primary_key_column}'),
                DBBColumnSchema(name=new_col_name, dtype='foreign_key', link_to=f'{new_col_name}.{new_col_name}ID')
            ],
            source=f'data/{c_new_table_name}.pqt',
            format='parquet'
        )
        tables.append(new_table)
        dbb.metadata.tables = tables
    # import ipdb; ipdb.set_trace()
    data = dbb.tables[original_table]
    rel_cols = {k:v for k, v in data.items() if k in [f'{primary_key_column}', f'{multi_cat_col}']}
    df = pd.DataFrame(rel_cols)
    del data[f'{multi_cat_col}']
    new_df = df[[f'{primary_key_column}', f'{multi_cat_col}']].explode(f'{multi_cat_col}').reset_index(drop=True)
    new_df = new_df.rename(columns={f'{multi_cat_col}': f'{new_col_name}'})
    if dtype != 'foreign_key':
        new_df[f'{new_table_name}ID'] = np.arange(new_df.shape[0])
    new_df_dict = {col: new_df[col].to_numpy() for col in new_df.columns}
    dbb.tables[new_table_name] = new_df_dict
    dbb.tables[original_table] = data
    return dbb

def generate_non_dummy_table(dbb, base_table_name, cols, new_table_name, **kwargs):
    """
        Description:
        Generate a non-dummy table with columns in the original table
        Parameters:
        dbb: the database object
        base_table_name: the name of the original table
        cols: the list of columns to be included in the new table and removed from the original table
        new_table_name: the name of the new table to be created
    """
    tables = dbb.metadata.tables
    del_idx = []
    col_info = []
    target_table_idx = 0
    for i, table in enumerate(tables):
        if table.name == f'{base_table_name}':
            for j, column in enumerate(table.columns):
                if column.name in cols:
                    del_idx.append((i, column.name))
                    if hasattr(column, 'link_to'):
                        col_info.append(DBBColumnSchema(name=column.name, dtype='foreign_key', link_to=column.link_to))
                    else:
                        col_info.append(DBBColumnSchema(name=column.name, dtype=column.dtype))
                    target_table_idx = i
    tables[target_table_idx].columns.append(DBBColumnSchema(name=f'{new_table_name}ID', dtype='foreign_key', link_to=f'{new_table_name}.{new_table_name}ID'))


    for (x, y) in del_idx:
        new_cols = [col for col in tables[x].columns if col.name != y]
        tables[x].columns = new_cols
    
    col_info.append(DBBColumnSchema(name=f'{new_table_name}ID', dtype='primary_key'))

    tables.append(DBBTableSchema(
        name=new_table_name,
        columns=col_info,
        source=f'data/{new_table_name}.pqt',
        format='parquet'
    ))
    dbb.metadata.tables = tables
    
    
    # import ipdb; ipdb.set_trace()
    data = dbb.tables[base_table_name]
    ndata = {k:v for k, v in data.items() if k in cols}
    ndata = pd.DataFrame(ndata)
    df_new = ndata.drop_duplicates().reset_index(drop=True).reset_index()
    df_new.rename(columns={'index': f'{new_table_name}ID'}, inplace=True)
    ndata = ndata.merge(df_new, on=cols, how='left')
    data = {k:v for k, v in data.items() if k not in cols}
    data[f'{new_table_name}ID'] = ndata[f'{new_table_name}ID'].values
    dbb.tables[base_table_name] = data
    new_df_dict = {col: df_new[col].to_numpy() for col in df_new.columns}
    dbb.tables[new_table_name] = new_df_dict
    return dbb
    
