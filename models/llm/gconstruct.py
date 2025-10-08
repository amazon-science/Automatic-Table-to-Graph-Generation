# import textgrad as tg
from prompts.autog import single_round_prompt
from prompts.autog import EXAMPLE_INPUT_1, EXAMPLE_INPUT_2, code_generation_prompt
# from llama_index.core.llms import MessageRole, ChatMessage
from dbinfer import DBBRDBDataset, DBBRDBDatasetMeta
import json
import re
import typer
import pandas as pd
import duckdb
import numpy as np
from collections import defaultdict
import os

def remove_dummy_table(table_dict):
    new_dict = []
    dummy = []
    for value in table_dict:
        if len(value['columns']) == 1:
            if value['columns'][0]['dtype'] == 'primary_key':
                ## skip dummy table
                dummy.append(value['name'])
                continue
        new_dict.append(value)
    return new_dict
    
                

def extract_between_tags(text, tag_name):
    """Extracts content enclosed between the specified HTML-like tags.

    Args:
        text: The input string containing the tags and content.
        tag_name: The name of the tag (without angle brackets).

    Returns:
        A list of strings containing the extracted content, or an empty list 
        if no matching tags are found.
    """

    pattern = rf"<{tag_name}>([^<]+)</{tag_name}>"
    matches = re.findall(pattern, text)
    return matches

def remove_comments(code_string):
    """Removes comments from a Python code string.

    Args:
        code_string (str): The Python code string.

    Returns:
        str: The code string with comments removed.
    """
    result = ""
    in_comment = False
    for line in code_string.split('\n'):  # Process line by line
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            continue  # Ignore comment lines
        else:
            result += line + '\n'  # Keep non-comment lines
    return result

def run_llm_code(llm_output, namespace = {}):
    clean_output = remove_comments(llm_output)
    clean_output = clean_output.strip()
    exec(clean_output, namespace)
    
def get_single_round_prompt(input_data_schema, downstream, dataset_stats, dataset_meta_info):
    return single_round_prompt(
            [EXAMPLE_INPUT_1, EXAMPLE_INPUT_2], input_data_schema, downstream, dataset_stats, dataset_meta_info
        )

def get_dataset_column_stats(rdb_data: DBBRDBDataset, schema: DBBRDBDatasetMeta):
        ## min, max, quantiles
        ## number of unique values vs total number of values
        ## number of nan values
        ## average value, mode value
        ## 10 sampled values
        description = ""
        for table in schema['tables']:
            typer.echo(f"Table: {table['name']}")
            description += f"Table: {table['name']}\n"
            col_json = {}
            for col in table['columns']:
                if col['dtype'] == "primary_key" or col['dtype'] == "foreign_key" or col['dtype'] == "float" or col['dtype'] == "timestamp":
                    col_json["Column"] = col['name']
                    col_json["data type"] = col['dtype']
                    continue
                elif col['dtype'] == 'multi_category':
                    all_uniques = []
                    col_values = rdb_data.tables[table['name']][col['name']]
                    for sublist in col_values:
                        for item in sublist:
                            all_uniques.append(item)
                    sete_all_uniques = list(set(all_uniques))
                    col_json['Column'] = col['name']
                    col_json['data type'] = col['dtype']
                    col_json['Number of unique values'] = len(sete_all_uniques)
                    # col_json['Number of nan values'] = sum([1 for v in rdb_data.tables[table['name']][col['name']] if v is None])
                    col_json['Number of total values'] = len(all_uniques)
                    # mode_result = stats.mode(all_uniques, keepdims=True)
                    # count = Counter(all_uniques)
                    # mode_result = count.most_common(1)[0][0]
                    col_json['5 sampled values'] = all_uniques[:5]
                elif col['name'] not in rdb_data.tables[table['name']]:
                    continue
                else:
                    col_json["Column"] = col['name']
                    col_json["data type"] = col['dtype']
                    col_json["Number of unique values"] = len(set(rdb_data.tables[table['name']][col['name']]))
                    # col_json["Number of nan values"] = sum([1 for v in rdb_data.tables[table['name']][col['name']] if v is None])
                    col_json["Number of total values"] = len(rdb_data.tables[table['name']][col['name']])
                    # count = Counter(rdb_data.tables[table['name']][col['name']].tolist())
                    # mode_result = count.most_common(1)[0][0]
                    # col_json["Mode values"] = mode_result
                    col_json["5 sampled values"] = rdb_data.tables[table['name']][col['name']][:5].tolist()
                
                col_json_str = json.dumps(col_json, indent=2)
                description += col_json_str + "\n"
        return description
    
    
# def generate_augmentation_code(llm, first_round_input, first_round_output, dataset, table_info, input_schema, output_schema):
#     code_prompt = code_generation_prompt(dataset, table_info, input_schema, output_schema)
#     messages = [
#         ChatMessage(
#             role=MessageRole.USER,
#             content=(code_prompt)
#         )
#     ]
#     print(code_prompt)
#     res = llm.chat(messages, temperature=0, max_tokens=4096).message.content
#     code_res = extract_between_tags(res, "code")[0]
#     print(code_res)
#     return code_res, code_prompt


def analyze_dataframes(dataframes, k=5, dbb = None):
    """
    Analyzes a set of Pandas DataFrames and returns a formatted string 
    with column statistics for each DataFrame.

    Args:
        dataframes: A list or dictionary of Pandas DataFrames. If a dictionary,
                  keys are used as DataFrame names. If a list, DataFrames are 
                  numbered.
        k: The number of sampled values to display for each column.
        dbb: if dbb not None, only output what's in the metadata

    Returns:
        A string containing the analysis for each DataFrame.
    """

    output_string = ""

    if isinstance(dataframes, dict):
        dataframe_dict = dataframes
    elif isinstance(dataframes, list) or isinstance(dataframes, tuple) :  # Handle lists and tuples
        dataframe_dict = {f"DataFrame {i+1}": df for i, df in enumerate(dataframes)}
    else:
        raise TypeError("Input must be a dictionary, list, or tuple of DataFrames.")

    column_dict = defaultdict(list)
    if dbb is not None:
        for table in dbb.metadata.tables:
            for col in table.columns:
                column_dict[f'Table {table.name}'].append(col.name)
    for df_name, df in dataframe_dict.items():
        print(f"Analyzing DataFrame: {df_name}")
        output_string += f"Analysis for {df_name}:\n"
        objective = None 
        if isinstance(df, pd.DataFrame):
            objective = df.columns
        elif isinstance(df, dict):
            objective = df.keys()
        elif isinstance(df, duckdb.duckdb.DuckDBPyRelation):
            df = df.to_df()
            objective = df.columns
        for col_name in objective:
            if dbb is not None:
                if col_name not in column_dict[df_name]:
                    continue
            print(f"Analyzing column: {col_name}")
            output_string += f"  Column: {col_name}\n"
            
            if col_name == 'name_tokens':
                ## a workaround for diginetica, analyzing this column is too time consuming
                output_string += f"  This is a very large column\n"
                output_string += f"  Sampled values: [14140, 91983, 91983] [514132, 514132, 96631, 510711] [62897, 62896, 22144, 22144, 14755, 62903] \n" 
                continue

            if df[col_name].dtype is np.dtype('O') and (pd.Series(df[col_name]).apply(lambda x: isinstance(x, list)).any() or pd.Series(df[col_name]).apply(lambda x: isinstance(x, np.ndarray)).any()):  # Handle object columns
                if not isinstance(df[col_name], pd.Series):
                    df[col_name] = pd.Series(df[col_name])
                exploded = df[col_name].explode()
                total_elements = exploded.count()
                exploded = exploded.astype(str)
                unique_elements = exploded.nunique()
                mode_element = exploded.mode().iloc[0] if not exploded.empty else None
                before_explode = df[col_name].count()
                df[col_name] = df[col_name].astype(str)
                before_explode_unique = df[col_name].nunique()
                output_string += f"Column is a list. Probably a multi-category column. If you explode it, it will have {unique_elements} unique elements from total {total_elements}, and mode {mode_element}. Before exploding, there are {before_explode}, out of {before_explode_unique} is unique. Please consider whether you should explode it, you should explode it if it can bring strong network effect.\n"
                continue
            if len(df[col_name].shape) > 1:  # Handle multi-dimensional columns
                output_string += f"Column is multi-dimensional. Probably an embedding type. Usually not of interest\n"
                continue
            # Handle different data types more robustly
            elif pd.api.types.is_numeric_dtype(df[col_name]):
                max_val = df[col_name].max()
                min_val = df[col_name].min()
                output_string += f"    Max: {max_val}\n"
                output_string += f"    Min: {min_val}\n"
            elif pd.api.types.is_datetime64_any_dtype(df[col_name]) :
                max_val = df[col_name].max()
                min_val = df[col_name].min()
                output_string += f"    Max: {max_val}\n"
                output_string += f"    Min: {min_val}\n"

            try:
                if isinstance(df[col_name], pd.Series):
                    mode_val = df[col_name].mode().iloc[0]  # Handle multiple modes
                else:
                    col_series = pd.Series(df[col_name])
                    mode_val = col_series.mode().iloc[0]
                output_string += f"    Mode: {mode_val}\n"
            except Exception as e: # Catch potential errors like no mode
                output_string += f"    Mode: Could not determine mode. Error: {e}\n" # Informative error message

            if isinstance(df[col_name], pd.Series):
                total_num = len(df[col_name])
                num_unique = df[col_name].nunique()
                sampled_vals = df[col_name].sample(min(k, len(df))).values #Handle smaller dfs
                ## if sampled_vals is of type str, limit length of each one to 50, for the other part, replace with ...
                if any(isinstance(val, str) for val in sampled_vals):
                    sampled_vals = [val[:50] + "..." if isinstance(val, str) else val for val in sampled_vals]
            else:
                col_series = pd.Series(df[col_name])
                total_num = len(col_series)
                num_unique = col_series.nunique()
                sampled_vals = col_series.sample(min(k, len(col_series))).values
                if any(isinstance(val, str) for val in sampled_vals):
                    sampled_vals = [val[:50] + "..." if isinstance(val, str) else val for val in sampled_vals]

            output_string += f"    Sampled Values: {sampled_vals}\n"
            output_string += f"    Number of Total Values: {total_num}\n"
            output_string += f"    Number of Unique Values: {num_unique}\n"

        output_string += "\n"  # Add separator between DataFrames

    return output_string 


def dummy_llm_interaction(
    query_text: str,
    query_filepath: str = "query.txt",
    response_filepath: str = "response.txt",
    use_cache: bool = True,
    dataset: str = "",
    task: str = "",
    round_num: int = 0
) -> str:
    """
    Simulates an interaction with an LLM with automatic cache support.

    This function can:
    1. Load pre-cached responses from test files (for testing/debugging)
    2. Fall back to manual LLM interaction mode if cache unavailable

    Args:
        query_text: The query to send to the (simulated) LLM
        query_filepath: Path to save the query file
        response_filepath: Path where user should save the LLM response
        use_cache: If True, try to load from test cache first
        dataset: Dataset name (for cache lookup)
        task: Task name (for cache lookup)
        round_num: Current round number (for multi-round cache)

    Returns:
        The LLM response (from cache or manual file)

    Cache File Format:
        test/{dataset}_{task}.txt contains one response per line.
        Line N corresponds to round N (0-indexed).
    """
    # Try cache first if enabled
    if use_cache and dataset and task:
        cache_response = _try_load_from_cache(dataset, task, round_num)
        if cache_response is not None:
            print(f"✓ Cache hit: Loaded response for {dataset}/{task} round {round_num}")
            # Still save query for reference
            _save_query_file(query_text, query_filepath)
            return cache_response

    # Fall back to manual interaction
    print(f"Cache miss or disabled - using manual LLM interaction mode")
    return _manual_llm_interaction(query_text, query_filepath, response_filepath)


def _try_load_from_cache(dataset: str, task: str, round_num: int):
    """
    Try to load cached LLM response from test directory.

    Args:
        dataset: Dataset name
        task: Task name
        round_num: Round number (0-indexed)

    Returns:
        Cached response if found, None otherwise
    """
    # Build cache file path
    this_file_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_file_dir, '..', '..', "test")
    cache_file = os.path.join(test_dir, f"{dataset}_{task}.txt")

    # Check if cache file exists
    if not os.path.exists(cache_file):
        print(f"Cache file not found: {cache_file}")
        return None

    try:
        # Read all cached responses
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_responses = [line.strip() for line in f if line.strip()]

        # Check if we have a response for this round
        if round_num >= len(cached_responses):
            print(f"No cached response for round {round_num} (file has {len(cached_responses)} rounds)")
            return None

        response = cached_responses[round_num]

        # Handle special "no more actions" marker
        if response == "<selection>None</selection>":
            print(f"Cache indicates no more actions for round {round_num}")
            return response

        return response

    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def _save_query_file(query_text: str, query_filepath: str) -> None:
    """Save query to file for reference."""
    try:
        with open(query_filepath, 'w', encoding='utf-8') as q_file:
            q_file.write(query_text)
        print(f"Query saved to: {os.path.abspath(query_filepath)}")
    except Exception as e:
        print(f"Warning: Could not save query file: {e}")


def _manual_llm_interaction(
    query_text: str,
    query_filepath: str,
    response_filepath: str
) -> str:
    """
    Manual LLM interaction mode - prompts user to copy/paste.

    Args:
        query_text: The query to send
        query_filepath: Path to save query
        response_filepath: Path where user saves response

    Returns:
        Content of the response file
    """
    try:
        # 1. Store the query content to a file
        _save_query_file(query_text, query_filepath)

        # 2. Halt the program and prompt the user
        print("\n" + "=" * 60)
        print("ACTION REQUIRED: Manual LLM Interaction")
        print("=" * 60)
        print(f"1. Open the file: {os.path.abspath(query_filepath)}")
        print(f"2. Copy the query from '{query_filepath}'")
        print(f"3. Paste the query into your preferred LLM interface (e.g., web browser)")
        print(f"4. Copy the LLM's complete response")
        print(f"5. Save the response to: {os.path.abspath(response_filepath)}")
        print("=" * 60 + "\n")

        # Loop until the response file is found
        while not os.path.exists(response_filepath):
            input(f"Press Enter after you have saved the LLM's response to '{response_filepath}'...")
            if not os.path.exists(response_filepath):
                print(f"✗ File not found: {os.path.abspath(response_filepath)}")
                print("Please ensure you have saved the file correctly.")
            else:
                print(f"✓ Response file found: {os.path.abspath(response_filepath)}")
                break

        # 3. Read the content of the response file
        with open(response_filepath, 'r', encoding='utf-8') as r_file:
            llm_response = r_file.read()
        print(f"✓ LLM response successfully read from: {os.path.abspath(response_filepath)}\n")

        return llm_response

    except FileNotFoundError:
        print(f"Error: Could not find one of the files. Please check paths.")
        return "Error: File not found during operation."
    except IOError as e:
        print(f"An I/O error occurred: {e}")
        return f"Error: I/O error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Error: Unexpected error: {e}"