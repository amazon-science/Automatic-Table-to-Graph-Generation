import typer 
from models.autog.agent import AutoG_Agent
from prompts.task import get_task_description
import os
from rich import traceback
from utils.misc import seed_everything
import ast
from utils.data.rdb import load_dbb_dataset_from_cfg_path_no_name
import numpy as np
import shutil
import pandas as pd

def retrieve_input_schema(full_schema):
    input_schema = {
        key: value for key, value in full_schema.items() if key in ["dataset_name", "tables"]
    }
    return input_schema

def generate_training_metainfo(data, meta_dict, this_task):
    """
        Generate the meta information for the training data.
        Also need to identify primary key columns.
        Args:
            data: The data to be used for training.
            meta_dict: The meta information dictionary.
            this_task: The task to be used for training.
    """
    overall_meta = {}
    overall_meta['dataset_name'] = data.dataset_name
    overall_meta['tables'] = []
    
    ## turn data metadata list into a dict for easier query
    ## this is only used for inferring the data format
    data_meta_dict = {key.name: key for key in data.metadata.tables}
    for table in data.tables:
        table_val = data.tables[table]
        table_meta_dict = {}
        table_meta_dict['name'] = table
        table_meta_dict['columns'] = []
        table_meta_dict['format'] = data_meta_dict[table].format.value
        table_meta_dict['source'] = data_meta_dict[table].source
        ## identify primary_key
        for column_name, column_value in table_val.items():
            ## if not present in metadata, then skip
            if column_name not in meta_dict[table]:
                continue
            ## check if the element type of current column is a list
            if column_value.dtype == 'int64' or column_value.dtype == 'float64':
                is_numerical = True
            elif column_value.dtype == 'object':
                try:
                    column_value = column_value.astype(int)
                    is_numerical = True
                except Exception as e:
                    is_numerical = False
            else:
                is_numerical = False
            if_primary_key = is_numerical and np.unique(column_value).size == column_value.size
            if if_primary_key:
                table_meta_dict['columns'].append({
                    'name': column_name,
                    'dtype': 'primary_key',
                    'description': meta_dict[table][column_name][1]
                })
                continue
            
            table_meta_dict['columns'].append({
                'name': column_name,
                'dtype': meta_dict[table][column_name][0],
                'description': meta_dict[table][column_name][1]
            })
        overall_meta['tables'].append(table_meta_dict)
    
    ## add the task meta information
    overall_meta['tasks'] = []
    for task in data.metadata.tasks: 
        if task.name != this_task:
            continue
        task_dict = {}
        task_dict['name'] = task.name
        task_dict['task_type'] = task.task_type
        task_dict['target_column'] = task.target_column
        task_dict['target_table'] = task.target_table
        task_dict['evaluation_metric'] = task.evaluation_metric
        task_dict['format'] = task.format
        task_dict['source'] = task.source
        columns = []
        for column in task.columns:
            ## if not present in metadata, then skip
            if column.name != task.target_column and column.dtype != 'datetime' and column.name not in meta_dict[task.target_table]:
                continue
            if column.name == task.target_column or column.dtype == 'datetime':
                columns.append({
                'name': column.name,
                'dtype': column.dtype,
                })
            else:
                columns.append({
                'name': column.name,
                'dtype': meta_dict[task.target_table][column.name][0],
            })
        task_dict['columns'] = columns
        overall_meta['tasks'].append(task_dict)
    return overall_meta

     
def read_txt_dict(file_path):
    """
        Read the dictionary from the text file.
    """
    # Step 1: Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    # Step 2: Parse the content into a Python dictionary
    data_dict = ast.literal_eval(content)
    return data_dict

def capitalize_first_alpha_concise(text):
    for i, char in enumerate(text):
        if char.isalpha():
            return text[:i] + text[i:].replace(char, char.upper(), 1)
            # Only replace the first instance of the first alpha character
    return text


def main(
        dataset: str = typer.Argument(
        "mag",
        help="The dataset name of the RDB dataset"),
        llm_name: str = typer.Argument("sonnet3", help="The name of the LLM model to use."), 
        schema_path: str = typer.Argument(
            "newdatasets",
            help="Path to the data storage directory."
        ),
        method: str = typer.Argument(
            "autog-s",
            help="The method to run the model in."),
        task_name : str = typer.Argument(
            "venue",
            help="Name of the task to fit the solution."),
        seed: int = typer.Option(0, help="The seed to use for the model."), 
        lm_path: str = typer.Option(
            "deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27")
        ):
    seed_everything(seed)
    typer.echo("Agent version of the Auto-G")
    if llm_name == "sonnet3":
        llm_model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
        context_size = 200000
    elif llm_name == "llama3":
        llm_model_name = "meta.llama3-70b-instruct-v1:0"
        context_size = -1
    elif llm_name == "mistralarge":
        llm_model_name = "mistral.mistral-large-2402-v1:0"
        context_size = 32000
    elif llm_name == "sonnet35":
        llm_model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        context_size = 200000
    elif llm_name == "opus3":
        llm_model_name = "anthropic.claude-3-opus-20240229-v1:0" 
        context_size = 200000
    elif llm_name == 'haiku3':
        llm_model_name = "anthropic.claude-3-haiku-20240229-v1:0"
        context_size = 200000
    
    if "llama" in llm_name:
        output_size = -1
    else:
        output_size = 4096
    
    path_of_the_dataset = f"{schema_path}/{dataset}"
    autog_path = os.path.join(path_of_the_dataset, "autog")
    os.makedirs(autog_path, exist_ok=True)
    metainfo_path = os.path.join(path_of_the_dataset, 'type.txt')
    metainfo = read_txt_dict(metainfo_path)
    information_path = os.path.join(path_of_the_dataset, 'information.txt')
    with open(information_path, 'r') as file:
        information = file.read()
    ## notation here: table name with first letter capitalized
    metainfo = {capitalize_first_alpha_concise(key): value for key, value in metainfo.items()}
    old_data_config_path = os.path.join(path_of_the_dataset, 'old')
    multi_tabular_data = load_dbb_dataset_from_cfg_path_no_name(old_data_config_path)
    task_description = get_task_description(dataset, task_name)
    ## generate the initial data schema by letting llms determine the initial data types
    schema_input = generate_training_metainfo(multi_tabular_data, metainfo, this_task=task_name)
    ## remove old autog directory and create a new one
    autog_dir_path = os.path.join(path_of_the_dataset, "autog")
    for d in os.listdir(autog_dir_path):
        if d != "round_0":
            shutil.rmtree(os.path.join(autog_dir_path, d))
    os.makedirs(autog_dir_path, exist_ok=True)
    agent = AutoG_Agent(initial_schema=schema_input, mode=method, oracle=None,
        llm_model_name=llm_model_name, context_size=context_size, path_to_file=autog_dir_path, llm_sleep=1, use_cache=False, threshold=10, output_size=output_size, task_description=task_description, dataset=dataset, task_name=task_name, schema_info = information, lm_path=lm_path)
    agent.augment()
    augment_history = "\n".join(agent.history)
    typer.echo(f"Augmentation history: \n{augment_history}")
    

if __name__ == '__main__':
    traceback.install(show_locals=True)
    typer.run(main)