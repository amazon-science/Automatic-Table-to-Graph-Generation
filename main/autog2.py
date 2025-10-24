import os
import ast
import typer
import yaml
import numpy as np
import pandas as pd
from rich import traceback
from models.autog.agent_old import AutoG_Agent
from models.llm.bedrock import get_bedrock_llm
from prompts.task import get_task_description
from prompts.identify import identify_prompt
from utils.misc import seed_everything
from utils.data.rdb import load_dbb_dataset_from_cfg_path_no_name
from models.llm.gconstruct import analyze_dataframes


CONTEXT_SIZE = 65536
OUTPUT_SIZE = 65536

dtype_mapping = {
    # Category
    'object': 'category',
    'string': 'category', 
    'category': 'category',
    
    # Float
    'int8': 'float',
    'int16': 'float',
    'int32': 'float', 
    'int64': 'float',
    'float16': 'float',
    'float32': 'float',
    'float64': 'float',
    'bool': 'float',
    
    # Datetime
    'datetime64[ns]': 'datetime',
    'timedelta64[ns]': 'datetime',
    'period[D]': 'datetime'
}

def retrieve_input_schema(full_schema):
    input_schema = {
        key: value for key, value in full_schema.items() if key in ["dataset_name", "tables"]
    }
    return input_schema


def generate_training_metainfo(data, meta_dict, this_task):
    """Generate the meta information for the training data.
    
    Args:
        data: The data to be used for training.
        meta_dict: The meta information dictionary.
        this_task: The task to be used for training.
    
    Returns:
        Dictionary containing dataset metadata including tables and tasks.
    """
    overall_meta = {
        'dataset_name': data.dataset_name,
        'tables': []
    }
    
    # Convert data metadata list to dict for easier query
    data_meta_dict = {key.name: key for key in data.metadata.tables}
    
    for table in data.tables:
        table_val = data.tables[table]
        table_meta_dict = {
            'name': table,
            'columns': [],
            'format': data_meta_dict[table].format.value,
            'source': data_meta_dict[table].source
        }

        for column_name, column_value in table_val.items():
            # Skip if not present in metadata
            if column_name not in meta_dict[table]:
                continue

            # Check if column is numerical
            is_numerical = False
            if column_value.dtype in ['int64', 'float64']:
                is_numerical = True
            elif column_value.dtype == 'object':
                try:
                    column_value.astype(int)
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
    
    # Add task meta information
    overall_meta['tasks'] = []
    # for task in data.metadata.tasks:
    #     if task.name != this_task:
    #         continue
            
    #     task_dict = {
    #         'name': task.name,
    #         'task_type': task.task_type,
    #         'target_column': task.target_column,
    #         'target_table': task.target_table,
    #         'evaluation_metric': task.evaluation_metric,
    #         'format': task.format,
    #         'source': task.source,
    #         'columns': []
    #     }

    #     for column in task.columns:
    #         # Skip if not in metadata and not special column
    #         is_special = (
    #             column.name == task.target_column or 
    #             column.dtype == 'datetime'
    #         )
    #         if not is_special and column.name not in meta_dict[task.target_table]:
    #             continue

    #         column_info = {
    #             'name': column.name,
    #             'dtype': (
    #                 column.dtype if is_special 
    #                 else meta_dict[task.target_table][column.name][0]
    #             )
    #         }
    #         task_dict['columns'].append(column_info)

    #     overall_meta['tasks'].append(task_dict)
    
    return overall_meta


def read_txt_dict(file_path):
    """Read dictionary from a text file."""
    with open(file_path, 'r') as file:
        content = file.read()
    return ast.literal_eval(content)


def capitalize_first_alpha_concise(text):
    """Capitalize the first alphabetic character in text."""
    for i, char in enumerate(text):
        if char.isalpha():
            return text[:i] + text[i:].replace(char, char.upper(), 1)
    return text

def generate_metadata(table_path, dataset_name, data_format='csv'):
    """ automatically explore the table data and generate metedata
    """
    meta_dict = {
        'dataset_name': dataset_name,
        'tables': []
    }

    files = os.listdir(table_path)
    file_paths = [os.path.join(table_path, file) for file in files]
    # filter out directories, leaving files only
    file_paths = [file_path for file_path in file_paths if os.path.isfile(file_path)]

    for file_path in file_paths:
        # TODO(Jian) remove the file name constraints.
        # Rename file name to capitalize first alphabetic letter for using LLMs
        file_name = os.path.basename(file_path)
        new_file_name = capitalize_first_alpha_concise(file_name)
        # rename the table name
        os.rename(os.path.join(table_path, file_name),
                  os.path.join(table_path, new_file_name))
        table_name = os.path.splitext(new_file_name)[0]

        if data_format == 'csv':
            table_df = pd.read_csv(os.path.join(table_path, new_file_name))
        elif data_format == 'parquet':
            table_df = pd.read_parquet(os.path.join(table_path, new_file_name))
        else:
            raise NotImplementedError
        
        table_meta_dict = {
            'name': table_name,
            'columns': [],
            'format': 'parquet',
            'source': 'data/' + table_name + '.pqt'
        }

        for column_name, column_dtype in table_df.dtypes.to_dict().items():
            # Check if column is numerical
            dtype = column_dtype
            if column_dtype == 'object':
                try:
                    column_dtype.astype(int)
                    dtype = 'int32'
                except Exception as e:
                    dtype = 'object'

            table_meta_dict['columns'].append({
                'name': column_name,
                'dtype': dtype_mapping.get(dtype, 'category'),
            })
        meta_dict['tables'].append(table_meta_dict)

    return meta_dict

def get_llm_config(llm_name):
    """Get LLM configuration based on model name."""
    configs = {
        "sonnet3": {
            "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE
        },
        "llama3": {
            "model_name": "meta.llama3-70b-instruct-v1:0",
            "context_size": -1,
            "output_size": -1
        },
        "mistralarge": {
            "model_name": "mistral.mistral-large-2402-v1:0",
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE
        },
        "sonnet37": {
            "model_name": "arn:aws:bedrock:us-west-2:911734752298:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE
        },
        "sonnet4": {
            "model_name": "arn:aws:bedrock:us-west-2:911734752298:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE
        },
        "opus3": {
            "model_name": "anthropic.claude-3-opus-20240229-v1:0",
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE
        },
        "opus4": {
            "model_name": "anthropic.claude-opus-4-20250514-v1:0",
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE
        },
        "haiku3": {
            "model_name": "anthropic.claude-3-haiku-20240229-v1:0",
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE
        },
        "sonnet45": {
            "model_name": "arn:aws:bedrock:us-west-2:911734752298:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE
        }
    }
    return configs.get(llm_name, configs["sonnet3"])

def main(
    dataset_path: str = typer.Argument(..., help="The path to the dataset"),
    llm_name: str = typer.Argument("sonnet3", help="The name of the LLM model to use."),
    method: str = typer.Argument("autog-s", help="The method to run the model."),
    task_name: str = typer.Argument("mag:venue", help="Name of the task to fit the solution."),
    seed: int = typer.Option(0, help="The seed to use for the model."),
    lm_path: str = typer.Option("deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27"),
    dataset_name: str = typer.Argument("dataset", help="The name of dataset to be augemented."),
    data_format: str = typer.Option("parquet", help="The format of tables. 'parquet' or 'csv'.")
):
    """Main function to run AutoG agent."""
    seed_everything(seed)
    typer.echo("Agent version of the Auto-G")

    # Get LLM configuration
    llm_config = get_llm_config(llm_name)
    print(f'Using the following LLM configurations:{llm_config}')
    
    # explore the original tables to generate the metadata.yaml for DBBDataset
    # here we assume tables are stored in the ./data folder in the dataset_path
    table_path = os.path.join(dataset_path, 'data')
    assert os.path.exists(table_path), ("Expected the data tables are stored under the "
            f"{table_path}, but the path does not exist! Please move your data tables "
            "to this location.")
    metadata_dict = generate_metadata(table_path, dataset_name, data_format)
    # save metedata to metadata.yaml
    metadata_path = os.path.join(dataset_path, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata_dict, f, default_flow_style=False)
    print(f"Saved table metadata to {metadata_path}...")
    
    # automatically create the information contents by calling analyze_dataframes()
    multi_tabular_data = load_dbb_dataset_from_cfg_path_no_name(dataset_path)
    dataset, task = task_name.split(':')[0], task_name.split(':')[1]
    task_description = get_task_description(dataset, task)

    table_meta_dict = {
            f'Table {table_name}': table for table_name, table in multi_tabular_data.tables.items()
        }
    print(f'Analyze given tables ...')
    analysis_rst = analyze_dataframes(table_meta_dict)

    info_path = os.path.join(dataset_path, 'information.txt') 
    with open(info_path, 'w') as f:
        f.write(analysis_rst)
    print(f"Saved analysis results to {info_path} ...")

    identify_inputs = identify_prompt(analysis_rst)

    bedrock_llm = get_bedrock_llm(llm_config["model_name"], context_size=llm_config["context_size"])
    response = bedrock_llm.complete(identify_inputs, max_tokens=OUTPUT_SIZE).text

    # handle response by extracting on the JSON contents for Sonnet 4
    start = response.find('{')
    end = response.rfind('}') + 1
    val_response = response[start:end]
    # start_pt = len('''Looking at the data analysis for each table, I'll identify the data types and provide descriptions:''')
    # start_pt = 0
    metainfo = ast.literal_eval(val_response)
    metainfo = {
        capitalize_first_alpha_concise(key): value 
        for key, value in metainfo.items()
    }

    # Load and prepare data
    print(f'The task for {dataset} data: {task_description}')

    schema_input = generate_training_metainfo(
        multi_tabular_data, 
        metainfo, 
        this_task=task
    )

    # Setup paths
    autog_path = os.path.join(dataset_path, "autog")
    os.makedirs(autog_path, exist_ok=True)

    # Initialize and run agent
    agent = AutoG_Agent(
        initial_schema=schema_input,
        mode=method,
        oracle=None,
        llm_model_name=llm_config["model_name"],
        context_size=llm_config["context_size"],
        path_to_file=autog_path,
        llm_sleep=1,
        use_cache=False,
        threshold=20,
        output_size=llm_config["output_size"],
        task_description=task_description,
        dataset=dataset,
        task_name=task,
        schema_info=analysis_rst,
        lm_path=lm_path,
        recalculate=False
    )
    
    agent.augment()
    augment_history = "\n".join(agent.history)
    typer.echo(f"Augmentation history: \n{augment_history}")


if __name__ == '__main__':
    traceback.install(show_locals=True)
    typer.run(main)