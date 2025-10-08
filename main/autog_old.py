import os
import ast
import typer
import numpy as np
from rich import traceback
from models.autog.agent_improved import AutoG_Agent
from prompts.task import get_task_description
from utils.misc import seed_everything
from models.autog.agent_improved import load_dbb_dataset_from_cfg_path_no_name


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
    for task in data.metadata.tasks:
        if task.name != this_task:
            continue
            
        task_dict = {
            'name': task.name,
            'task_type': task.task_type,
            'target_column': task.target_column,
            'target_table': task.target_table,
            'evaluation_metric': task.evaluation_metric,
            'format': task.format,
            'source': task.source,
            'columns': []
        }

        for column in task.columns:
            # Skip if not in metadata and not special column
            is_special = (
                column.name == task.target_column or 
                column.dtype == 'datetime'
            )
            if not is_special and column.name not in meta_dict[task.target_table]:
                continue

            column_info = {
                'name': column.name,
                'dtype': (
                    column.dtype if is_special 
                    else meta_dict[task.target_table][column.name][0]
                )
            }
            task_dict['columns'].append(column_info)

        overall_meta['tasks'].append(task_dict)
    
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



def main(
    dataset: str = typer.Argument("rel-amazon", help="The dataset name of the RDB dataset"),
    schema_path: str = typer.Argument("output/data", help="Path to the data storage directory."),
    method: str = typer.Argument("autog-s", help="The method to run the model."),
    data_type_file: str = typer.Argument("stypes.json", help="The file to store the data type."),
    task_name: str = typer.Argument("user-churn", help="Name of the task to fit the solution."),
    seed: int = typer.Option(0, help="The seed to use for the model."),
    lm_path: str = typer.Option("deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27")
):
    """Main function to run AutoG agent."""
    seed_everything(seed)
    typer.echo("Agent version of the Auto-G")

    # Get LLM configuration
    llm_config = {"dummy": None}
    
    # Setup paths
    path_of_the_dataset = f"{schema_path}/{dataset}"
    # autog_path = os.path.join(path_of_the_dataset, "autog")
    # os.makedirs(autog_path, exist_ok=True)

    # Load metadata
    metainfo_path = os.path.join(path_of_the_dataset, 'type.txt')
    metainfo = read_txt_dict(metainfo_path)
    metainfo = {
        key: value 
        for key, value in metainfo.items()
    }

    # Load dataset information
    information_path = os.path.join(path_of_the_dataset, f'information.txt')
    with open(information_path, 'r') as file:
        information = file.read()

    # Load and prepare data
    ## this can be under the old subdirectory or directly under the dataset directory
    if os.path.exists(os.path.join(path_of_the_dataset, 'metadata.yaml')):
        old_data_config_path = os.path.join(path_of_the_dataset)
    else:
        old_data_config_path = os.path.join(path_of_the_dataset, 'old')
        path_of_the_dataset = old_data_config_path
    multi_tabular_data = load_dbb_dataset_from_cfg_path_no_name(old_data_config_path)
    task_description = get_task_description(dataset, task_name)
    schema_input = generate_training_metainfo(
        multi_tabular_data, 
        metainfo, 
        this_task=task_name
    )

    # Initialize and run agent
    agent = AutoG_Agent(
        initial_schema=schema_input,
        mode=method,
        oracle=None,
        path_to_file=path_of_the_dataset,
        use_cache=False,
        threshold=10,
        task_description=task_description,
        dataset=dataset,
        task_name=task_name,
        schema_info=information,
        lm_path=lm_path,
        data_type_file=data_type_file
    )
    
    agent.augment()
    augment_history = "\n".join(agent.history)
    typer.echo(f"Augmentation history: \n{augment_history}")


if __name__ == '__main__':
    traceback.install(show_locals=True)
    typer.run(main)