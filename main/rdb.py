import typer 
from pathlib import Path
import os.path as osp
from utils.data.rdb import load_dbb_dataset_from_cfg_path
from utils.plot import plot_rdb_dataset_schema
from dbinfer.cli import preprocess
from dbinfer.cli import construct_graph
from dbinfer.cli import fit_gml, GMLSolutionChoice
from dbinfer.task_construct_utils import dataset_stats
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def main(dataset: str = typer.Argument(
        "MAG", 
        help="The dataset name of the RDB dataset"),
         cache_path: str = typer.Argument(
                "./multi-table-benchmark/datasets", help="The path to the cache directory"),
         schema_path: str = typer.Argument(
                "./multi-table-benchmark/datasets/mag", help="The path to the schema directory"),
         method: str = typer.Argument(
             "r2n",
             help="The method to use for graph construction"),
         model: str = typer.Argument(
             "gat",
             help="The model to use for graph learning"),
         task_name : str = typer.Argument(
            "venue",
            help="Name of the task to fit the solution."  
        ),
         num_of_seeds: int = typer.Argument(
            1,
            help="Number of seeds to use for training."),
         solution_config_path : Path = typer.Option(
            "configs/mag/gat-cite.yaml",
            "--config_path", "-c",
            help="Solution configuration path. Use default if not specified."
        ),
        checkpoint_path : str = typer.Option(
            None, 
            "--checkpoint_path", "-p",
            help="Checkpoint path."
        ),
         use_wandb: bool = typer.Option(
                True,
                "--enable-wandb/--disable-wandb",
                help="Whether to use wandb for logging"),
         plot_only: bool = typer.Option(
                False,
                "--plot-only",
                help="Whether to only plot the schema of the dataset"),
         stats_only: bool = typer.Option(
                False,
                "--stats-only",
                help="Whether to only compute the statistics of the dataset"),
         skip_train: bool = typer.Option(
                False,
                "--skip-train",
                help="Whether to skip the training process"),
         use_cache: bool = typer.Option(
                False,
                "--use-cache",
                help="Whether to use the cache for the dataset")
         ):
    
    # import ipdb; ipdb.set_trace()    
    loaded_rdb_dataset, data_id = load_dbb_dataset_from_cfg_path(dataset, schema_path)
    plot_rdb_dataset_schema(loaded_rdb_dataset, f"{cache_path}/{data_id}-schema")
    
    if plot_only:
        return
    # import ipdb; ipdb.set_trace()
    
    # dataset_stats(loaded_rdb_dataset)
    if stats_only:
        return
    
    # if skip_process:
        ## preprocess the datasets
    if use_cache and osp.exists(f"{cache_path}/{data_id}/{data_id}-preprocessed"):
        preprocessed_file_paths = f"{cache_path}/{data_id}/{data_id}-preprocessed"
    else:
        preprocessed_file_paths = preprocess(dataset, schema_path, "transform", f"{cache_path}/{data_id}/{data_id}-preprocessed", None)
    
    config_file_name = schema_path.split("/")[-1].split(".")[0]
    ## construct the graph
    if osp.exists(method):
        typer.echo("Method is provided in a file.")
        with open(method, "r") as f:
            method = f.read().strip()
            if method == 'Row2Node/Edge':
                method = 'r2ne'
            elif method == 'Row2Node':
                method = 'r2n'
            else:
                method = 'r2ne'
    if use_cache and osp.exists(f"{cache_path}/{data_id}/{data_id}-{config_file_name}-graph-{method}"):
        output_g_path = f"{cache_path}/{data_id}/{data_id}-{config_file_name}-graph-{method}"
        logger.info(f"Graph already constructed at {output_g_path}")
    else:
        output_g_path = construct_graph(preprocessed_file_paths, method, f"{cache_path}/{data_id}/{data_id}-{config_file_name}-graph-{method}", None)
        logger.info(f"Graph constructed at {output_g_path}")

    # solution_config_path = osp.join(solution_config_path, data_id, f"{model}-{task_name}.yaml")
    if skip_train:
        return
    # import ipdb; ipdb.set_trace()
    fit_gml(
        output_g_path, 
        task_name,
        GMLSolutionChoice(model),
        solution_config_path,
        checkpoint_path,  
        use_wandb,
        num_of_seeds
    )
    
    
    
    
    
     

if __name__ == '__main__':
    typer.run(main)