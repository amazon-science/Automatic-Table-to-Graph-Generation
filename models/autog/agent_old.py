import traceback

from models.llm.bedrock import get_bedrock_llm, bedrock_llm_query
import os
# from models.autog.action import get_autog_actions, pack_function_introduction_prompt, turn_dbb_into_a_lookup_table
from models.autog.action_new import get_autog_actions, pack_function_introduction_prompt, turn_dbb_into_a_lookup_table
from prompts.mautog import get_multi_round_action_selection_prompt, get_single_round_multi_step_prompt
from utils.data.rdb import load_dbb_dataset_from_cfg_path_no_name
from models.llm.gconstruct import extract_between_tags, analyze_dataframes
import typer
import time
from copy import deepcopy
from utils.misc import copy_directory
import shutil
import ast
import yaml
from models.autog.deepjoin import join_discovery, load_pretrain_jtd_lm
import joblib
from typing import Dict, Tuple
import json
from copy import deepcopy
from utils.plot import plot_rdb_dataset_schema
from dbinfer_bench.dataset_meta import DBBColumnSchema


OUTPUT_SIZE = 32000


def format_top_k_similarities(dbb, similarity_dict: Dict[Tuple[str, str, str, str], float], k: int) -> str:
    """Formats the top k most similar pairs into a string.

    Args:
        similarity_dict: A dictionary where keys are tuples representing 
                         (Table 1, column 1, Table 2, column 2) and values are 
                         similarity scores.
        k: The number of top pairs to include in the output.

    Returns:
        A string containing information about the top k pairs, sorted by 
        similarity in descending order.
    """

    sorted_similarities = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)

    output_string = ""
    lookup = turn_dbb_into_a_lookup_table(dbb)
    valid_pair = 0
    for i, ((table1, col1, table2, col2), similarity) in enumerate(sorted_similarities):
        ## skip if these two columns already have some relationships
        ## case 1. table 1 col 1 already to pointing to table 2 col 2
        if not lookup.get((table1, col1)) or not lookup.get((table2, col2)):
            continue
        if hasattr(lookup[(table1, col1)], 'link_to') and lookup[(table1, col1)].link_to == (table2, col2):
            continue
        ## case 2. table 2 col 2 already to pointing to table 1 col 1
        if hasattr(lookup[(table2, col2)], 'link_to') and lookup[(table2, col2)].link_to == (table1, col1):
            continue
        ## case 3. table 1 col 1 and table 2 col 2 are already linked
        if hasattr(lookup[(table1, col1)], 'link_to') and hasattr(lookup[(table2, col2)], 'link_to') and lookup[(table1, col1)].link_to == lookup[(table2, col2)].link_to:
            continue
        ## case 4. pk and fk
        if lookup[(table1, col1)].dtype == 'primary_key' and lookup[(table2, col2)].dtype == 'foreign_key' and lookup[(table2, col2)].link_to == f"{table1}.{col1}":
            continue
        ## symmetry to case 4
        if lookup[(table2, col2)].dtype == 'primary_key' and lookup[(table1, col1)].dtype == 'foreign_key' and lookup[(table1, col1)].link_to == f"{table2}.{col2}":
            continue
        ## case 5. if any one is float
        if lookup[(table1, col1)].dtype == 'float' or lookup[(table2, col2)].dtype == 'float':
            continue
        if valid_pair >= k:
            break
        valid_pair += 1
        output_string += (
            f"The pair with the {ordinal(valid_pair)} highest similarity is column "
            f'"{col1}" from Table "{table1}" and column "{col2}" from Table "{table2}" '
            f"with similarity {similarity:.3f}\n"  # Format similarity to 3 decimal places
        )
    return output_string


def ordinal(n: int) -> str:
    """Returns the ordinal form of a number (e.g., 1st, 2nd, 3rd, 4th, etc.)."""
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    suffix = suffixes.get(n % 10, "th") if n % 10 in suffixes or n % 100 // 10 == 1 else "th"
    return str(n) + suffix


class AutoG_Agent():
    def __init__(self, initial_schema, mode="autog-s", oracle = None,  
                 llm_model_name = "anthropic.claude-3-sonnet-20240229-v1:0", context_size = 4096, 
                 icl_k = 6, path_to_file = "",
                 llm_sleep=1, use_cache = False, 
                 threshold = 10, output_size = 4096, task_description = 'autog', dataset = 'mag', task_name = 'venue', schema_info = "", lm_path = "", jtd_k = 20, recalculate = True) -> None:
        """
            Main agent program for AutoG
            Args:
                initial_schema: dict: the initial schema of the data inferred by llms 
                mode: str: the mode of the agent, either autog-a or autog-s. autog-a will generate several candidates while autog-s will use the last state as the output
                oracle: the oracle model to evaluate the schema
                llm_model_name: str: the name of the llm model
                context_size: int: the context size of the llm model
                icl_k: int: the number of ICL samples
                path_to_file: str: the path to the file
                llm_sleep: int: the sleep time of the llm api call 
                use_cache: bool: whether to use cache for llm calling
                threshold: int: the maximum number of running rounds for autog
                output_size: int: the output context size of llm
                task_description: str: the task description of the current task
                dataset: str: the dataset name
                task_name: str: the task name
                schema_info: str: the schema information
                lm_path: str: the path to the pre-trained deep join model
                jtd_k: int: the number of top k similar columns
                recalculate: bool: whether to recalculate the deep join and statistics for each round
        """
        self.llm = get_bedrock_llm(llm_model_name, context_size=context_size)
        self.action_list = get_autog_actions()
        self.threshold = threshold
        self.state = initial_schema
        self.original_state = deepcopy(initial_schema)
        self.icl_k = icl_k
        self.mode = mode
        self.oracle = oracle
        self.icl_strategy = "random"
        self.path_to_file = path_to_file
        self.use_cache = use_cache
        self.dataset = dataset  
        ## eval mode: whether eval each step or eval at the end ('step', 'end')
        self.error = 0
        self.success = 0        
        ## find the file name of the target table
        ## storing an object: 
        ## index 
        ## path to saved objects
        self.output_size = output_size
        self.task_name = task_name
        self.task_description = task_description
        self.schema_info = schema_info
        self.lm_path = lm_path
        self.jtd_k = jtd_k
        self.recalculate = recalculate
        self.round = 0
        
        ## by default, we use the default prompts
        self.icl_demonstrations = []
        self.history = []
        self.llm_sleep = llm_sleep
        #if self.mode == 'autog-s':
        examples = get_single_round_multi_step_prompt()
        for example in examples:
            self.icl_demonstrations.append(example)

    def backup(self, dbb):
        """
            Backup the current state
        """
        dbb.save(os.path.join(self.path_to_file, f"backup_{len(self.history)}")) 
    
    def calculate_deepjoin(self, rdb_dataset):
        """
            Calculate the deepjoin
        """
        if self.jtd_k == 0:
            return ""
        # import ipdb; ipdb.set_trace()
        if os.path.exists(os.path.join(self.path_to_file, 'round_0', 'deepjoin.pkl')) and not self.recalculate:
            typer.echo("Load the deepjoin from cache")
            result=joblib.load(os.path.join(self.path_to_file, 'round_0', 'deepjoin.pkl'))
        elif self.recalculate:
            typer.echo("First try to see the deepjoin state")
            if os.path.exists(os.path.join(self.path_to_file, 'round_0', f'deepjoin{self.round}.pkl')):
                result = joblib.load(os.path.join(self.path_to_file, 'round_0', f'deepjoin{self.round}.pkl'))
            else:
                typer.echo("Calculate the deepjoin")
                model = load_pretrain_jtd_lm(self.lm_path)
                result = join_discovery(rdb_dataset, model)
                joblib.dump(result, os.path.join(self.path_to_file, 'round_0', f'deepjoin{self.round}.pkl'))
        else:
            typer.echo("Calculate the deepjoin")
            model = load_pretrain_jtd_lm(self.lm_path)
            result = join_discovery(rdb_dataset, model)
            joblib.dump(result, os.path.join(self.path_to_file, 'round_0', 'deepjoin.pkl'))
        ## turn numerical result into prompts
        result_prompt = format_top_k_similarities(rdb_dataset, result, self.jtd_k)
        return result_prompt
    
    def clean_backup(self):
        all_dirs = os.listdir(self.dataset_cache_path)
        for dir in all_dirs:
            if os.path.isdir(os.path.join(self.dataset_cache_path, dir)) and "backup_" in dir:
                full_path = os.path.join(self.dataset_cache_path, dir)
                shutil.rmtree(full_path, ignore_errors=True)

    def retrieve_icl_samples(self):
        """
            Retrieve the ICL samples
        """
        return self.icl_demonstrations
    
    def pack_prompts(self, dbb):
        """
            Generate the AutoG prompts
            args:
                update: bool: whether to update the stats after changing the schema
        """
        ## get examples 
        examples = self.retrieve_icl_samples()
        example_str = "\n".join(examples)
        ## get instruction of the actions 
        action_strs = [pack_function_introduction_prompt(action_f) for _, action_f in self.action_list.items()]
        # import ipdb ; ipdb.set_trace()
        action_description = "\n\n".join(action_strs)
        history_str = "\n\n".join(self.history) if len(self.history) > 0 else "First iteration, no history yet\n\n"
        ## get the schema
        schema = dbb.metadata.json()
        if not self.recalculate:
            stats = self.schema_info
        else:
            table_meta_dict = {
                f'Table {table_name}': table for table_name, table in dbb.tables.items()
            }
            if self.dataset == 'stackexchange':
                stats = analyze_dataframes(table_meta_dict, dbb = dbb)
            else:
                stats = analyze_dataframes(table_meta_dict)
        deepjoin_prior = self.calculate_deepjoin(dbb)
        full_prompts = get_multi_round_action_selection_prompt(
            action_description, example_str, history_str, schema, stats, self.task_description, deepjoin_prior
        )

        ## save the current prompt for debug 
        with open(os.path.join(self.dataset_cache_path, 'prompt.txt'), 'w') as f:
            f.write(full_prompts)
        return full_prompts
    
    def parse_args(self, parameters):
        return ast.literal_eval(parameters)
    
    def update_task(self, dbb):
        """
            Update the task type information. 
            Here, we assume the table of the task won't be change since it should be given by users.
        """
        ## get the task (should be only one)
        ## diginetica is a special case, directly return, don't change
        if self.dataset == 'diginetica':
            return dbb
        ## go through all table metadata, if there's multi_category type change them to category
        for table in dbb.metadata.tables:
            time_column = None
            for column in table.columns:
                if column.dtype == 'multi_category':
                    column.dtype = 'category'
                elif column.dtype == 'datetime':
                    time_column = column.name
            table.time_column = time_column

        task = dbb.metadata.tasks[0]
        columns = task.columns
        target_table = task.target_table
        ## find the table 
        find_table = [table for table in dbb.metadata.tables if table.name == target_table][0]
        ## mapping between column name and type 
        column_type_mapping = {column.name: column.dtype for column in find_table.columns}
        link_to_mapping = {column.name: column.link_to for column in find_table.columns if hasattr(column, 'link_to')}
        find_table_column_name = set([column.name for column in find_table.columns])
        ## update the task
        pop_name = []
        time_column = None
        for idx, _ in enumerate(columns):
            # if columns[idx].d
            if columns[idx].name == task.target_column or columns[idx].dtype == 'datetime':
                continue
            if columns[idx].name not in find_table_column_name:
                pop_name.append(columns[idx].name)
                continue
            if columns[idx].name in column_type_mapping:
                columns[idx].dtype = column_type_mapping[columns[idx].name]
            if columns[idx].dtype == 'foreign_key':
                columns[idx].link_to = link_to_mapping[columns[idx].name]
        ## remove the columns with name in pop_name
        for name in pop_name:
            columns = [col for col in columns if col.name != name]
            dbb.tasks[0].train_set.pop(name)
            dbb.tasks[0].validation_set.pop(name)
            dbb.tasks[0].test_set.pop(name)
        task.columns = columns
        dbb.metadata.tasks[0] = task
        return dbb

    def decide_next_step(self, dbb, epoch):
        """
            Input the current state, let LLMs determine the next action
        """
        # import ipdb; ipdb.set_trace()
        selection = "nothing selected yet"
        # import ipdb; ipdb.set_trace()
        this_round_dbb = dbb
        this_round_prompt = self.pack_prompts(this_round_dbb)
        response = bedrock_llm_query(self.llm, this_round_prompt, max_tokens = self.output_size, cache=self.use_cache,
                                     debug=False, debug_dataset=self.dataset, debug_task=self.task_name,
                                     debug_round=epoch-1)

        selection = extract_between_tags(response, "selection")[0].strip()

        # print(f'Selection: {selection} ...')

        if selection == "None":
            return this_round_dbb, False
        # method = extract_between_tags(response, "construction")[0].strip()
        ## update here, no need to let llm tell method, just try r2n and r2ne both
        method = 'r2n'
        

        ## use a for loop to run these commands
        selection = json.loads(selection)
        # import ipdb; ipdb.set_trace()
        for move in selection:
            typer.echo(f"Move: {move}")
            last_valid_dbb = deepcopy(this_round_dbb)
            explanation = move['explanation']
            methods = move['action']
            parameters = move['parameters']
            action_code = self.action_list[methods] 
            parameters['dbb'] = this_round_dbb
            try:
                this_round_dbb = action_code(**parameters)
                ## remove non-serializable objects
                move['parameters'] = {k: v for k, v in move['parameters'].items() if k != 'dbb'}
                self.history.append(json.dumps(move))
                this_round_dbb.method = method
                self.success += 1
                if self.mode == 'autog-a':
                    this_round_dbb = self.update_task(this_round_dbb)
                    self.backup(this_round_dbb)
                    ## if autog-a, update after every action, otherwise update once
                    
                this_round_dbb = self.update_task(this_round_dbb)
            except Exception as e:
                ## recover from error
                traceback.print_exc()
                typer.echo(f"Error: {e}")
                self.history.append("Error: " + str(e) + "Problem action: " + str(move))   
                this_round_dbb = last_valid_dbb
                self.error += 1
                if self.error >= 3:
                    return this_round_dbb, False 
        return this_round_dbb, True

    
    def get_current_state(self):
        """
            Get the current state
        """
        return self.state
    
    def manual_post_process(self, dbb):
        ## this is a manual post process function
        ## some datasets require time_column otherwise there will be leakage
        ## this has nothing with todo graph construction, only for evaluation purpose
        ## we put process here
        if self.dataset == 'outbrain':
            for i, table in enumerate(dbb.metadata.tables):
                if table.name == 'Event':
                    table.time_column = 'timestamp' 
                if table.name == 'PageView':
                    table.time_column = 'timestamp'
                if table.name == 'Click':
                    table.time_column = 'timestamp'
                if table.name == 'DocumentsMeta':
                    table.time_column = 'publish_time'
            dbb.metadata.tasks[0].time_column = 'timestamp'
        if self.dataset == 'avs':
            # dbb.tasks[0].train_set.pop('history_chain')
            # dbb.tasks[0].validation_set.pop('history_chain')
            # dbb.tasks[0].test_set.pop('history_chain')
            dbb.tasks[0].metadata.columns.append(
                DBBColumnSchema(
                    name='timestamp',
                    dtype='datetime',
                )
            )
            dbb.tasks[0].time_column = 'timestamp'
            remove_table = -1
            remove_col = -1
            for i, table in enumerate(dbb.metadata.tables):
                if table.name == 'History':
                    table.time_column = 'offerdate'
                if table.name == 'Transaction':
                    table.time_column = 'date'
                if table.name == 'History':
                    for j, col in enumerate(table.columns):
                        if col.name == 'repeater':
                            remove_table = i
                            remove_col = j
            if remove_table != -1 and remove_col != -1:
                dbb.metadata.tables[remove_table].columns.pop(remove_col)
            dbb.metadata.tasks[0].time_column = 'timestamp'
            ## for history table, need to pop the repeater column, otherwise it will be a leakage; why for mag we don't need to remove? didn't figure out yet

        if self.dataset == 'diginetica':
            #
            for i, table in enumerate(dbb.metadata.tables):
                if table.name == 'QueryResult':
                    table.time_column = 'timestamp'
                elif table.name == 'Click':
                    table.time_column = 'timestamp'
                elif table.name == 'View':
                    table.time_column = 'timestamp'
                elif table.name == 'Purchase':
                    table.time_column = 'timestamp'
                elif table.name == 'Query':
                    table.time_column = 'timestamp'
            if self.task_name == 'purchase':
                dbb.metadata.tasks[0].time_column = 'timestamp' 
                dbb.metadata.tasks[0].columns = [
                    DBBColumnSchema(
                        name='timestamp',
                        dtype='datetime',
                    ),
                    DBBColumnSchema(
                        name='purchase_session',
                        link_to = 'Session.sessionId',
                        dtype='foreign_key',
                    ),
                    DBBColumnSchema(
                        name='itemId',
                        link_to = 'Product.itemId',
                        dtype='foreign_key',
                    ),
                ]
            else:
                dbb.metadata.tasks[0].time_column = 'timestamp'
                dbb.metadata.tasks[0].columns = [
                    DBBColumnSchema(
                        name='timestamp',
                        dtype='datetime',
                    ),
                    DBBColumnSchema(
                        name='queryId',
                        link_to = 'Query.queryId',
                        dtype='foreign_key',
                    ),
                    DBBColumnSchema(
                        name='itemId',
                        link_to = 'Product.itemId',
                        dtype='foreign_key',
                    ),
                    DBBColumnSchema(name = 'clicked',
                                    dtype = 'category')
                ]
        return dbb

    def fix_pk_fk(self, dbb):
        """
            Since in connect_two_columns, we make it possible to turn a pk in a fk. However, link_to can only be made towards a pk. As a result, we need to check every link_to to make sure that they are point to a pk.
        """
        
        for table in dbb.metadata.tables:
            for column in table.columns:
                pass 

    def augment(self):
        """
            Augment the schema
        """
        for i in range(self.threshold):
            typer.echo(f"Round: {i} ...")
            ## generate the folder for round i
            self.dataset_cache_path = os.path.join(self.path_to_file, f"round_{i}")
            os.makedirs(self.dataset_cache_path, exist_ok=True)
            os.system(f"mkdir -p {self.dataset_cache_path}/data")
            os.system(f"mkdir -p {self.dataset_cache_path}/{self.task_name}")
            with open(os.path.join(self.dataset_cache_path, 'metadata.yaml'), 'w') as f:
                yaml.dump(self.state, f)
            if i == 0 and (not os.path.exists(os.path.join(self.dataset_cache_path, 'data')) or len(os.listdir(os.path.join(self.dataset_cache_path, 'data'))) == 0):
                # initial state, move the data from old to autog
                parent_dir = os.path.dirname(self.path_to_file)

                initial_data_path = os.path.join(parent_dir, 'data')
                initial_task_path = os.path.join(parent_dir, self.task_name)
                
                
                target_data_path = os.path.join(self.dataset_cache_path, 'data')
                # target_data_path = os.path.join(self.dataset_cache_path)
                target_task_path = os.path.join(self.dataset_cache_path, self.task_name)
                # target_task_path = os.path.join(self.dataset_cache_path)

                copy_directory(initial_data_path, target_data_path)
                copy_directory(initial_task_path, target_task_path)
                self.round += 1
                continue
            elif i == 0:
                self.round += 1
                continue
            if i == 1:
                dbb = load_dbb_dataset_from_cfg_path_no_name(os.path.join(self.path_to_file, "round_0"))
            res, need_continue = self.decide_next_step(dbb, i)
            self.round += 1
            if need_continue == False:
                # import ipdb; ipdb.set_trace()
                res = self.manual_post_process(res)
                if self.error < 3:
                    typer.echo("No more action can be taken")
                    # import ipdb; ipdb.set_trace()
                    res.save(os.path.join(self.path_to_file, 'final'))
                    ## plot the schema
                    plot_rdb_dataset_schema(res, os.path.join(self.path_to_file, 'final', 'schema'))
                    return
                elif self.error >= 3:
                    typer.echo("Too many errors, halt the process")
                    res.save(os.path.join(self.path_to_file, 'final'))
                    plot_rdb_dataset_schema(res, os.path.join(self.path_to_file, 'final', 'schema'))
                    return
            else:
                dbb = res
                time.sleep(self.llm_sleep)

        res.save(os.path.join(self.path_to_file, 'final'))
        ## plot the schema
        plot_rdb_dataset_schema(res, os.path.join(self.path_to_file, 'final', 'schema'))
