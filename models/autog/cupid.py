from valentine import valentine_match
from dbinfer import DBBRDBDataset
from valentine.algorithms import Cupid, JaccardDistanceMatcher, DistributionBased
from valentine.algorithms.jaccard_distance import StringDistanceFunction
import pandas as pd


def cupid_matching(dataset: DBBRDBDataset, matcher_name = 'cupid'):
    """
        Traditional data profiling method 
        Very slow
    """
    table_pairs = []
    for table_name in dataset.tables.keys():
        for table_name2 in dataset.tables.keys():
            if table_name != table_name2:
                table_pairs.append((table_name, table_name2))
    result = []
    if matcher_name == 'cupid':
        matcher = Cupid()
    elif matcher_name == 'levenshtein':
        matcher = JaccardDistanceMatcher(distance_fun=StringDistanceFunction.Levenshtein)
    elif matcher_name == 'hamming':
        matcher = JaccardDistanceMatcher(distance_fun=StringDistanceFunction.Hamming)
    elif matcher_name == 'distribution':
        matcher = DistributionBased()
    for table_pair in table_pairs:
        table1_name, table2_name = table_pair
        table1 = dataset.tables[table1_name]
        table2 = dataset.tables[table2_name]
        table_pair_df = pd.DataFrame(table1).astype(str)
        table_pair_df2 = pd.DataFrame(table2).astype(str)
        matches = valentine_match(table_pair_df, table_pair_df2, matcher, df1_name=table1_name, df2_name=table2_name)
        result.append(matches)
    return result

