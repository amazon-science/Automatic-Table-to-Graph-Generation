"""
    A unified data interface for kaggle's tabular data
"""
import os
from abc import abstractmethod
import pandas as pd
from utils.data.ieee_cis import load_data, get_features_and_labels, get_relations_and_edgelist, create_homogeneous_edgelist
import logging
log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)

class KaggleData:
    def __init__(self, url, cache_path = "./tmp") -> None:
        self.url = url 
        self.cache_path = cache_path
        self.download()

    def download(self):
        os.system(f"kaggle competitions download -c {self.url} -p {self.cache_path}")

    @abstractmethod
    def process(self):
        raise NotImplementedError
    
    @abstractmethod
    def manual_process(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_up_table(self):
        """
            Set up a dict to store tables
        """
        ## return {"trains": <train_table, key-val pairs>, "tests": <test_table, key-val pairs>}
        raise NotImplementedError


class IEEECIS(KaggleData):
    def __init__(self) -> None:
        super().__init__("ieee-fraud-detection") 
    
    def process(self):
        zip_name = os.path.join(self.cache_path, "ieee-fraud-detection.zip")
        os.system(f"unzip -n {zip_name} -d {self.cache_path}")
    
    def set_up_table(self):
        transaction_file_path = os.path.join(self.cache_path, "train_transaction.csv")
        identity_file_path = os.path.join(self.cache_path, "train_identity.csv")
        transaction_file = pd.read_csv(transaction_file_path)
        identity_file = pd.read_csv(identity_file_path)

        self.tables = {
            "transaction": transaction_file,
            "identity": identity_file
        }

        



    def manual_process(self, homogeneous = False):
        transactions, identity, test_transactions = load_data(
            self.cache_path, "train_transaction.csv", "train_identity.csv", 0.8, self.cache_path
        )

        ## soji's graph construction, following "https://github.com/awslabs/sagemaker-graph-fraud-detection/blob/master/source/sagemaker/dgl-fraud-detection.ipynb"
        transaction_id_cols = "card1,card2,card3,card4,card5,card6,ProductCD,addr1,addr2,P_emaildomain,R_emaildomain"
        category_cols = "M1,M2,M3,M4,M5,M6,M7,M8,M9"
        get_features_and_labels(transactions, transaction_id_cols, category_cols, self.cache_path)
        relational_edges = get_relations_and_edgelist(transactions, identity, transaction_id_cols, self.cache_path)
        if homogeneous:
            create_homogeneous_edgelist(relational_edges, self.cache_path)








