from torch_frame.data import Dataset
from torch_frame.datasets import Yandex, TabularBenchmark
from torch_frame.datasets import BankMarketing

def get_pytorch_frame_dataset(dataset_name, data_root):
    mapper = {
        "diabetes": "Diabetes130US",
        "adult": "Adult"
    }
    if dataset_name in ["diabetes"]:
        return TabularBenchmark(data_root, mapper[dataset_name])

    elif dataset_name in ["bank"]:
        return BankMarketing(data_root)

