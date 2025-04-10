from abc import ABC, abstractmethod


def get_oracle(strategy):
    """
        Get the oracle based on the strategy
    """
    if strategy == "vanilla_index":
        return VanillaIndexOracle()
    else:
        raise NotImplementedError("Strategy not implemented")


class Oracle(ABC):
    def __init__(self) -> None:
        self.scores = []
    
    @abstractmethod
    def append_schema(self, schema, score = None):
        pass
    
    def retrieve_best_ones(self):
        index_min = max(range(len(self.scores)), key=self.scores.__getitem__)
        return index_min
    

class VanillaIndexOracle(Oracle):
    """
        Vanilla oracle based on the index, always return the last one
    """
    
    def append_schema(self, schema, score = None):
        current_len = len(self.scores)
        self.scores.append(current_len)
    
    
    
class FullGraphOracle(Oracle):
    """
        Oracle based on the full graph
    """
    
    def append_schema(self, schema, score = None):
        self.scores.append(score)