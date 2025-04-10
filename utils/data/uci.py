from ucimlrepo import fetch_ucirepo 
from abc import abstractmethod

class UCIData:
    def __init__(self, uci_repo_id, cache_path = "./tmp") -> None:
        self.repo_id = uci_repo_id
        self.cache_path = cache_path
        self.data = self.download()
        self.tabular_data = self.data['data']['features']
        self.tabular_data_target = self.data['data']['target']
        


    def download(self):
        return fetch_ucirepo(id=self.repo_id)

    @abstractmethod
    def process(self):
        raise NotImplementedError
    
    @abstractmethod
    def manual_process(self):
        raise NotImplementedError


class UCIAdult(UCIData):
    def __init__(self) -> None:
        super().__init__("adult") 
    
    def process(self):
        pass
    
    def manual_process(self):
        pass

class UCIBank(UCIData):
    def __init__(self) -> None:
        super().__init__(uci_repo_id=222) 
    
    def process(self):
        pass
    
    def manual_process(self):
        pass