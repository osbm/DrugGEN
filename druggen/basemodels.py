

class BaseModel:
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_huggingface(cls, repo_id: str):
        '''
        Initialize the model from a huggingface repo
        '''
        raise NotImplementedError

    def save_huggingface(self, repo_id: str):
        '''
        Save the model to a huggingface repo
        '''
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, path: str):
        '''
        Initialize the model from a local checkpoint
        '''
        raise NotImplementedError

    def save_checkpoint(self, path):
        self.config.to_json(path)
        


    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError
