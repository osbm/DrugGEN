


class BaseTrainer:
    def __init__(self, config):
        '''
        Base class for all trainers.
        '''
        self.config = config

    def train(self):
        '''
        Train the model
        '''
        raise NotImplementedError

    def log(self):
        '''
        Log the training process
        '''
        raise NotImplementedError

    def evaluate(self):
        '''
        Evaluate the model
        '''
        raise NotImplementedError

    def save(self):
        '''
        Save the model
        '''
        

    def save_to_huggingface(self):
        '''
        Save the model to huggingface
        '''
        raise NotImplementedError

    