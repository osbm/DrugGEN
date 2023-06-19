import json
import os

class BaseConfig:
    def __init__(self, kwarg1=None, kwarg2=None, kwarg3=None):
        self.kwarg1 = kwarg1
        self.kwarg2 = kwarg2
        self.kwarg3 = kwarg3

    @classmethod
    def from_dict(cls, config_dict):
        '''
        Initialize a config from a dictionary
        '''
        return cls(**config_dict)

    def to_dict(self):
        '''
        Dump the config as a dictionary
        '''
        return self.__dict__

    @classmethod
    def from_json(cls, path: str):
        '''
        Load a config from a json file

        Args:
            path (str): path to the json file
        '''
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json(self, path):
        '''
        Save a config to a json file

        Args:
            path (str): path to the json file
        '''
        from pathlib import Path
        path = Path(path)
        
        with open(path / "config.json", 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def __str__(self):
        '''
        Print the config as a string
        '''
        return str(self.to_dict())

    def __repr__(self):
        '''
        Print the config as a string
        '''
        return str(self.to_dict())

    def __eq__(self, other) -> bool:
        '''
        Dunder method for comparing two configs. Returns True if the two configs are the same.
        '''
        return self.to_dict() == other.to_dict()

