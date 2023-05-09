import json



class DrugGENConfig():
    def __init__(
        self,
        submodel: str="CrossLoss",
        act: str="relu",
        z_dim: int=16,
        max_atom: int=45,
        lambda_gp: float=1,
    ):
        '''
        
        Args:
            submodel: str, default="CrossLoss"
                Chose model subtype: Prot, CrossLoss, Ligand, RL, NoTarget

            z_dim: int, default=16
                Prior noise for the first GAN

        '''

        # sanity checks

        # submodel
        all_possible_submodels = ["Prot", "CrossLoss", "Ligand", "RL", "NoTarget"]
        assert submodel in all_possible_submodels, f"submodel must be one of {all_possible_submodels}"

        # act
        all_possible_acts = ["relu", "gelu", "tanh", "sigmoid"]
        assert act in all_possible_acts, f"act must be one of {all_possible_acts}"

        # z_dim
        assert z_dim > 0, f"z_dim must be positive, got {z_dim}"

        # max_atom
        assert max_atom > 0, f"max_atom must be positive, got {max_atom}"


        self.submodel = submodel
        self.act = act
        self.z_dim = z_dim
        self.max_atom = max_atom


    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)


if __name__ == '__main__':
    # test 1 save and load
    config = DrugGENConfig()
    config.save('config.json')

    config = DrugGENConfig.load('config.json')
    print(config.__dict__)



