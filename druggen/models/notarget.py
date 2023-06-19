import torch
import datasets
from torch_geometric.data import Data, download_url, Dataset, DataLoader
import os

from ..datasets import BaseMoleculeDataset
from .. import utils
from .. import layers
        
class DrugGENNoTargetDataset:
    '''
    Dataset class for the DrugGEN without target

    '''
    def __init__(
        self,
        data_folder=None,
        train_file_name: str="chembl_train.smi",
        test_file_name: str="chembl_test.smi",
        smiles_train_dataset: str="HUBioDataLab/DrugGEN-chembl-smiles",
        smiles_test_dataset: str="HUBioDataLab/DrugGEN-chembl-smiles",
        batch_size: int=32,
    ):
        
        self.smiles_train_dataset = BaseMoleculeDataset(root=os.path.join(data_folder, "smiles_train"), filename=train_file_name, huggingface_repo=smiles_train_dataset)
        self.smiles_test_dataset = BaseMoleculeDataset(root=os.path.join(data_folder, "smiles_test"), filename=test_file_name, huggingface_repo=smiles_test_dataset)

        self.smiles_train_dataloader = DataLoader(self.smiles_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.smiles_test_dataloader = DataLoader(self.smiles_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class DrugGENNoTargetConfig(BaseConfig):
    '''
    Config class for the DrugGEN without target
    '''
    def __init__(
        self,
        mlp_ratio: int = 4,
        transformer_depth: int = 1,
        generator_dropout: float = 0.0,
        # n_critic = 1, # unused
        z_dim: int = 16,
        act: str = "relu",
        vertexes: int = 9,
        b_dim: int = 5,
        m_dim: int = 8,
        dim: int = 128,
    ):
        self.mlp_ratio = mlp_ratio
        self.transformer_depth = transformer_depth
        self.generator_dropout = generator_dropout
        self.z_dim = z_dim
        self.act = act
        self.vertexes = vertexes
        self.b_dim = b_dim
        self.m_dim = m_dim
        self.dim = dim


class DrugGENNoTarget:
    '''
    Model class for the DrugGEN without target. This model consists of one GAN.
    It uses GNN as generator and MLP as discriminator.
    '''
    def __init__(self, config: DrugGENNoTargetConfig):
        self.config = config

        self.generator = Generator(
            config.z_dim,
            config.act,
            config.vertexes,
            config.b_dim,
            config.generator_dropout,
            dim=config.dim,
            depth=config.transformer_depth,
            mlp_ratio=config.mlp_ratio,
            submodel="NoTarget",
        )
        # self.discriminator = simple_disc(
        #     "tanh",

            
        # self.D2 = simple_disc("tanh", self.drugs_m_dim, self.drug_vertexes, self.drugs_b_dim)
        self.D = simple_disc("tanh", self.m_dim, self.vertexes, self.b_dim)
        self.V = simple_disc("tanh", self.m_dim, self.vertexes, self.b_dim)
        # self.V2 = simple_disc("tanh", self.drugs_m_dim, self.drug_vertexes, self.drugs_b_dim)


    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def discriminator_loss():
        ...

    def generator_loss():
        ...


class DrugGENNoTargetTrainerConfig(BaseConfig):
    def __init__(
        self,
        generator_optimizer_lr = 0.00001,
        discriminator_optimizer_lr = 0.00001,
        adam_optimizer_beta_1 = 0.9,
        adam_optimizer_beta_2 = 0.999,
        epochs = 20,
        # clipping_value
        # n_critic
        # init_type
        # discriminator_type
        huggingface_model_repo = "HUBioDataLab/DrugGEN-NoTarget",
    ):
        self.generator_optimizer_lr = generator_optimizer_lr
        self.discriminator_optimizer_lr = discriminator_optimizer_lr
        self.epochs = epochs
        self.huggingface_model_repo = huggingface_model_repo


    def check_sanity(self):
        '''
        Check if the config makes sense
        '''
        raise NotImplementedError


class DrugGENNoTargetTrainer(BaseTrainer):
    def __init__(self, trainer_config: DrugGENNoTargetTrainerConfig, model: DrugGENNoTarget):
        self.trainer_config = trainer_config
        self.model = model

    def train(self, dataset: DrugGENNoTargetDataset):
        '''
        Train the model
        '''
        # bulk data
        # epochs
        # log to tensorboard, wandb
        # save checkpoints, plot losses and metrics

    def save_checkpoint(self, path: str):
        '''
        Save the model checkpoint
        '''
        self.model.save_checkpoint(path)
        self.trainer_config.to_json(path)

        