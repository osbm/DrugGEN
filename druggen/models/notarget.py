import os
import tempfile

import torch
import datasets
import numpy as np
from torch_geometric.data import Data, download_url, Dataset, DataLoader
import torch_geometric

from ..datasets import BaseMoleculeDataset
from .. import utils
from .. import layers

def load_data(
    data, # PyG dataloader for first gan 1 batch
    drugs, # PyG dataloader for first gan 1 batch
    batch_size, # int 
    device, # devie
    b_dim, # number of bonds
    m_dim, # number of atoms
    drugs_b_dim, # number of bonds for drugs
    drugs_m_dim, # number of atoms for the drugs
    z_dim, # Prior noise for the first GAN
    vertexes # Number of nodes in the graph
):

    z = sample_z(batch_size, z_dim) # np.random.normal(0,1, size=(batch_size,z_dim)) # (batch,max_len)

    z = torch.from_numpy(z).to(device).float().requires_grad_(True)
    data = data.to(device)
    drugs = drugs.to(device)                                               
    z_e = sample_z_edge(batch_size,vertexes,b_dim)                                                   # (batch,max_len,max_len)    
    z_n = sample_z_node(batch_size,vertexes,m_dim)                                                   # (batch,max_len)          
    z_edge = torch.from_numpy(z_e).to(device).float().requires_grad_(True)                                      # Edge noise.(batch,max_len,max_len)
    z_node = torch.from_numpy(z_n).to(device).float().requires_grad_(True)                                      # Node noise.(batch,max_len)       
    a = torch_geometric.utils.to_dense_adj(edge_index = data.edge_index,batch=data.batch,edge_attr=data.edge_attr, max_num_nodes=int(data.batch.shape[0]/batch_size)) 
    x = data.x.view(batch_size,int(data.batch.shape[0]/batch_size),-1)

    a_tensor = label2onehot(a, b_dim, device)
    #x_tensor = label2onehot(x, m_dim)
    x_tensor = x

    a_tensor = a_tensor #+ torch.randn([a_tensor.size(0), a_tensor.size(1), a_tensor.size(2),1], device=a_tensor.device) * noise_strength_0
    x_tensor = x_tensor #+ torch.randn([x_tensor.size(0), x_tensor.size(1),1], device=x_tensor.device) * noise_strength_1

    drugs_a = torch_geometric.utils.to_dense_adj(edge_index = drugs.edge_index,batch=drugs.batch,edge_attr=drugs.edge_attr, max_num_nodes=int(drugs.batch.shape[0]/batch_size))

    drugs_x = drugs.x.view(batch_size,int(drugs.batch.shape[0]/batch_size),-1)

    drugs_a = drugs_a.to(device).long() 
    drugs_x = drugs_x.to(device)
    drugs_a_tensor = label2onehot(drugs_a, drugs_b_dim,device).float()
    drugs_x_tensor = drugs_x

    drugs_a_tensor = drugs_a_tensor #+ torch.randn([drugs_a_tensor.size(0), drugs_a_tensor.size(1), drugs_a_tensor.size(2),1], device=drugs_a_tensor.device) * noise_strength_2
    drugs_x_tensor = drugs_x_tensor #+ torch.randn([drugs_x_tensor.size(0), drugs_x_tensor.size(1),1], device=drugs_x_tensor.device) * noise_strength_3

    a_tensor_vec = a_tensor.reshape(batch_size,-1)
    x_tensor_vec = x_tensor.reshape(batch_size,-1)               
    real_graphs = torch.concat((x_tensor_vec,a_tensor_vec),dim=-1)                      

    a_drug_vec = drugs_a_tensor.reshape(batch_size,-1)
    x_drug_vec = drugs_x_tensor.reshape(batch_size,-1)               
    drug_graphs = torch.concat((x_drug_vec,a_drug_vec),dim=-1)  

    return drug_graphs, real_graphs, a_tensor, x_tensor, drugs_a_tensor, drugs_x_tensor, z, z_edge, z_node

class NoTargetDataset:
    '''
    Dataset class for the DrugGEN without target

    '''
    def __init__(
        self,
        data_folder=None,
        smiles_train_file_name: str="chembl_train.smi",
        smiles_test_file_name: str="chembl_test.smi",
        smiles_train_dataset: str="HUBioDataLab/DrugGEN-chembl-smiles",
        smiles_test_dataset: str="HUBioDataLab/DrugGEN-chembl-smiles",
        drugs_train_file_name: str="akt_test.smi",
        drugs_test_file_name: str="akt_train.smi",
        drugs_train_dataset: str="osbm/akt_drugs",
        drugs_test_dataset: str="osbm/akt_drugs",
        batch_size: int=32,
    ):

        self.batch_size = batch_size

        if data_folder is None:
            data_folder = tempfile.mkdtemp()

        self.smiles_train_dataset = BaseMoleculeDataset(root=os.path.join(data_folder, "smiles_train"), filename=smiles_train_file_name, huggingface_repo=smiles_train_dataset)
        self.smiles_test_dataset = BaseMoleculeDataset(root=os.path.join(data_folder, "smiles_test"), filename=smiles_test_file_name, huggingface_repo=smiles_test_dataset)

        self.smiles_train_dataloader = DataLoader(self.smiles_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.smiles_test_dataloader = DataLoader(self.smiles_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.drugs_train_dataset = BaseMoleculeDataset(root=os.path.join(data_folder, "drugs_train"), filename=drugs_train_file_name, huggingface_repo=drugs_train_dataset)
        self.drugs_test_dataset = BaseMoleculeDataset(root=os.path.join(data_folder, "drugs_test"), filename=drugs_test_file_name, huggingface_repo=drugs_train_dataset)

        self.drugs_train_dataloader = DataLoader(self.drugs_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.drugs_test_dataloader = DataLoader(self.drugs_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class NoTargetConfig(BaseConfig):
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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NoTarget:
    '''
    Model class for the DrugGEN without target. This model consists of one GAN.
    It uses GNN as generator and MLP as discriminator.
    '''
    def __init__(self, config: NoTargetConfig):
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

        # self.D2 = simple_disc("tanh", self.drugs_m_dim, self.drug_vertexes, self.drugs_b_dim)
        self.discriminator = simple_disc("tanh", self.m_dim, self.vertexes, self.b_dim)
        # self.V2 = simple_disc("tanh", self.drugs_m_dim, self.drug_vertexes, self.drugs_b_dim)


    def discriminator_loss(
        self,
        mol_graph,
        batch_size,
        z_edge,
        z_node
    ):
            
        # Compute loss with real molecules.
        
        logits_real_disc = self.discriminator(mol_graph)              
        prediction_real =  - torch.mean(logits_real_disc)

        # Compute loss with fake molecules.
        node, edge, node_sample, edge_sample  = self.generator(z_edge,  z_node)

        # do we really need to do this this way?
        graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)
        
        logits_fake_disc = self.discriminator(graph.detach())

        prediction_fake = torch.mean(logits_fake_disc)
        
        # Compute gradient loss.
        
        eps = torch.rand(mol_graph.size(0),1).to(device)
        x_int0 = (eps * mol_graph + (1. - eps) * graph).requires_grad_(True)

        grad0 = self.discriminator(x_int0)
        d_loss_gp = utils.gradient_penalty(grad0, x_int0, self.config.device)
        
        # Calculate total loss
        
        d_loss = prediction_fake + prediction_real +  d_loss_gp * self.lambda_gp
        
        return node, edge, d_loss

    def generator_loss(adj, annot, batch_size, matrices2mol):
            
        # Compute loss with fake molecules.
        node, edge, node_sample, edge_sample  = self.generator(adj,  annot)
        graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1) # vectorize and concatonate

        logits_fake_disc = self.discriminator(graph)
        prediction_fake = - torch.mean(logits_fake_disc)
        # Produce molecules.

        g_edges_hat_sample = torch.max(edge_sample, -1)[1] 
        g_nodes_hat_sample = torch.max(node_sample , -1)[1]   
                    
        fake_mol = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) 
                for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]        
        g_loss = prediction_fake

        return g_loss, fake_mol, g_edges_hat_sample, g_nodes_hat_sample, node, edge

class NoTargetTrainerConfig(BaseConfig):
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


class NoTargetTrainer(BaseTrainer):
    def __init__(self, trainer_config: NoTargetTrainerConfig, model: NoTarget):
        self.trainer_config = trainer_config
        self.model = model

        # create optimizers
        self.generator_optimizer = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=self.trainer_config.generator_optimizer_lr,
            betas=(self.trainer_config.adam_optimizer_beta_1, self.trainer_config.adam_optimizer_beta_2)
        )

        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.trainer_config.discriminator_optimizer_lr,
            betas=(self.trainer_config.adam_optimizer_beta_1, self.trainer_config.adam_optimizer_beta_2)
        )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def train(self, dataset: NoTargetDataset):
        '''
        Train the model
        '''
        
        drugs_dataloader_iterator = iter(dataset.drugs_train_dataloader)
        
        vertexes = int(dataset.smiles_train_dataloader.dataset[0].x.shape[0]) # what does this mean?

        for epoch_index in range(self.trainer_config.epochs):
            for smiles_batch in dataset.smiles_train_dataloader:
                try:
                    drugs_batch = next(drugs_dataloader_iterator)
                except StopIteration:
                    drugs_dataloader_iterator = iter(self.drugs_loader)
                    drugs_batch = next(drugs_dataloader_iterator)


                # load bulk data
                bulk_data = load_data(
                    smiles_batch,
                    drugs_batch,
                    dataset.batch_size, 
                    self.device,
                    self.model.config.b_dim,
                    self.model.config.m_dim,
                    self.model.config.b_dim,
                    self.model.config.m_dim,
                    self.model.config.z_dim,
                    vertexes
                )

                
                drug_graphs, real_graphs, a_tensor, x_tensor, drugs_a_tensor, drugs_x_tensor, z, z_edge, z_node = bulk_data

                GAN1_input_e = z_edge
                GAN1_input_x = z_node
                GAN1_disc_e = a_tensor
                GAN1_disc_x = x_tensor
                

                loss = {}

                node, edge, d_loss = self.model.discriminator_loss( 
                    real_graphs, 
                    self.batch_size,
                    GAN1_input_e,
                    GAN1_input_x,
                )

                d_total = d_loss
                loss["d_total"] = d_total.item()


                d_total.backward()
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()

                # clean up
                generator_output = generator_loss(
                    self.G,
                    self.D,
                    self.V,
                    GAN1_input_e,
                    GAN1_input_x,
                    self.batch_size,
                    sim_reward,
                    self.dataset.matrices2mol_drugs,
                    fps_r,
                    self.submodel
                )

                g_loss, fake_mol, g_edges_hat_sample, g_nodes_hat_sample, node, edge = generator_output    






    def save_checkpoint(self, path: str):
        '''
        Save the model checkpoint
        '''
        self.model.save_checkpoint(path)
        self.trainer_config.to_json(path)

        