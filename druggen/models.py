import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TransformerEncoder, TransformerDecoder
from .config import DrugGENConfig

class Generator(nn.Module):
    """Generator network."""
    def __init__(
            self,
            z_dim: int,
            activation_function: str,
            vertexes: int,
            edges: int,
            nodes: int,
            dropout: int,
            dim: int,
            depth, heads, mlp_ratio, submodel):
        super(Generator, self).__init__()

        self.submodel = submodel
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio

        self.dropout = dropout
        self.z_dim = z_dim

        if act == "relu":
            act = nn.ReLU()
        elif act == "leaky":
            act = nn.LeakyReLU()
        elif act == "sigmoid":
            act = nn.Sigmoid()
        elif act == "tanh":
            act = nn.Tanh()
        self.features = vertexes**2 * edges + vertexes * nodes
        self.transformer_dim = vertexes**2 * dim + vertexes * dim
        self.pos_enc_dim = 5
        #self.pos_enc = nn.Linear(self.pos_enc_dim, self.dim)

        self.node_layers = nn.Sequential(nn.Linear(nodes, 64), act, nn.Linear(64,dim), act, nn.Dropout(self.dropout))
        self.edge_layers = nn.Sequential(nn.Linear(edges, 64), act, nn.Linear(64,dim), act, nn.Dropout(self.dropout))

        self.TransformerEncoder = TransformerEncoder(dim=self.dim, depth=self.depth, heads=self.heads, act = act,
                                                                    mlp_ratio=self.mlp_ratio, drop_rate=self.dropout)         

        self.readout_e = nn.Linear(self.dim, edges)
        self.readout_n = nn.Linear(self.dim, nodes)
        self.softmax = nn.Softmax(dim = -1) 
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def laplacian_positional_enc(self, adj):
        
        A = adj
        D = torch.diag(torch.count_nonzero(A, dim=-1))
        L = torch.eye(A.shape[0], device=A.device) - D * A * D
        
        EigVal, EigVec = torch.linalg.eig(L)
    
        idx = torch.argsort(torch.real(EigVal))
        EigVal, EigVec = EigVal[idx], torch.real(EigVec[:,idx])
        pos_enc = EigVec[:,1:self.pos_enc_dim + 1]
        
        return pos_enc

    def forward(self, z_e, z_n):
        b, n, c = z_n.shape
        _, _, _ , d = z_e.shape
        #random_mask_e = torch.randint(low=0,high=2,size=(b,n,n,d)).to(z_e.device).float()
        #random_mask_n = torch.randint(low=0,high=2,size=(b,n,c)).to(z_n.device).float()
        #z_e = F.relu(z_e - random_mask_e)
        #z_n = F.relu(z_n - random_mask_n)

        #mask = self._generate_square_subsequent_mask(self.vertexes).to(z_e.device)
        
        node = self.node_layers(z_n)
        
        edge = self.edge_layers(z_e)
        
        edge = (edge + edge.permute(0,2,1,3))/2
        
        #lap = [self.laplacian_positional_enc(torch.max(x,-1)[1]) for x in edge]
        
        #lap = torch.stack(lap).to(node.device)
        
        #pos_enc = self.pos_enc(lap)
        
        #node = node + pos_enc
        
        node, edge = self.TransformerEncoder(node,edge)

        node_sample = self.softmax(self.readout_n(node))
        
        edge_sample = self.softmax(self.readout_e(edge))
        
        return node, edge, node_sample, edge_sample
     
     
     
class Generator2(nn.Module):
    def __init__(self, dim, dec_dim, depth, heads, mlp_ratio, drop_rate,drugs_m_dim,drugs_b_dim,b_dim,m_dim, submodel):
        super().__init__()
        self.submodel = submodel
        self.depth = depth
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.heads = heads
        self.dropout_rate = drop_rate
        self.drugs_m_dim = drugs_m_dim
        self.drugs_b_dim = drugs_b_dim

        self.pos_enc_dim = 5
        
     
        if self.submodel == "Prot":
            self.prot_n = torch.nn.Linear(3822, 45)   ## exact dimension of protein features
            self.prot_e = torch.nn.Linear(298116, 2025) ## exact dimension of protein features
        
            self.protn_dim = torch.nn.Linear(1,dec_dim)
            self.prote_dim = torch.nn.Linear(1,dec_dim)
            
            
        self.mol_nodes = nn.Linear(dim, dec_dim)
        self.mol_edges = nn.Linear(dim, dec_dim)
        
        self.drug_nodes =  nn.Linear(self.drugs_m_dim, dec_dim)
        self.drug_edges =  nn.Linear(self.drugs_b_dim, dec_dim)
        
        self.TransformerDecoder = TransformerDecoder(dec_dim, depth, heads, mlp_ratio, drop_rate=0.)

        self.nodes_output_layer = nn.Linear(dec_dim, self.drugs_m_dim)
        self.edges_output_layer = nn.Linear(dec_dim, self.drugs_b_dim)
        self.softmax = nn.Softmax(dim = -1) 
    def laplacian_positional_enc(self, adj):
        
        A = adj
        D = torch.diag(torch.count_nonzero(A, dim=-1))
        L = torch.eye(A.shape[0], device=A.device) - D * A * D
        
        EigVal, EigVec = torch.linalg.eig(L)
    
        idx = torch.argsort(torch.real(EigVal))
        EigVal, EigVec = EigVal[idx], torch.real(EigVec[:,idx])
        pos_enc = EigVec[:,1:self.pos_enc_dim + 1]
        
        return pos_enc
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, edges_logits, nodes_logits ,akt1_adj,akt1_annot):
        
        edges_logits = self.mol_edges(edges_logits)
        nodes_logits = self.mol_nodes(nodes_logits)
        
        if self.submodel != "Prot":
            akt1_annot = self.drug_nodes(akt1_annot)
            akt1_adj = self.drug_edges(akt1_adj)
         
        else:
            akt1_adj = self.prote_dim(self.prot_e(akt1_adj).view(1,45,45,1))
            akt1_annot = self.protn_dim(self.prot_n(akt1_annot).view(1,45,1))       


        #lap = [self.laplacian_positional_enc(torch.max(x,-1)[1]) for x in drug_e]
        #lap = torch.stack(lap).to(drug_e.device)
        #pos_enc = self.pos_enc(lap)
        #drug_n = drug_n + pos_enc
                
        nodes_logits,akt1_annot, edges_logits, akt1_adj = self.TransformerDecoder(nodes_logits,akt1_annot,edges_logits,akt1_adj)
     
        edges_logits = self.edges_output_layer(edges_logits)
        nodes_logits = self.nodes_output_layer(nodes_logits)
        
        edges_logits = self.softmax(edges_logits)
        nodes_logits = self.softmax(nodes_logits)
 
        return edges_logits, nodes_logits


class MLPDisctiminator(nn.Module):
    """Simple discriminator with 3 linear layers"""
    def __init__(self, activation_func, m_dim, vertexes, b_dim):
        super().__init__()

        activation_funcs = {
            "relu": nn.ReLU(),
            "leaky": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }
        activation_func = activation_funcs[activation_func]

        features = vertexes * m_dim + vertexes * vertexes * b_dim 
        
        self.mlp = nn.Sequential(
            nn.Linear(features,256),
            activation_func,
            nn.Linear(256,128),
            activation_func,
            nn.Linear(128,64),
            activation_func,
            nn.Linear(64,32),
            activation_func,
            nn.Linear(32,16),
            activation_func,
            nn.Linear(16,1)
        )
    
    def forward(self, x):
        x = self.mlp(x)
        #prediction = F.softmax(prediction,dim=-1)        
        return x

"""class Discriminator(nn.Module):
  
    def __init__(self,deg,agg,sca,pna_in_ch,pna_out_ch,edge_dim,towers,pre_lay,post_lay,pna_layer_num, graph_add):
        super(Discriminator, self).__init__()
        self.degree = deg
        self.aggregators = agg
        self.scalers = sca
        self.pna_in_channels = pna_in_ch
        self.pna_out_channels = pna_out_ch
        self.edge_dimension = edge_dim
        self.towers = towers
        self.pre_layers_num = pre_lay
        self.post_layers_num = post_lay
        self.pna_layer_num = pna_layer_num
        self.graph_add = graph_add
        self.PNA_layer = PNA(deg=self.degree, agg =self.aggregators,sca = self.scalers,
                             pna_in_ch= self.pna_in_channels, pna_out_ch = self.pna_out_channels, edge_dim = self.edge_dimension,
                             towers = self.towers, pre_lay = self.pre_layers_num, post_lay = self.post_layers_num,
                             pna_layer_num = self.pna_layer_num, graph_add = self.graph_add)

    def forward(self, x, edge_index, edge_attr, batch, activation=None):

        h = self.PNA_layer(x, edge_index, edge_attr, batch)

        h = activation(h) if activation is not None else h
        
        return h"""

"""class Discriminator2(nn.Module):

    def __init__(self,deg,agg,sca,pna_in_ch,pna_out_ch,edge_dim,towers,pre_lay,post_lay,pna_layer_num, graph_add):
        super(Discriminator2, self).__init__()
        self.degree = deg
        self.aggregators = agg
        self.scalers = sca
        self.pna_in_channels = pna_in_ch
        self.pna_out_channels = pna_out_ch
        self.edge_dimension = edge_dim
        self.towers = towers
        self.pre_layers_num = pre_lay
        self.post_layers_num = post_lay
        self.pna_layer_num = pna_layer_num
        self.graph_add = graph_add
        self.PNA_layer = PNA(deg=self.degree, agg =self.aggregators,sca = self.scalers,
                             pna_in_ch= self.pna_in_channels, pna_out_ch = self.pna_out_channels, edge_dim = self.edge_dimension,
                             towers = self.towers, pre_lay = self.pre_layers_num, post_lay = self.post_layers_num,
                             pna_layer_num = self.pna_layer_num, graph_add = self.graph_add)

    def forward(self, x, edge_index, edge_attr, batch, activation=None):

        h = self.PNA_layer(x, edge_index, edge_attr, batch)

        h = activation(h) if activation is not None else h
        
        return h"""


    
"""class PNA_Net(nn.Module): # ???
    def __init__(self,deg):
        super().__init__()

      

        self.convs = nn.ModuleList()
        
        self.lin = nn.Linear(5, 128)
        for _ in range(1):
            conv = DenseGCNConv(128, 128, improved=False, bias=True)
            self.convs.append(conv)
            
        self.agg_layer = GraphAggregation(128, 128, 0, dropout=0.1)
        self.mlp = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(),
                              nn.Linear(32, 1))

    def forward(self, x, adj,mask=None):
        x = self.lin(x)
        
        for conv in self.convs:
            x = F.relu(conv(x, adj,mask=None))

        x = self.agg_layer(x,torch.tanh)
       
        return self.mlp(x) """    
    

class DrugGEN():
    # this model should be able to run fully with only using preprocessed configuration
    # and pretrained weights
    # TODO
    # finish the model and test it with 
    # hub upload download
    # hub load

    def __init__(self, config: DrugGENConfig):
        self.config = config


        self.g1 = Generator(config=config)
        self.g2 = Generator2(config=config)
        self.d1 = MLPDisctiminator(config=config)
        self.d2 = MLPDisctiminator(config=config)

    def save_model(self, path):
        torch.save(self.g1.state_dict(), path + "/g1.pth")
        torch.save(self.g2.state_dict(), path + "/g2.pth")
        torch.save(self.d1.state_dict(), path + "/d1.pth")
        torch.save(self.d2.state_dict(), path + "/d2.pth")

        self.config.save(path + "/config.json")
    
    def g1_loss(self, batch):
        ...

    def g2_loss(self, batch):
        ...

    def d1_loss(self, batch):
        ...

    def d2_loss(self, batch):
        ...
