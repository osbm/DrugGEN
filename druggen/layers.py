from torch import nn
import torch

class Attention_new(nn.Module):
    def __init__(self, dim, heads, act, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = 1./dim**0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        #self.attention_dropout = nn.Dropout(attention_dropout)

        self.d_k = dim // heads  
        self.heads = heads
        self.out_e = nn.Linear(dim,dim)
        self.out_n = nn.Linear(dim, dim)
        
        
    def forward(self, node, edge):
        b, n, c = node.shape
        
        
        q_embed = self.q(node).view(-1, n, self.heads, c//self.heads)
        k_embed = self.k(node).view(-1, n, self.heads, c//self.heads)
        v_embed = self.v(node).view(-1, n, self.heads, c//self.heads)
   
        e_embed = self.e(edge).view(-1, n, n, self.heads, c//self.heads)
        
        q_embed = q_embed.unsqueeze(2)
        k_embed = k_embed.unsqueeze(1)
        
        attn = q_embed * k_embed
        
        attn = attn/ math.sqrt(self.d_k)
        
     
        attn = attn * (e_embed + 1) * e_embed

        edge = self.out_e(attn.flatten(3))  
      
        attn = F.softmax(attn, dim=2)
        
        v_embed = v_embed.unsqueeze(1)
  
        v_embed = attn * v_embed
        
        v_embed = v_embed.sum(dim=2).flatten(2)
        
        node  = self.out_n(v_embed)
           
        return node, edge

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads,act, mlp_ratio=4, drop_rate=0., ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
   
        self.attn = Attention_new(dim, heads, act, drop_rate, drop_rate)
        self.ln3 = nn.LayerNorm(dim)
        self.ln4 = nn.LayerNorm(dim)
        self.mlp = MLP(act,dim,dim*mlp_ratio, dim, dropout=drop_rate)
        self.mlp2 = MLP(act,dim,dim*mlp_ratio, dim, dropout=drop_rate)
        self.ln5 = nn.LayerNorm(dim)
        self.ln6 = nn.LayerNorm(dim)

    def forward(self, x,y):
        x1 = self.ln1(x)
        x2,y1 = self.attn(x1,y)
        x2 = x1 + x2
        y2 = y1 + y
        x2 = self.ln3(x2)   
        y2 = self.ln4(y2)   
        
        x = self.ln5(x2 + self.mlp(x2))
        y = self.ln6(y2 + self.mlp2(y2))
        return x, y

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, act, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, act, mlp_ratio, drop_rate)
            for _ in range(depth)]
            )

    def forward(self, x,y):
        
        for Encoder_Block in self.Encoder_Blocks:
            x,  y = Encoder_Block(x,y)
            
        return x, y


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, z_dim, act: str, vertexes, edges, nodes, dropout, dim, depth, heads, mlp_ratio):
        '''
        Initialize the generator network.

        Args:
            z_dim (int): dimension of the latent space
            act (str): activation function
        '''
        super().__init__()
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
        self.features = vertexes * vertexes * edges + vertexes * nodes
        self.transformer_dim = vertexes * vertexes * dim + vertexes * dim
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
     

class simple_disc(nn.Module):
    def __init__(self, act, m_dim, vertexes, b_dim):
        super().__init__()
        if act == "relu":
            act = nn.ReLU()
        elif act == "leaky":
            act = nn.LeakyReLU()
        elif act == "sigmoid":
            act = nn.Sigmoid()
        elif act == "tanh":
            act = nn.Tanh()  
        features = vertexes * m_dim + vertexes * vertexes * b_dim 
        
        self.predictor = nn.Sequential(nn.Linear(features,256), act, nn.Linear(256,128), act, nn.Linear(128,64), act,
                                       nn.Linear(64,32), act, nn.Linear(32,16), act,
                                       nn.Linear(16,1))
    
    def forward(self, x):
        
        prediction = self.predictor(x)
        
        #prediction = F.softmax(prediction,dim=-1)
        
        return prediction
