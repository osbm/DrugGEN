from statistics import mean
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
import pickle
import gzip
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NP_model = pickle.load(gzip.open('DrugGEN/data/NP_score.pkl.gz'))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open('DrugGEN/data/SA_score.pkl.gz')) for j in range(1, len(i))}
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')     
class MolecularMetrics(object):

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32).mean()

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data, mols)))

    @staticmethod
    def novel_filter(mols, data):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data, mols))

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data).mean()
    @staticmethod
    def novel_total_score_forsave(mols, data):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data)
    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(
            0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)

    @staticmethod
    def natural_product_scores(mols, norm=False):

        # calculating the score
        scores = [sum(NP_model.get(bit, 0)
                      for bit in Chem.rdMolDescriptors.GetMorganFingerprint(mol,
                                                                            2).GetNonzeroElements()) / float(
            mol.GetNumAtoms()) if mol is not None else None
                  for mol in mols]

        # preventing score explosion for exotic molecules
        scores = list(map(lambda score: score if score is None else (
            4 + math.log10(score - 4 + 1) if score > 4 else (
                -4 - math.log10(-4 - score + 1) if score < -4 else score)), scores))

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -3, 1), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(list(map(lambda x: 0 if x is None else x, [
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
            mols])))
    @staticmethod
    def quantitative_estimation_druglikeness_scores_forsave(mol, norm=False):
        return np.array([
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol))])
    @staticmethod
    def water_octanol_partition_coefficient_scores_forsave(mol, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol))]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores        
        
    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in mols]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
                 spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [MolecularMetrics._compute_SAS(mol) if mol is not None else None for mol in mols]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores

        return scores
    @staticmethod
    def synthetic_accessibility_score_scores_forsave(mol, norm=False):
        scores = [MolecularMetrics._compute_SAS(mol)]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):

        scores = (MolecularMetrics.constant_bump(
            MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True), 0.210,
            0.945) + MolecularMetrics.synthetic_accessibility_score_scores(mols,
                                                                           norm=True) + MolecularMetrics.novel_scores(
            mols, data) + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3) / 4

        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(condlist=[x <= x_low, x >= x_high],
                         choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                     np.exp(- (x - x_high) ** 2 / decay)],
                         default=np.ones_like(x))

    @staticmethod
    def tanimoto_sim_1v2(data1, data2):
        min_len = data1.size if data1.size > data2.size else data2
        sims = []
        for i in range(min_len):
            sim = DataStructs.FingerprintSimilarity(data1[i], data2[i])
            sims.append(sim)
        mean_sim = mean(sim)
        return mean_sim

    @staticmethod
    def mol_length(x):
        if x is not None:
            return  len([char for char in max(Chem.MolToSmiles(x).split(sep =".")).upper() if char.isalpha()])
        else:
            return 0
    
    @staticmethod
    def max_component(data, max_len):
        
        return (np.array(list(map(MolecularMetrics.mol_length, data)), dtype=np.float32)/max_len).mean()
        
        

def mols2grid_image(mols,path):
    mols = [e if e is not None else Chem.RWMol() for e in mols]
    
    for i in range(len(mols)):
        if MolecularMetrics.valid_lambda(mols[i]):
        #if Chem.MolToSmiles(mols[i]) != '':
            AllChem.Compute2DCoords(mols[i])
            Draw.MolToFile(mols[i], os.path.join(path,"{}.png".format(i+1)), size=(1200,1200)) 
        else:
            continue


def all_scores_val(mols, data, max_len, norm=False):

    m = {'valid score': MolecularMetrics.valid_total_score(mols) * 100,
          'unique score': MolecularMetrics.unique_total_score(mols) * 100,
          'novel score': MolecularMetrics.novel_total_score(mols, data) * 100,
          'max len': MolecularMetrics.max_component(mols,max_len) * 100}
     
    return m

def all_scores_chem(mols, data, max_len, norm=False):
    
    m = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'QED score': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'NP score': MolecularMetrics.natural_product_scores(mols, norm=norm),
        'drugcandidate score': MolecularMetrics.drugcandidate_scores(mols, data),
        'logP score': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
        'SA score': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm)}.items()}

    return m

         #   'diversity score': MolecularMetrics.diversity_scores(mols, data),
        #'drugcandidate score': MolecularMetrics.drugcandidate_scores(mols, data),
        #             'NP score': MolecularMetrics.natural_product_scores(mols, norm=norm),    
        # 'novel score': MolecularMetrics.novel_total_score(mols, data) * 100
        # 'drugcandidate score': MolecularMetrics.drugcandidate_scores(mols, data)
    return m0
def save_smiles_matrices(mols,edges_hard, nodes_hard,path,data_source = None): 
    mols = [e if e is not None else Chem.RWMol() for e in mols]
    
    for i in range(len(mols)):
        if MolecularMetrics.valid_lambda(mols[i]):
            #m0= all_scores_for_print(mols[i], data_source, norm=False)
        #if Chem.MolToSmiles(mols[i]) != '':
            save_path = os.path.join(path,"{}.txt".format(i+1))
            with open(save_path, "a") as f:
                np.savetxt(f, edges_hard[i].cpu().numpy(), header="edge matrix:\n",fmt='%1.2f')
                f.write("\n")
                np.savetxt(f, nodes_hard[i].cpu().numpy(), header="node matrix:\n", footer="\nsmiles:",fmt='%1.2f')
                f.write("\n")
                #f.write(m0)
                f.write("\n")
        

            print(Chem.MolToSmiles(mols[i]), file=open(save_path,"a"))
        else:
            continue
                
def reward(mols,data):
    
    ''' Rewards that can be used for Reinforcement Networks. '''
    
    rr = 1.
    #for m in ('logp,sas,qed,unique' if self.metrics == 'all' else self.metrics).split(','):
    m = ""
    if m == 'np':
        rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
    elif m == 'logp':
        rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
    elif m == 'sas':
        rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
    elif m == 'qed':
        rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
    elif m == 'novelty':
        rr *= MolecularMetrics.novel_scores(mols, data)
    elif m == 'dc':
        rr *= MolecularMetrics.drugcandidate_scores(mols, data)
    elif m == 'unique':
        rr *= MolecularMetrics.unique_scores(mols)
    elif m == 'diversity':
        rr *= MolecularMetrics.diversity_scores(mols, data)
    elif m == 'validity':
        rr *= MolecularMetrics.valid_total_score(mols)
    else:
        raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)

    
def dense_to_sparse_with_attr(self, adj):
    ### 
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])
        #index = torch.stack(index, dim=0)
    return index, edge_attr
"""
def _genDegree():
    
    ''' Generates the Degree distribution tensor for PNA, should be used everytime a different
        dataset is used.
        Can be called without arguments and saves the tensor for later use. If tensor was created
        before, it just loads the degree tensor.
        '''
    
    degree_path = os.path.join(self.degree_dir, self.dataset_name + '-degree.pt')
    if not os.path.exists(degree_path):
        
        
        max_degree = -1
        for data in self.dataset:
            d = geoutils.degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in self.dataset:
            d = geoutils.degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        torch.save(deg, 'DrugGEN/data/' + self.dataset_name + '-degree.pt')            
    else:    
        deg = torch.load(degree_path, map_location=lambda storage, loc: storage)
        
    return deg        
"""    

def plot_attn(self, attn_w, model, iter, epoch):
    
    cols = 4
    rows = int(self.heads/cols)

    fig, axes = plt.subplots( rows,cols, figsize = (30, 14))
    axes = axes.flat
    attentions_pos = attn_w[0]
    attentions_pos = attentions_pos.cpu().detach().numpy()
    for i,att in enumerate(attentions_pos):

        #im = axes[i].imshow(att, cmap='gray')
        sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i])
        axes[i].set_title(f'head - {i} ')
        axes[i].set_ylabel('layers')
    pltsavedir = "/home/atabey/attn/second"
    plt.savefig(os.path.join(pltsavedir, "attn" + model + "_" + self.dataset_name + "_"  + str(iter) + "_" + str(epoch) +  ".png"), dpi= 500,bbox_inches='tight')


def plot_grad_flow(named_parameters, model, iter, epoch):
    
    # Based on https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            print(p.grad,n)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=1) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    pltsavedir = "/home/atabey/gradients/tryout"
    plt.savefig(os.path.join(pltsavedir, "weights_" + model  + "_"  + str(iter) + "_" + str(epoch) +  ".png"), dpi= 500,bbox_inches='tight')

