from torch_geometric.data import (Data, download_url, Dataset)
from .. import utils

class BaseMoleculeDataset(Dataset):
    '''
    Base class for the molecule dataset
    '''
    def __init__(self, root, filename: str=None, huggingface_repo: str=None, add_features=False, transform=None, pre_transform=None, pre_filter=None):
        '''
        Initialize the dataset class.

        Args:
            root (str): Root directory where the dataset should be saved.

            filename (str): Name of the file to be downloaded from huggingface. If None, the
            raw_file_names function should be overriden.

            huggingface_repo (str): Huggingface dataset id. For example
            'HUBioDataLab/druggen-chembl'. If None, the download function should be overriden.

            add_features (bool): Whether to add features to the dataset. Defaults to False.

            transform (callable, optional): A function/transform that takes in an
            torch_geometric.data.Data object and returns a transformed version. The data
            object will be transformed before every access. (default: :obj:`None`)

            pre_transform (callable, optional): A function/transform that takes in an
            torch_geometric.data.Data object and returns a transformed version. The data
            object will be transformed before being saved to disk. (default: :obj:`None`)

            pre_filter (callable, optional): A function that takes in an
            torch_geometric.data.Data object and returns a boolean value, indicating whether
            the data object should be included in the final dataset. (default: :obj:`None`)
        '''
        self.filename = filename
        self.huggingface_repo = huggingface_repo
        self.add_features = add_features
        super().__init__(root, transform, pre_transform, pre_filter)
        

    @property
    def raw_file_names(self):
        '''
        Return the raw file names. If these names are not present, they will be automatically downloaded using download function of this class.
        '''
        return [self.filename]

    def download(self):
        '''
        Download the dataset from huggingface.
        '''
        if self.huggingface_repo is None:
            raise ValueError(f"Raw file not found in the: {self.root}{os.path.sep}{self.filename}."
            "Please provide the huggingface dataset id. For example 'HUBioDataLab/druggen-chembl'")

        download_url(f"https://huggingface.co/datasets/{self.huggingface_repo}/raw/main/{filename}", self.root)

    def processed_file_names(self):
        '''
        Return the processed file names. If these names are not present, they will be automatically processed using process function of this class.
        '''
        proccessed_filename_prefix = self.huggingface_repo.replace("/", "_") + self.filename

        
        return [self.huggingface_repo.replace("/", "_") + self.filename + ".pt"]

    def generate_adjacency_matrix(self, mol, connected=True, max_length=None):
        """
        Generates the adjacency matrix for a molecule.

        Args:
            mol (Molecule): The molecule object.
            connected (bool): Whether to check for connectivity in the molecule. Defaults to True.
            max_length (int): The maximum length of the adjacency matrix. Defaults to the number of atoms in the molecule.

        Returns:
            numpy.ndarray or None: The adjacency matrix if connected and all atoms have a degree greater than 0, 
            otherwise None.
        """
        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        A = np.zeros(shape=(max_length, max_length))
        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]
        A[begin, end] = bond_type
        A[end, begin] = bond_type
        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)
        return A if connected and (degree > 0).all() else None


    def generate_node_features(self, mol, max_length=None):
        """
        Generates the node features for a molecule.

        Args:
            mol (Molecule): The molecule object.
            max_length (int): The maximum length of the node features. Defaults to the number of atoms in the molecule.

        Returns:
            numpy.ndarray: The node features matrix.
        """
        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (max_length - mol.GetNumAtoms()))


    def generate_additional_features(self, mol, max_length=None):
        """
        Generates additional features for a molecule.

        Args:
            mol (Molecule): The molecule object.
            max_length (int): The maximum length of the additional features. Defaults to the number of atoms in the molecule.

        Returns:
            numpy.ndarray: The additional features matrix.
        """
        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        features = np.array([[*[a.GetDegree() == i for i in range(5)],
                            *[a.GetExplicitValence() == i for i in range(9)],
                            *[int(a.GetHybridization()) == i for i in range(1, 7)],
                            *[a.GetImplicitValence() == i for i in range(9)],
                            a.GetIsAromatic(),
                            a.GetNoImplicit(),
                            *[a.GetNumExplicitHs() == i for i in range(5)],
                            *[a.GetNumImplicitHs() == i for i in range(5)],
                            *[a.GetNumRadicalElectrons() == i for i in range(5)],
                            a.IsInRing(),
                            *[a.IsInRingSize(i) for i in range(2, 9)]] for a in mol.GetAtoms()], dtype=np.int32)
        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))


    def _generate_encoders_decoders(self):
        """
        Generates the encoders and decoders for the atoms and bonds.
        """
        
        # self.data = data
        print('Creating atoms encoder and decoder..')
        # atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        atom_labels = [0, 6, 7, 8, 9, 15, 16, 17, 53] # these are number of protons (a.k.a element nums, 6 for carbon, 52 for iodine etc.)
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        print(f'Created atoms encoder and decoder with {self.atom_num_types - 1} atom types and 1 PAD symbol!')
        print("atom_labels", atom_labels)
        print('Creating bonds encoder and decoder..')
        # bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
        #                                                             for mol in self.data
        #                                                             for bond in mol.GetBonds())))
        bond_labels = [
            Chem.rdchem.BondType.ZERO,
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]

        print("bond labels", bond_labels)
        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        print(f'Created bonds encoder and decoder with {self.bond_num_types - 1} bond types and 1 PAD symbol!')        
        #dataset_names = str(self.dataset_name)
  
        
    def process(self):
        '''
        Process the dataset. This function will be run if there is no processed file names.
        '''
        if self.raw_file_names[0].endswith('.smi'):
            smiles = pd.read_csv(osp.join(self.root, self.raw_file_names[0]), header=None)[0]
        elif self.raw_file_names[0].endswith('.csv'):
            smiles = pd.read_csv(osp.join(self.root, self.raw_file_names[0]))['smiles']
        else:
            raise ValueError('Unsupported file format. Only .smi and .csv are supported.')
        
        
        # mols = [Chem.MolFromSmiles(line) for line in open(osp.join(self.root, self.raw_file_names[0]), 'r').readlines()]
        # mols = list(filter(lambda x: x.GetNumAtoms() <= self.max_atom, mols))
        # mols = mols[:size]
        # indices = range(len(mols))
        indices = range(len(smiles))
        # self._generate_encoders_decoders(mols)
        self._generate_encoders_decoders()
        
  
    
        pbar = tqdm(total=len(indices), leave=True)
        pbar.set_description(f'Processing chembl dataset')
        # max_length = max(mol.GetNumAtoms() for mol in mols) # this is actually pretty good because if max atom number is not reached we wouldnt need bigger graph matrices
        data_list = []
      
        self.m_dim = len(self.atom_decoder_m)
        for idx in indices:
            # mol = mols[idx]
            smile = smiles[idx]
            mol = Chem.MolFromSmiles(smile)
            
            # filter by max atom size
            if mol.GetNumAtoms() > self.max_atom:
                continue
            
            A = self.generate_adjacency_matrix(mol, connected=True, max_length=self.max_atom)
            if A is not None:
                

                x = torch.from_numpy(self.generate_node_features(mol, max_length=self.max_atom)).to(torch.long).view(1, -1)
          
                x = utils.label2onehot(x,self.m_dim).squeeze()
                if self.add_features: 
                    f = torch.from_numpy(self.generate_additional_features(mol, max_length=self.max_atom)).to(torch.long).view(x.shape[0], -1)
                    x = torch.concat((x,f), dim=-1)
             
                adjacency = torch.from_numpy(A)
                
                edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adjacency[edge_index[0], edge_index[1]].to(torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data_list.append(data)
                pbar.update(1)

        pbar.close()

        torch.save(self.collate(data_list), osp.join(self.processed_dir, self.processed_file_names[0]))
