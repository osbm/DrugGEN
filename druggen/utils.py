import torch
from rdkit import Chem
from typing import List, Tuple, Union


def check_valency(mol: Chem.Mol) -> Tuple[bool, Union[List[int], None]]:
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(x):
    xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        # else:
        assert len(atomid_valence) == 2
        idx = atomid_valence[0]
        v = atomid_valence[1]
        queue = []
        for b in mol.GetAtomWithIdx(idx).GetBonds():
            queue.append(
                (
                    b.GetIdx(),
                    int(b.GetBondType()),
                    b.GetBeginAtomIdx(),
                    b.GetEndAtomIdx(),
                )
            )
        queue.sort(key=lambda tup: tup[1], reverse=True)
        if len(queue) > 0:
            start = queue[0][2]
            end = queue[0][3]
            t = queue[0][1] - 1
            mol.RemoveBond(start, end)

            # if t >= 1:
            #     mol.AddBond(start, end, self.decoder_load('bond_decoders')[t])
            # if '.' in Chem.MolToSmiles(mol, isomericSmiles=True):
            #     mol.AddBond(start, end, self.decoder_load('bond_decoders')[t])
            #     print(tt)
            #     print(Chem.MolToSmiles(mol, isomericSmiles=True))

    return mol


def label2onehot(labels, dim):
    
    """Convert label indices to one-hot vectors."""
    
    out = torch.zeros(list(labels.size())+[dim])
    out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)
    
    return out.float()