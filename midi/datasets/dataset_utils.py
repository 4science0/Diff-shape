import pickle
import numpy as np
from rdkit import Chem
import torch
import os
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
# from midi.datasets import utils
from midi.diffusion.distributions import DistributionNodes
from midi.utils import PlaceHolder
import torch.nn.functional as F

def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    centroid_pos = torch.mean(pos, dim=0, keepdim=True).numpy().astype(np.double)
    pos = pos - torch.mean(pos, dim=0, keepdim=True)#将原子坐标拉到质心：原子坐标-质心坐标
    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())        # TODO: check if implicit Hs should be kept

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    #control data
    # control_atom_types = (atom_types != 0).long()
    control_atom_types = torch.full_like(atom_types, atom_encoder['C'])
    control_charges = torch.zeros_like(all_charges)
    control_edge_attr = (edge_attr != 0).long()
    ##add noise
    # control_pos = pos+torch.normal(mean=0, std=0.01, size=(5,))
    if mol.GetProp('_Name'):
        id = mol.GetProp('_Name')
    else:
        id = mol.GetProp('id') if 'id' in mol.GetPropNames() else ''
    data = Data(x=atom_types, edge_index=edge_index, edge_attr=edge_attr, pos=pos, charges=all_charges, smiles=smiles,
                cx=control_atom_types, ccharges=control_charges, cedge_attr=control_edge_attr, id=id,
                centroid_pos=centroid_pos)

    return data


def mol_to_control_data(geometric_data):
    """
    geometric_data: result of the mol_to_torch_geometric
    """

    data = geometric_data.clone()
    data.x = (data.x != 0).long() #atom type is changed to C
    data.charges = torch.zeros_like(data.charges) #charge is changed to 0

    return data

def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(to_keep, data.edge_index, data.edge_attr, relabel_nodes=True,
                                             num_nodes=len(to_keep))
    centroid_pos = data.centroid_pos + torch.mean(data.pos[to_keep], dim=0, keepdim=True).numpy().astype(np.double)
    new_pos = data.pos[to_keep] - torch.mean(data.pos[to_keep], dim=0)
    new_cedge_attr = (new_edge_attr != 0).long()
    return Data(x=data.x[to_keep] - 1,
                edge_index=new_edge_index,# Shift onehot encoding to match atom decoder
                edge_attr=new_edge_attr,
                pos=new_pos,
                charges=data.charges[to_keep],
                smiles=data.smiles,
                cx=data.cx[to_keep]-1,
                ccharges=data.ccharges[to_keep],
                cedge_attr=new_cedge_attr,
                id=data.id,
                centroid_pos=centroid_pos
                )


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Statistics:
    def __init__(self, num_nodes, atom_types, bond_types, charge_types, valencies, bond_lengths, bond_angles):
        self.num_nodes = num_nodes
        print("NUM NODES IN STATISTICS", num_nodes)
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles

class MolInfos():
    def __init__(self, statistics_path, atom_encoder):
        self.atom_decoder = [key for key in atom_encoder.keys()]
        self.num_atom_types = len(self.atom_decoder)

        statistics = Statistics(num_nodes=load_pickle(os.path.join(statistics_path, 'train_n_h.pickle')),
                                                atom_types=torch.from_numpy(np.load(os.path.join(statistics_path, 'train_atom_types_h.npy'))),
                                                bond_types=torch.from_numpy(np.load(os.path.join(statistics_path, 'train_bond_types_h.npy'))),
                                                charge_types=torch.from_numpy(np.load(os.path.join(statistics_path, 'train_charges_h.npy'))),
                                                valencies=load_pickle(os.path.join(statistics_path, 'train_valency_h.pickle')),
                                                bond_lengths=load_pickle(os.path.join(statistics_path, 'train_bond_lengths_h.pickle')),
                                                bond_angles=torch.from_numpy(np.load(os.path.join(statistics_path, 'train_angles_h.npy'))))
        
        train_n_nodes = load_pickle(os.path.join(statistics_path, 'train_n_h.pickle'))
        val_n_nodes = load_pickle(os.path.join(statistics_path, 'val_n_h.pickle'))
        test_n_nodes = load_pickle(os.path.join(statistics_path, 'test_n_h.pickle'))
        max_n_nodes = max(max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys()))
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value


        self.statistics = statistics
        self.n_nodes = n_nodes / n_nodes.sum()
        self.atom_types = statistics.atom_types
        self.edge_types = statistics.bond_types
        self.charges_types = statistics.charge_types
        self.charges_marginals = (self.charges_types * self.atom_types[:, None]).sum(dim=0)
        self.valency_distribution = statistics.valencies
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=0, pos=3)
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()

    def to_one_hot(self, X, charges, E, node_mask, X_sup = None, just_control=False):
        x = X.clone()
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 2, num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
        pl = placeholder.mask(node_mask, just_control)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()