import os
from glob import glob
import numpy as np
from rdkit import Chem, RDLogger
import pickle
import torch
from torch_geometric.data.collate import collate
import midi.datasets.dataset_utils as dataset_utils
from midi.metrics.metrics_utils import compute_all_statistics
from torch import Tensor
from torch_geometric.data import Batch, Data
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)

def my_collate(
        data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
    r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
    to the internal storage format of
    :class:`~torch_geometric.data.InMemoryDataset`."""
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )

    return data, slices


remove_h = False
h = 'noh' if remove_h else 'h'
ligand_paths = ["./data/1z95/1z95_ligand.sdf"]
save_dir = './data/geom/processed'


template_data = []
all_smiles = []
for i, ligand_path in enumerate(ligand_paths):
    print(ligand_path)
    # ligand_name = ligand_path.rsplit('/', maxsplit=2)[1]
    ligand_name = (os.path.basename(ligand_path)).split('_ligand.sdf')[0]
    processed_paths = [f'test_{h}_{ligand_name}.pt', f'test_n_{h}_{ligand_name}.pickle', f'test_atom_types_{h}_{ligand_name}.npy', f'test_bond_types_{h}_{ligand_name}.npy',
                       f'test_charges_{h}_{ligand_name}.npy', f'test_valency_{h}_{ligand_name}.pickle', f'test_bond_lengths_{h}_{ligand_name}.pickle',
                       f'test_angles_{h}_{ligand_name}.npy', f'test_smiles_{ligand_name}.pickle', f'test_template_{h}_{ligand_name}.pt']
    ligand_mol = Chem.SDMolSupplier(ligand_path, removeHs=False)[0]
    ligand_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(ligand_mol)))
    print(ligand_smiles)
    # all_smiles.append(ligand_smiles)
    data = dataset_utils.mol_to_torch_geometric(ligand_mol, full_atom_encoder, ligand_smiles)
    if remove_h:
        data = dataset_utils.remove_hydrogens(data)
    data.idx = ligand_name
    template_data = [data]
    # template_data.append(data)
    # for i in range(len(template_data)):
    #     template_data[i].idx = i

    os.makedirs(save_dir)
    os.chdir(save_dir)
    torch.save(my_collate(template_data), processed_paths[0])
    torch.save(template_data, processed_paths[9])

    statistics = compute_all_statistics(template_data, full_atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5})
    save_pickle(statistics.num_nodes, processed_paths[1])
    np.save(processed_paths[2], statistics.atom_types)
    np.save(processed_paths[3], statistics.bond_types)
    np.save(processed_paths[4], statistics.charge_types)
    save_pickle(statistics.valencies, processed_paths[5])
    save_pickle(statistics.bond_lengths, processed_paths[6])
    np.save(processed_paths[7], statistics.bond_angles)
    save_pickle(set(all_smiles), processed_paths[8])

