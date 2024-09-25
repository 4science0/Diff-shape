import os
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

import midi.datasets.dataset_utils as dataset_utils
from midi.datasets.abstract_dataset import AbstractDatasetInfos, AbstractAdaptiveDataModule
from midi.datasets.dataset_utils import load_pickle, save_pickle
from midi.metrics.molecular_metrics import filter_substructure
from midi.utils import PlaceHolder

full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}


class GeomDrugsDataset(InMemoryDataset):
    def __init__(self, split, root, dataset_cfg, transform=None, pre_transform=None, pre_filter=None,
                 val_template_num=200, test_template_num=200, filter=False):
        assert split in ['train', 'val', 'test']
        self.filter_smarts = [Chem.MolFromSmarts(subst) for subst in filter_substructure if Chem.MolFromSmarts(subst)]
        self.filter = filter
        self.split = split
        self.dataset_cfg = dataset_cfg
        self.remove_h = self.dataset_cfg.remove_h
        self.atom_encoder = full_atom_encoder
        self.template_name = self.dataset_cfg.template_name if hasattr(self.dataset_cfg, "template_name") else None
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
        self.val_template_num = val_template_num
        self.test_template_num = test_template_num
        self.control_data_dict = self.dataset_cfg.control_data_dict
        self.control_add_noise_dict = self.dataset_cfg.control_add_noise_dict

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistics = dataset_utils.Statistics(num_nodes=load_pickle(self.processed_paths[1]),
                                                   atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
                                                   bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
                                                   charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
                                                   valencies=load_pickle(self.processed_paths[5]),
                                                   bond_lengths=load_pickle(self.processed_paths[6]),
                                                   bond_angles=torch.from_numpy(np.load(self.processed_paths[7])))
        self.smiles = load_pickle(self.processed_paths[8])
        if len(self.processed_paths) > 9:
            self.template_data = torch.load(self.processed_paths[9])

        self.data.cx = self.data.cx if self.control_data_dict['cX'] == 'cX' else self.data.x
        self.data.ccharges = self.data.ccharges if self.control_data_dict['cX'] == 'cX' else self.data.charges
        if self.control_data_dict['cE'] == 'cE':
            self.data.cedge_attr = self.data.cedge_attr
        elif self.control_data_dict['cE'] == 'None':
            print(f"control_data_dict['cE']:{self.control_data_dict['cE']}")
            self.data.cedge_attr = torch.zeros_like(self.data.cedge_attr)
        elif self.control_data_dict['cE'] == 'E':
            self.data.cedge_attr = self.data.edge_attr
        elif self.control_data_dict['cE'] == 'single_mask_None':
            mask_tensor = torch.rand(self.data.cedge_attr.shape[0])>0.5
            self.data.cedge_attr[mask_tensor] = 0
        for i, _ in enumerate(self.template_data):
            self.template_data[i].cx = self.template_data[i].cx if self.control_data_dict['cX'] == 'cX' else self.template_data[i].x
            self.template_data[i].ccharges = self.template_data[i].ccharges if self.control_data_dict['cX'] == 'cX' else self.template_data[i].charges
            if self.control_data_dict['cE'] == 'cE':
                self.template_data[i].cedge_attr = self.template_data[i].cedge_attr
            elif self.control_data_dict['cE'] == 'None':
                self.template_data[i].cedge_attr = torch.zeros_like(self.template_data[i].cedge_attr)
            elif self.control_data_dict['cE'] == 'E':
                self.template_data[i].cedge_attr = self.template_data[i].edge_attr
            elif self.control_data_dict['cE'] == 'single_mask_None':
                mask_tensor = torch.rand(self.template_data[i].cedge_attr.shape[0]) > 0.5
                self.template_data[i].cedge_attr[mask_tensor] = 0


    @property
    def raw_file_names(self):
        if self.split == 'train':
            return ['train_data.txt']
        elif self.split == 'val':
            return ['val_data.txt']
        else:
            if self.template_name is None:
                return ['test_data.txt']
            else:
                return [f'test_data_{self.template_name}.txt']
    @property
    def processed_file_names(self):
        h = 'noh' if self.remove_h else 'h'
        if self.split == 'train':
            return [f'train_{h}.pt', f'train_n_{h}.pickle', f'train_atom_types_{h}.npy', f'train_bond_types_{h}.npy',
                    f'train_charges_{h}.npy', f'train_valency_{h}.pickle', f'train_bond_lengths_{h}.pickle',
                    f'train_angles_{h}.npy', f'train_smiles.pickle', f'train_template_{h}.pt']
        elif self.split == 'val':
            return [f'val_{h}.pt', f'val_n_{h}.pickle', f'val_atom_types_{h}.npy', f'val_bond_types_{h}.npy',
                    f'val_charges_{h}.npy', f'val_valency_{h}.pickle', f'val_bond_lengths_{h}.pickle',
                    f'val_angles_{h}.npy', f'val_smiles.pickle', f'val_template_{h}.pt']
        else:
            if self.template_name is None:
                return [f'test_{h}.pt', f'test_n_{h}.pickle', f'test_atom_types_{h}.npy', f'test_bond_types_{h}.npy',
                        f'test_charges_{h}.npy', f'test_valency_{h}.pickle', f'test_bond_lengths_{h}.pickle',
                        f'test_angles_{h}.npy', f'test_smiles.pickle', f'test_template_{h}.pt']
            else:
                return [f'test_{h}_{self.template_name}.pt', f'test_n_{h}_{self.template_name}.pickle',
                        f'test_atom_types_{h}_{self.template_name}.npy', f'test_bond_types_{h}_{self.template_name}.npy',
                        f'test_charges_{h}_{self.template_name}.npy', f'test_valency_{h}_{self.template_name}.pickle',
                        f'test_bond_lengths_{h}_{self.template_name}.pickle',f'test_angles_{h}_{self.template_name}.npy',
                        f'test_smiles_{self.template_name}.pickle', f'test_template_{h}_{self.template_name}.pt']

    def download(self):
        raise ValueError('Download and preprocessing is manual. If the data is already downloaded, '
                         f'check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}')

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        print(self.split, self.raw_paths[0])
        with open(self.raw_paths[0], 'r')as f:
            all_files = f.read().splitlines()

        all_smiles = []
        data_list = []
        for i, smiles in enumerate(tqdm(all_files)):
            if self.remove_h:
                # all_conformers = Chem.SDMolSupplier(f"{self.root}/raw/geom_drug_sdf/{smiles}.sdf")
                all_conformers = Chem.SDMolSupplier(f"/mnt/home/linjie/projects/diffusion_model/data/geom_drug_data/geom/raw/geom_drug_sdf/{smiles}.sdf")
            else:
                # all_conformers = Chem.SDMolSupplier(f"{self.root}/raw/geom_drug_sdf/{smiles}.sdf", removeHs=False)
                all_conformers = Chem.SDMolSupplier(
                    f"/mnt/home/linjie/projects/diffusion_model/data/geom_drug_data/geom/raw/geom_drug_sdf/{smiles}.sdf", removeHs=False)

            for j, conformer in enumerate(all_conformers):
                if j >= 5:
                    break
                if '.' in Chem.MolToSmiles(conformer):
                    print(smiles, 'have .')
                    break
                if conformer.GetNumHeavyAtoms() == 1:
                    print(smiles, 'GetNumHeavyAtoms=1')
                    break
                if self.filter:
                    match = any([conformer.HasSubstructMatch(subst) for subst in self.filter_smarts])
                    if match:
                        break

                data = dataset_utils.mol_to_torch_geometric(conformer, full_atom_encoder, smiles)
                if self.remove_h:
                    data = dataset_utils.remove_hydrogens(data)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            all_smiles.append(smiles)
        print(f"raw data:{len(all_files)}, processed data:{len(data_list)}")
        # torch.save(self.collate(data_list), self.processed_paths[0])
        # ###########
        template_num = self.test_template_num if self.split=='test' else self.val_template_num
        # random.seed(0)
        # template_data = random.sample(data_list, template_num)
        template_data = data_list[::5][:template_num]
        # template_data = data_list[:template_num]
        for i in range(len(template_data)):
            template_data[i].idx = str(i)

        torch.save(template_data, self.processed_paths[9])
        statistics = compute_all_statistics(data_list, self.atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2,
                                                                                       1: 3, 2: 4, 3: 5})
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        save_pickle(set(all_smiles), self.processed_paths[8])


class GeomDataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        train_dataset = GeomDrugsDataset(split='train', root=root_path, dataset_cfg=cfg.dataset, val_template_num=cfg.general.val_template_num, filter=cfg.general.filter_substructure)
        val_dataset = GeomDrugsDataset(split='val', root=root_path, dataset_cfg=cfg.dataset, val_template_num=cfg.general.val_template_num, filter=cfg.general.filter_substructure)
        test_dataset = GeomDrugsDataset(split='test', root=root_path, dataset_cfg=cfg.dataset, test_template_num=cfg.general.test_template_num, filter=cfg.general.filter_substructure)
        self.remove_h = cfg.dataset.remove_h
        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}
        super().__init__(cfg, train_dataset, val_dataset, test_dataset)


class GeomInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        self.statistics = datamodule.statistics
        self.name = 'geom'
        self.atom_encoder = full_atom_encoder
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().complete_infos(datamodule.statistics, self.atom_encoder)
        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=0, pos=3)

    def to_one_hot(self, X, charges, E, node_mask, just_control=False):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 2, num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
        pl = placeholder.mask(node_mask, just_control)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()
