import os
from copy import deepcopy
from typing import Optional, Union, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops, to_scipy_sparse_matrix
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError,MetricCollection,KLDivergence
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
import wandb
from torch_geometric.data.batch import Batch
from midi.analysis.rdkit_functions import Molecule
# from dgd.ggg_utils_deps import approx_small_symeig, our_small_symeig,extract_canonical_k_eigenfeat
# from dgd.ggg_utils_deps import  ensure_tensor, get_laplacian, asserts_enabled

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def to_one_hot(X, charges, E, node_mask, just_control=False):
    num_atom_types = 16
    x = X.clone()
    X = F.one_hot(X, num_classes=num_atom_types).float()
    E = F.one_hot(E, num_classes=5).float()
    charges = F.one_hot(charges + 2, num_classes=6).float()
    placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
    pl = placeholder
    return pl.X, pl.charges, pl.E

class NoSyncMetricCollection(MetricCollection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) #disabling syncs since it messes up DDP sub-batching


class NoSyncMetric(Metric):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncKL(KLDivergence):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncMAE(MeanAbsoluteError):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching>>>>>>> main:utils.py

# Folders
def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs', exist_ok=True)
        os.makedirs('chains', exist_ok=True)
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name, exist_ok=True)
        os.makedirs('chains/' + args.general.name, exist_ok=True)
    except OSError:
        pass


def to_dense(data, dataset_info, device=None):
    X, node_mask = to_dense_batch(x=data.x, batch=data.batch)
    pos, _ = to_dense_batch(x=data.pos, batch=data.batch)
    pos = pos.float()
    assert pos.mean(dim=1).abs().max() < 1e-3
    charges, _ = to_dense_batch(x=data.charges, batch=data.batch)
    max_num_nodes = X.size(1)
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=data.batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)#torch.Size([64, 17, 17])
    if dataset_info is not None:
        X, charges, E = dataset_info.to_one_hot(X, charges=charges, E=E, node_mask=node_mask)
    else:
        X, charges, E = to_one_hot(X, charges=charges, E=E, node_mask=node_mask)
    

    cX, _ = to_dense_batch(x=data.cx, batch=data.batch)
    ccharges, _ = to_dense_batch(x=data.ccharges, batch=data.batch)
    cedge_index, cedge_attr = remove_self_loops(data.edge_index, data.cedge_attr)
    cE = to_dense_adj(edge_index=cedge_index, batch=data.batch, edge_attr=cedge_attr, max_num_nodes=max_num_nodes)
    cpos, _ = to_dense_batch(x=data.pos, batch=data.batch)
    cpos = cpos.float()


    if dataset_info is not None:
        cX, ccharges, cE = dataset_info.to_one_hot(X=cX, charges=ccharges, E=cE, node_mask=node_mask, just_control=True)
    else:
        cX, ccharges, cE = to_one_hot(X=cX, charges=ccharges, E=cE, node_mask=node_mask, just_control=True)

    y = X.new_zeros((X.shape[0], 0))
    cy = cX.new_zeros((cX.shape[0], 0))

    if device is not None:
        X = X.to(device)
        E = E.to(device)
        y = y.to(device)
        pos = pos.to(device)
        node_mask = node_mask.to(device)
        cX = cX.to(device)
        ccharges = ccharges.to(device)
        cE = cE.to(device)
        cy = cy.to(device)
        cpos = cpos.to(device)


    data = PlaceHolder(X=X, charges=charges, pos=pos, E=E, y=y, node_mask=node_mask,
                       cX=cX, ccharges=ccharges, cpos=cpos, cE=cE, cy=cy)
    return data.mask()


class PlaceHolder:
    def __init__(self, pos, X, charges, E, y, t_int=None, t=None, node_mask=None,
                 cpos=None, cX=None, ccharges=None, cE=None, cy=None, idx=None):
        self.pos = pos
        self.X = X
        self.charges = charges
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask
        self.cpos = cpos
        self.cX = cX
        self.ccharges = ccharges
        self.cE = cE
        self.cy = cy
        self.idx = idx


    def device_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.pos = self.pos.to(x.device) if self.pos is not None else None
        self.X = self.X.to(x.device) if self.X is not None else None
        self.charges = self.charges.to(x.device) if self.charges is not None else None
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None

        self.cpos = self.cpos.to(x.device) if self.cpos is not None else None
        self.cX = self.cX.to(x.device) if self.cX is not None else None
        self.ccharges = self.ccharges.to(x.device) if self.ccharges is not None else None
        self.cE = self.cE.to(x.device) if self.cE is not None else None
        self.cy = self.cy.to(x.device) if self.cy is not None else None
        self.idx = self.idx.to(x.device) if self.idx is not None else None

        return self

    def mask(self, node_mask=None, just_control=False):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        diag_mask = ~torch.eye(n, dtype=torch.bool,
                               device=node_mask.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1)  # bs, n, n, 1

        if self.X is not None:
            self.X = self.X * x_mask
        if self.charges is not None:
            self.charges = self.charges * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
        if self.pos is not None:
            self.pos = self.pos * x_mask
            self.pos = self.pos - self.pos.mean(dim=1, keepdim=True)
        if just_control==False:
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))

        # if self.cX is not None:
        #     self.cX = self.cX * x_mask
        # if self.ccharges is not None:
        #     self.ccharges = self.ccharges * x_mask
        # if self.cE is not None:
        #     self.cE = self.cE * e_mask1 * e_mask2 * diag_mask
        #     # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        # if self.cpos is not None:
        #     self.cpos = self.cpos * x_mask
        #     self.cpos = self.cpos - self.cpos.mean(dim=1, keepdim=True)

        return self

    def mask_no_mean(self, node_mask=None):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        node_mask = node_mask.to(self.X.device)
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        # diag_mask = ~torch.eye(n, dtype=torch.bool,
        #                        device=node_mask.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1)  # bs, n, n, 1

        if self.X is not None:
            self.X = self.X * x_mask
        if self.charges is not None:
            self.charges = self.charges * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2
        if self.pos is not None:
            self.pos = self.pos * x_mask
        return self
    
    def collapse(self, collapse_charges):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        copy.charges = collapse_charges.to(self.charges.device)[torch.argmax(self.charges, dim=-1)]
        copy.E = torch.argmax(self.E, dim=-1)
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = - 1
        copy.charges[self.node_mask == 0] = 1000
        copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1

        # copy.cX = torch.argmax(self.cX, dim=-1)
        # copy.ccharges = collapse_charges.to(self.ccharges.device)[torch.argmax(self.ccharges, dim=-1)]
        # copy.cE = torch.argmax(self.cE, dim=-1)
        # copy.cX[self.node_mask == 0] = - 1
        # copy.ccharges[self.node_mask == 0] = 1000
        # copy.cE[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1

        copy.cX = self.cX
        copy.ccharges = self.ccharges
        copy.cE = self.cE
        copy.idx = self.idx

        return copy

    def __repr__(self):
        return (f"pos: {self.pos.shape if type(self.pos) == torch.Tensor else self.pos} -- " +
                f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- " +
                f"charges: {self.charges.shape if type(self.charges) == torch.Tensor else self.charges} -- " +
                f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- " +
                f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y} -- " +
                f"cpos: {self.cpos.shape if type(self.cpos) == torch.Tensor else self.cpos} -- " +
                f"cX: {self.cX.shape if type(self.cX) == torch.Tensor else self.cX} -- " +
                f"ccharges: {self.ccharges.shape if type(self.ccharges) == torch.Tensor else self.ccharges} -- " +
                f"cE: {self.cE.shape if type(self.cE) == torch.Tensor else self.cE} -- " +
                f"cy: {self.cy.shape if type(self.cy) == torch.Tensor else self.cy}")


    def copy(self):
        return PlaceHolder(X=self.X, charges=self.charges, E=self.E, y=self.y, pos=self.pos, t_int=self.t_int, t=self.t,
                           node_mask=self.node_mask, cX=self.cX, ccharges=self.ccharges, cE=self.cE, cy=self.cy, cpos=self.cpos,
                           idx=self.idx)

    def mul_scales(self, scale):
        X = scale * self.X if self.X is not None else self.X
        charges = scale * self.charges if self.charges is not None else self.charges
        E = scale * self.E if self.E is not None else self.E
        y = scale * self.y if self.y is not None else self.y
        pos = scale * self.pos if self.pos is not None else self.pos
        return PlaceHolder(X=X, charges=charges, E=E, y=y, pos=pos, node_mask=self.node_mask)

    def minus_scales(self, feature, node_mask):
        X = self.X - feature.X if self.X is not None else self.X
        charges = self.charges - feature.charges if self.charges is not None else self.charges
        E = self.E - feature.E if self.E is not None else self.E
        y = self.y - feature.y if self.y is not None else self.y
        pos = self.pos - feature.pos if self.pos is not None else self.pos
        return PlaceHolder(X=X, charges=charges, E=E, y=y, pos=pos, node_mask=node_mask)

    def add_scales(self, feature, node_mask):
        X = self.X + feature.X if self.X is not None else self.X
        charges = self.charges + feature.charges if self.charges is not None else self.charges
        E = self.E + feature.E if self.E is not None else self.E
        y = self.y + feature.y if self.y is not None else self.y
        pos = self.pos + feature.pos if self.pos is not None else self.pos
        return PlaceHolder(X=X, charges=charges, E=E, y=y, pos=pos, node_mask=node_mask)
    
    def add(self, feature, node_mask=None):
        X = self.X + feature.X
        charges = self.charges + feature.charges
        E = self.E + feature.E
        y = self.y + feature.y
        pos = self.pos + feature.pos
        return PlaceHolder(X=X, charges=charges, E=E, y=y, pos=pos, t_int=self.t_int, t=self.t, node_mask=self.node_mask,
                           cX=feature.cX, ccharges=feature.ccharges, cE=feature.cE, cy=feature.cy, cpos=feature.cpos)


def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'MolDiffusion_{cfg.dataset["name"]}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb, 'resume': 'allow'}
    kwargs.update({'id': cfg.general.resume_id}) if hasattr(cfg.general, 'resume_id') else kwargs
    # print(f"wandb init kwargs:{kwargs}")
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


def remove_mean_with_mask(x, node_mask):
    """ x: bs x n x d.
        node_mask: bs x n """
    assert node_mask.dtype == torch.bool, f"Wrong type {node_mask.dtype}"
    node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (x * (~node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def mean_with_mask(x, node_mask):
    """ x: bs x n x d.
        node_mask: bs x n """
    assert node_mask.dtype == torch.bool, f"Wrong type {node_mask.dtype}"
    node_mask = node_mask.unsqueeze(-1)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    return mean * node_mask

def get_template_sdf(template_list, dataset_infos, template_name):
    template_batch = Batch.from_data_list(template_list)
    dense_data = to_dense(template_batch, dataset_infos)
    dense_data = dense_data.collapse(dataset_infos.collapse_charges)
    molecule_list, template_list_new = [], []
    count = 0
    if template_name is not None:
        if 'template' not in template_name:
            template_name = [template_name]
        else:
            template_name = [i.idx for i in template_list]
    else:
        template_name = list(range(len(template_list)))
    for i in range(len(template_list)):
        mol = Molecule(atom_types=dense_data.X[i], charges=dense_data.charges[i],
                       bond_types=dense_data.E[i], positions=dense_data.pos[i],
                       atom_decoder=dataset_infos.atom_decoder,
                       template_idx=template_name[i]).rdkit_mol
        if mol is not None:
            # mol.SetProp('template_idx', str(count))
            mol.SetProp('smiles', Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol))))
            molecule_list.append(mol)
            # template_list[i].idx = count
            template_list_new.append(template_list[i])
            count += 1
        else:
            print(f"template {i} is None")

    return molecule_list, template_list_new

def save_template(template_list, dataset_infos, sdf_filename):
    molecule_list, template_list_new = get_template_sdf(template_list, dataset_infos)
    with Chem.SDWriter(sdf_filename) as f:
        for i, (mol, data) in enumerate(zip(molecule_list, template_list_new)):
            if mol is not None:
                # mol.SetProp('smiles', Chem.MolToSmiles(mol))
                mol.SetProp('smiles', data.smiles)
                mol.SetProp('qed', str(QED.qed(mol)))
                mol.SetProp('id', data.id)
                f.write(mol)
    print('*'*10)

    return template_list_new


def get_template_sdf_new(template_list, dataset_infos):
    template_batch = Batch.from_data_list(template_list)
    dense_data = to_dense(template_batch, dataset_infos)
    dense_data_control = PlaceHolder(X=dense_data.cX, charges=dense_data.ccharges,
                                     pos=dense_data.cpos, E=dense_data.cE, y=dense_data.cy,
                                     node_mask=dense_data.node_mask)
    dense_data = dense_data_control.collapse(dataset_infos.collapse_charges)
    molecule_list = []
    for i in range(len(template_list)):
        molecule_list.append(Molecule(atom_types=dense_data.X[i], charges=dense_data.charges[i],
                                      bond_types=dense_data.E[i], positions=dense_data.pos[i],
                                      atom_decoder=dataset_infos.atom_decoder,
                                      template_idx=i).rdkit_mol)
    return molecule_list


def save_template_new(template_list, dataset_infos, sdf_filename):
    molecule_list = get_template_sdf_new(template_list, dataset_infos)
    with Chem.SDWriter(sdf_filename) as f:
        for i, mol in enumerate(molecule_list):
            mol.SetProp('smiles', Chem.MolToSmiles(mol))
            f.write(mol)