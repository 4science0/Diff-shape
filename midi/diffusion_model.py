import itertools
import math
import os
import time
from collections import defaultdict
from glob import glob
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from rdkit import Chem
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader

import midi.metrics.abstract_metrics as custom_metrics
from midi import utils
from midi.analysis.rdkit_functions import Molecule
from midi.control_model import ControlNet, ControlGraphTransformer
from midi.diffusion import diffusion_utils
from midi.diffusion.diffusion_utils import sum_except_batch
from midi.diffusion.extra_features import ExtraFeatures
# print("RUNNING ABLATION")
from midi.diffusion.noise_model import DiscreteUniformTransition, MarginalUniformTransition
from midi.metrics.abstract_metrics import NLL
from midi.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMetrics
from midi.metrics.molecular_metrics import filter_substructure
from midi.metrics.train_metrics import TrainLoss

from midi.utils import PlaceHolder

from rdkit.Chem import QED
from typing import Union


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class GateResidue(torch.nn.Module):
    def __init__(self, input_dims: utils.PlaceHolder, full_gate:bool=True):
        super(GateResidue, self).__init__()
        self.input_dims = input_dims
        if full_gate:
            self.gate_X = torch.nn.Linear((input_dims.X + input_dims.charges) * 3, input_dims.X + input_dims.charges)
            self.gate_E = torch.nn.Linear(input_dims.E * 3, input_dims.E)
            self.gate_pos = torch.nn.Linear(input_dims.pos * 3, input_dims.pos)
            # self.gate_y = torch.nn.Linear(input_dims.y * 3, input_dims.y)
        else:
            self.gate_X = torch.nn.Linear(input_dims.X * 3, 1)
            self.gate_E = torch.nn.Linear(input_dims.E * 3, 1)
            self.gate_pos = torch.nn.Linear(input_dims.pos * 3, 1)
            # self.gate_y = torch.nn.Linear(input_dims.y * 3, 1)

    def forward(self, x, res):
        x_X_tmp = torch.cat((x.X, x.charges), dim=-1)#torch.Size([20, 64, 22])
        res_X_tmp = torch.cat((res.X, res.charges), dim=-1)
        g_X = self.gate_X(torch.cat((
            x_X_tmp,
            res_X_tmp,
            x_X_tmp - res_X_tmp), dim=-1)).sigmoid()#torch.Size([20, 64, 22])
        g_E = self.gate_E(torch.cat((x.E, res.E, x.E - res.E), dim=-1)).sigmoid()#x.E.shape:torch.Size([20, 64, 64, 5])
        g_pos = self.gate_pos(torch.cat((x.pos, res.pos, x.pos - res.pos), dim=-1)).sigmoid()#torch.Size([20, 64, 3])
        # g_y = self.gate_y(torch.cat((x.y, res.y, x.y - res.y), dim=-1)).sigmoid()


        X = x_X_tmp * g_X + res_X_tmp * (1 - g_X)
        E = x.E * g_E + res.E * (1 - g_E)
        pos = x.pos * g_pos + res.pos * (1 - g_pos)
        E = 1 / 2 * (E + torch.transpose(E, 1, 2))
        out = utils.PlaceHolder(X=X[..., :self.input_dims.X], charges=X[..., self.input_dims.X:],
                                E=E, pos=pos, y=res.y, node_mask=res.node_mask).mask()
        return out


class FullDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    train_iterations = None
    val_iterations = None

    def __init__(self, cfg, dataset_infos, train_smiles, val_template=None, test_template=None, test_smiles=None):
        super().__init__()
        nodes_dist = dataset_infos.nodes_dist
        self.filter_smarts = [Chem.MolFromSmarts(subst) for subst in filter_substructure if Chem.MolFromSmarts(subst)]
        self.filter = cfg.general.filter_substructure
        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps
        self.condition_control = cfg.model.condition_control if hasattr(self.cfg.model, "condition_control") else False
        self.only_last_control = cfg.model.only_last_control if hasattr(self.cfg.model, "only_last_control") else False
        self.guess_mode = cfg.model.guess_mode
        self.control_scales = [cfg.model.strength * (0.825 ** float(12 - i)) for i in range(13)]
        self.unconditional_guidance_scale = cfg.model.unconditional_guidance_scale
        self.control_data_dict = cfg.dataset.control_data_dict if hasattr(self.cfg.dataset, "control_data_dict") else {'cX': 'cX', 'cE': 'cE', 'cpos': 'cpos'}
        self.control_add_noise_dict = cfg.dataset.control_add_noise_dict if hasattr(self.cfg.dataset, "control_add_noise_dict") else {'cX': False, 'cE': False, 'cpos': False}
        self.add_gru_output_model = cfg.model.add_gru_output_model if hasattr(self.cfg.model, "add_gru_output_model") else False
        self.dropout_rate = cfg.model.dropout_rate if hasattr(self.cfg.model, "dropout_rate") else 0
        self.noise_std = cfg.model.noise_std if hasattr(self.cfg.model, "noise_std") else 0.3
        self.template_name = cfg.dataset.template_name if hasattr(self.cfg.dataset, "template_name") else None
        self.val_template = val_template
        self.test_template = test_template
        self.test_template_dict = {i.idx: i for i in self.test_template}
        self.node_dist = nodes_dist
        self.dataset_infos = dataset_infos
        self.extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        self.input_dims = self.extra_features.update_input_dims(dataset_infos.input_dims)
        self.output_dims = dataset_infos.output_dims
        # self.domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        # Train metrics
        self.train_loss = TrainLoss(lambda_train=self.cfg.model.lambda_train
                                     if hasattr(self.cfg.model, "lambda_train") else self.cfg.train.lambda0)
        self.train_metrics = TrainMolecularMetrics(dataset_infos)

        # Val Metrics
        self.val_metrics = torchmetrics.MetricCollection([custom_metrics.PosMSE(), custom_metrics.XKl(),
                                                          custom_metrics.ChargesKl(), custom_metrics.EKl()])
        self.val_nll = NLL()
        self.val_sampling_metrics = SamplingMetrics(train_smiles, dataset_infos, test=False, template=self.val_template, test_smiles=test_smiles, filter=self.filter)

        # Test metrics
        self.test_metrics = torchmetrics.MetricCollection([custom_metrics.PosMSE(), custom_metrics.XKl(),
                                                           custom_metrics.ChargesKl(), custom_metrics.EKl()])
        self.test_nll = NLL()
        self.test_sampling_metrics = SamplingMetrics(train_smiles, dataset_infos, test=True, template=self.test_template,
                                                     template_name=self.template_name, test_smiles=test_smiles, filter=self.filter)

        self.save_hyperparameters(ignore=['train_metrics', 'val_sampling_metrics', 'test_sampling_metrics',
                                          'dataset_infos', 'train_smiles'])

        self.control_model = ControlNet(
            input_dims=self.input_dims,
            n_layers=cfg.model.n_layers,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            dropout_rate=self.dropout_rate
        )
        self.model = ControlGraphTransformer(input_dims=self.input_dims,
                                             n_layers=cfg.model.n_layers,
                                             hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                             hidden_dims=cfg.model.hidden_dims,
                                             output_dims=self.output_dims)

        if self.add_gru_output_model:
            self.output_model = GateResidue(input_dims=self.output_dims, full_gate=True)

        self.instantiate_model_stage()

        if cfg.model.transition == 'uniform':
            self.noise_model = DiscreteUniformTransition(output_dims=self.output_dims,
                                                         cfg=cfg)
        elif cfg.model.transition == 'marginal':
            print(f"Marginal distribution of the classes: nodes: {self.dataset_infos.atom_types} --"
                  f" edges: {self.dataset_infos.edge_types} -- charges: {self.dataset_infos.charges_marginals}")

            self.noise_model = MarginalUniformTransition(x_marginals=self.dataset_infos.atom_types,
                                                         e_marginals=self.dataset_infos.edge_types,
                                                         charges_marginals=self.dataset_infos.charges_marginals,
                                                         y_classes=self.output_dims.y,
                                                         cfg=cfg)
        else:
            assert ValueError(f"Transition type '{cfg.model.transition}' not implemented.")

        self.log_every_steps = cfg.general.log_every_steps

    def instantiate_model_stage(self):
        self.model = self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False

    def on_train_epoch_end(self) -> None:
        self.print(f"Train epoch {self.current_epoch} ends, data_size={self.local_rank}:{self.local_rank_data_size}")
        tle_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch} finished: epoch loss: {tle_log['train_epoch/epoch_loss'] :.5f} -- "
                   f"pos: {tle_log['train_epoch/pos_mse'] :.5f} -- "
                   f"X: {tle_log['train_epoch/x_CE'] :.5f} --"
                   f" charges: {tle_log['train_epoch/charges_CE']:.5f} --"
                   f" E: {tle_log['train_epoch/E_CE'] :.5f} --"
                   f" y: {tle_log['train_epoch/y_CE'] :.5f} -- {time.time() - self.start_epoch_time:.2f}s ")
        self.log_dict(tle_log, batch_size=self.BS)
        # if self.local_rank == 0:
        tme_log = self.train_metrics.log_epoch_metrics(self.current_epoch, self.local_rank)
        if tme_log is not None:
            self.log_dict(tme_log, batch_size=self.BS)
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    def on_train_epoch_start(self) -> None:
        self.print("Starting epoch", self.current_epoch)
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        self.local_rank_data_size = 0


    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data, self.noise_std)
        self.local_rank_data_size += dense_data.X.shape[0]
        # print(f"local_rank {self.local_rank}, batch{i}: {z_t.X.shape}")
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        loss, tl_log_dict = self.train_loss(masked_pred=pred, masked_true=dense_data,
                                            log=i % self.log_every_steps == 0)

        # if self.local_rank == 0:
        tm_log_dict = self.train_metrics(masked_pred=pred, masked_true=dense_data,
                                         log=i % self.log_every_steps == 0)
        if tl_log_dict is not None:
            self.log_dict(tl_log_dict, batch_size=self.BS)
        if tm_log_dict is not None:
            self.log_dict(tm_log_dict, batch_size=self.BS)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_metrics.reset()

    def validation_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data, self.noise_std)
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        nll, log_dict = self.compute_val_loss(pred, z_t, clean_data=dense_data, test=False)
        return {'loss': nll}, log_dict

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_metrics.compute()]
        log_dict = {"val/epoch_NLL": metrics[0],
                    "val/pos_mse": metrics[1]['PosMSE'] * self.T,
                    "val/X_kl": metrics[1]['XKl'] * self.T,
                    "val/E_kl": metrics[1]['EKl'] * self.T,
                    "val/charges_kl": metrics[1]['ChargesKl'] * self.T}
        self.log_dict(log_dict, on_epoch=True, on_step=False, sync_dist=True)
        if wandb.run:
            wandb.log(log_dict)

        print_str = []
        for key, val in log_dict.items():
            new_val = f"{val:.2f}"
            print_str.append(f"{key}: {new_val} -- ")
        print_str = ''.join(print_str)
        print(f"Epoch {self.current_epoch}: {print_str}."[:-4])

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))


        self.val_counter += 1
        if self.name == "debug" or (self.val_counter % self.cfg.general.sample_every_val == 0 and
                                    self.current_epoch > 0):
            self.print(f"Sampling start")
            start = time.time()
            samples = self.sample_n_graphs(samples_to_generate=self.cfg.general.samples_to_generate,
                                           test=False)
            print(f'Done on {self.local_rank}. Sampling took {time.time() - start:.2f} seconds\n')
            print(f"Computing sampling metrics on {self.local_rank}...")
            self.val_sampling_metrics(samples, self.name, self.current_epoch, self.local_rank)
        self.print(f"Val epoch {self.current_epoch} ends")

    def on_test_epoch_start(self):
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
        self.test_nll.reset()
        self.test_metrics.reset()

    def test_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data, self.noise_std)
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        nll, log_dict = self.compute_val_loss(pred, z_t, clean_data=dense_data, test=True)
        return {'loss': nll}, log_dict

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_metrics.compute()]
        test_nll = metrics[0]
        print(f'Test loss: {test_nll :.4f}')
        log_dict = {"test/epoch_NLL": metrics[0],
                    "test/pos_mse": metrics[1]['PosMSE'] * self.T,
                    "test/X_kl": metrics[1]['XKl'] * self.T,
                    "test/E_kl": metrics[1]['EKl'] * self.T,
                    "test/charges_kl": metrics[1]['ChargesKl'] * self.T}
        self.log_dict(log_dict, sync_dist=True)

        print_str = []
        for key, val in log_dict.items():
            new_val = f"{val:.4f}"
            print_str.append(f"{key}: {new_val} -- ")
        print_str = ''.join(print_str)
        print(f"Epoch {self.current_epoch}: {print_str}."[:-4])

        if wandb.run:
            wandb.log(log_dict)


        column_names = ['template_idx', 'sample_num_count', 'Validity', 'Uniqueness',
                        'Validity*Uniqueness', 'validity_num', 'uniqueness_num', 'Novelty',
                        'Connected_Components', 'mol_stable', 'atom_stable']
        all_template_start_time = time.time()
        run_time_list = []
        if self.cfg.general.sample_type == "mixed_sample":
            print(f"Sampling start on GR{self.global_rank}")
            samples = self.sample_n_graphs(samples_to_generate=self.cfg.general.samples_to_generate,
                                           test=True)
            print("Computing sampling metrics...")
            self.test_sampling_metrics(samples, self.name, self.current_epoch, self.local_rank, sample_type="mixed_sample")
            run_time_list = [time.time() - all_template_start_time]
            print(f'Done. Sampling took {time.time() - all_template_start_time:.2f} seconds\n')
            print(f"Test ends.")
            column_names.pop(0)

        elif self.cfg.general.sample_type == "separate_no_while":
            for template in self.test_template:
                template_idx = template.idx
                print(f"template{template_idx} Sampling start on GR{self.global_rank}")
                start = time.time()
                template = [template]
                samples = self.sample_n_graphs(samples_to_generate=self.cfg.general.samples_to_generate,
                                               test=True, template=template)
                print("Computing sampling metrics...")
                self.test_sampling_metrics(samples, self.name, self.current_epoch, self.local_rank,
                                           sample_type="separate_no_while")
                if self.test_sampling_metrics.validity_flag == True:
                    template_end_time = round(time.time() - start, 2)
                    run_time_list.append(template_end_time)
                    print(f'Done. {template_idx} Sampling took {template_end_time:.2f} seconds\n')
                print('*' * 20)

        elif self.cfg.general.sample_type == "separate_while":
            for template in self.test_template:
                template_idx = template.idx
                print(f"template{template_idx} Sampling start on GR{self.global_rank}")
                start = time.time()
                template = [template]
                self.uniqueness_dict = {str(i.idx): defaultdict(dict) for i in template}
                all_samples = []
                sample_while_count = 0
                samples_to_generate = self.cfg.general.samples_to_generate
                samples_to_generate = 10 if samples_to_generate < 10 else samples_to_generate
                while len(template) > 0:
                    sample_batch_size = self.cfg.train.reference_batch_size \
                        if self.cfg.dataset.adaptive_loader else math.ceil(8 * self.cfg.train.batch_size)
                    samples_to_generate = (samples_to_generate//sample_batch_size+1)*sample_batch_size
                    samples = self.sample_n_graphs(samples_to_generate=samples_to_generate,
                                                   test=True, template=template,
                                                   sample_batch_size=sample_batch_size,
                                                   )
                    template, len_dict = self.check_sample_count(self.uniqueness_dict, resample_idx=template_idx)
                    print(f"sample template: {len_dict}")
                    all_samples.extend(samples)
                    sample_while_count += 1
                    samples_to_generate = min(sample_batch_size, self.cfg.general.samples_to_generate)
                    samples_to_generate = 10 if samples_to_generate<10 else samples_to_generate
                    if sample_while_count > 100:
                        break
                print(f"sample_batch_count:{sample_while_count}")
                print("Computing sampling metrics...")
                self.test_sampling_metrics(all_samples, self.name, self.current_epoch, self.local_rank, sample_type="separate_while")
                if self.test_sampling_metrics.validity_flag == True:
                    template_end_time = round(time.time() - start, 2)
                    run_time_list.append(template_end_time)
                    print(f'Done. {template_idx} Sampling took {template_end_time:.2f} seconds\n')
                print('*' * 20)

        result_metric_path = os.path.join(os.getcwd(), f"test/epoch{self.current_epoch}/result_metric.csv")
        if os.path.exists(result_metric_path):
            result_metric_df = pd.read_csv(result_metric_path, header=None, names=column_names)
            result_metric_df['run_time'] = run_time_list
            result_metric_df.to_csv(result_metric_path, index=False, float_format='%.2f')
        print(f"all template Sampling took {time.time() - all_template_start_time:.2f} seconds\n")
        print(f"Test ends.")



    def sample_n_graphs(self, samples_to_generate: int, test: bool, template: list=None, sample_batch_size: int=None):
        potential_ebs = self.cfg.train.reference_batch_size \
            if self.cfg.dataset.adaptive_loader else math.ceil(8 * self.cfg.train.batch_size)
        print(f"potential_ebs:{potential_ebs}")
        sample_batch_size = potential_ebs if potential_ebs is None else potential_ebs

        if samples_to_generate <= 0:
            return []
        samples = []
        if test:
            template = self.test_template if template is None else template
        else:
            template = self.val_template
        print(f"{len(template)} template, sample {samples_to_generate}")
        template = Batch.from_data_list(sum([list(itertools.repeat(i, samples_to_generate)) for i in template], []))
        template_loader = DataLoader(template, sample_batch_size, shuffle=True)
        for i, template_batch in enumerate(template_loader):
            template_batch = template_batch.cuda(self.device)
            dense_data = utils.to_dense(template_batch, self.dataset_infos)
            dense_data.idx = template_batch.idx
            current_n_list = torch.unique(template_batch.batch, return_counts=True)[1]
            n_nodes = current_n_list
            samples.extend(self.sample_batch(n_nodes=n_nodes, batch_id=i, template=dense_data,
                                             save_final=len(current_n_list),
                                             test=test))
            if self.cfg.general.sample_type == "separate_while" and test==True:
                self.uniqueness_dict = self.compute_uniqueness_dict(samples, self.uniqueness_dict)
                resample_template, len_dict = self.check_sample_count(self.uniqueness_dict, resample_idx=template.idx[0].item())
                if len(resample_template)==0:
                    break

        return samples

    def check_sample_count(self, uniqueness_dict, resample_idx=None):
        resample_dict = {}
        len_dict = {}
        for idx, sample_dict in uniqueness_dict.items():
            len_dict.update({idx: len(sample_dict)})
            if len(sample_dict) < self.cfg.general.samples_to_generate:
                resample_dict.update({idx: self.test_template_dict.get(idx)})

        if resample_idx is None:
            template_list = list(resample_dict.values())
        else:
            template_list = [resample_dict.get(resample_idx)] if resample_idx in resample_dict else []

        return template_list, len_dict

    def compute_uniqueness_dict(self, samples, uniqueness_dict):
        for mol in samples:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    Chem.SanitizeMol(largest_mol)
                    if self.filter:
                        match = any([largest_mol.HasSubstructMatch(subst) for subst in self.filter_smarts])
                        if match:
                            continue
                    smiles = Chem.MolToSmiles(largest_mol)
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                    if smiles not in uniqueness_dict[rdmol.GetProp('template_idx')].keys():
                        uniqueness_dict[rdmol.GetProp('template_idx')][smiles] = largest_mol
                except:
                    continue
        return uniqueness_dict

    @torch.no_grad()
    def sample_batch(self, n_nodes: list, batch_id: int = 0,
                     save_final: int = 0, test: bool = True, template: PlaceHolder = None):
        """
        :param batch_id: int
        :param n_nodes: list of int containing the number of nodes to sample for each graph
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        print(f"Sampling a batch with {len(n_nodes)} graphs.")
        assert save_final >= 0
        n_nodes = torch.Tensor(n_nodes).long().to(self.device)
        batch_size = len(n_nodes)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask, template=template, noise_std=self.noise_std)

        assert (z_T.E == torch.transpose(z_T.E, 1, 2)).all()

        # n_max = z_T.X.size(1)
        z_t = z_T
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T, 1 if test else self.cfg.general.faster_sampling)):
            s_array = s_int * torch.ones((batch_size, 1), dtype=torch.long, device=z_t.X.device)

            z_s = self.sample_zs_from_zt(z_t=z_t, s_int=s_array)

            z_t = z_s

        # Sample final data
        sampled = z_t.collapse(self.dataset_infos.collapse_charges)
        X, charges, E, y, pos = sampled.X, sampled.charges, sampled.E, sampled.y, sampled.pos
        idx = sampled.idx

        molecule_list, molecules_visualize = [], []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n]
            charge_vec = charges[i, :n]
            edge_types = E[i, :n, :n]
            conformer = pos[i, :n]
            template_idx = idx[i]
            molecule_list.append(Molecule(atom_types=atom_types, charges=charge_vec,
                                          bond_types=edge_types, positions=conformer,
                                          atom_decoder=self.dataset_infos.atom_decoder,
                                          template_idx=template_idx))

        return molecule_list

    @torch.no_grad()
    def inpainting_sample_batch(self, n_nodes: list, fixed_data, fixed_atoms, resamplings=1):
        """
        :param n_nodes: list of int containing the number of nodes to sample for each graph
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        n_nodes = torch.Tensor(n_nodes).long().to(self.device)
        batch_size = len(n_nodes)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        fixed_mask = torch.zeros_like(node_mask, device=self.device)
        fixed_mask[:, fixed_atoms] = True
        

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask, template=fixed_data, noise_std=self.noise_std)

        assert (z_T.E == torch.transpose(z_T.E, 1, 2)).all()
        n_max = z_T.X.size(1)
  
        z_t = z_T

        blank_x = torch.full_like(z_t.X, 1e-7, device=self.device)
        blank_y = torch.zeros_like(z_t.y, device=self.device)
        blank_E = torch.full_like(z_t.E, 1e-7, device=self.device)
        blank_charges = torch.full_like(z_t.charges, 1e-7, device=self.device)
        blank_pos = torch.zeros_like(z_t.pos, device=self.device)
        blank_x[:, fixed_atoms, :] = fixed_data.X[:, fixed_atoms, :]
        blank_E[:, fixed_atoms.unsqueeze(-1), fixed_atoms, :] = fixed_data.E[:, fixed_atoms.unsqueeze(-1), fixed_atoms, :]
        blank_charges[:, fixed_atoms, :] = fixed_data.charges[:, fixed_atoms, :]
        blank_pos[:, fixed_atoms, :] = fixed_data.pos[:, fixed_atoms, :]
        blank_data = PlaceHolder(X=blank_x, charges=blank_charges, y=blank_y, E=blank_E, pos=blank_pos, node_mask=node_mask,
                                 cX=fixed_data.cX, ccharges=fixed_data.ccharges, cE=fixed_data.cE, cy=fixed_data.cy, cpos=fixed_data.cpos
                                 ).mask(node_mask)   #The control information here is only to avoid errors when applying noise, and has no practical use.
 
  

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T, 1)):
            print(f'The {s_int} step diffusion' )
            for u in range(resamplings):

                s_array = s_int * torch.ones((batch_size, 1), dtype=torch.long, device=z_t.X.device)

                z_s_unknown = self.sample_zs_from_zt(z_t=z_t, s_int=s_array)
                
                z_s_known = self.noise_model.apply_noise(dense_data=blank_data, noise_std=self.noise_std, fixed_time=s_int)

                
                com_noised = utils.mean_with_mask(z_s_known.pos * fixed_mask.unsqueeze(-1), fixed_mask)
                com_denoised = utils.mean_with_mask(z_s_unknown.pos * fixed_mask.unsqueeze(-1), fixed_mask)
                dx = com_denoised - com_noised
                z_s_known.pos = z_s_known.pos + dx

                z_copy = z_s_unknown.copy()
                z_copy2 = z_s_unknown.copy()
                E_unknown = z_s_unknown.E - z_copy.mask_no_mean(fixed_mask).E
                z_copy2 = z_copy2.mask_no_mean(~fixed_mask)
                z_copy2.E = E_unknown

                z_s = z_s_known.mask_no_mean(fixed_mask).add(z_copy2).mask(node_mask)



                if u == resamplings - 1:
                    z_t = z_s
                if u < resamplings - 1:
                    t_array = (s_int + 1) * torch.ones((batch_size, 1), dtype=torch.long, device=z_s.X.device)
                    z_t = self.noise_model.sample_zt_from_zs(z_s=z_s, t_int=t_array)
                

        # Sample final data
        sampled = z_t.collapse(self.dataset_infos.collapse_charges)
        X, charges, E, y, pos = sampled.X, sampled.charges, sampled.E, sampled.y, sampled.pos

        molecule_list, molecules_visualize = [], []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n]
            charge_vec = charges[i, :n]
            edge_types = E[i, :n, :n]
            conformer = pos[i, :n]
            molecule_list.append(Molecule(atom_types=atom_types, charges=charge_vec,
                                          bond_types=edge_types, positions=conformer,
                                          atom_decoder=self.dataset_infos.atom_decoder))

        return molecule_list
    

    def kl_prior(self, clean_data, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((clean_data.X.size(0), 1), dtype=torch.long, device=clean_data.X.device)
        Ts = self.T * ones
        Qtb = self.noise_model.get_Qt_bar(t_int=Ts)

        # Compute transition probabilities
        probX = clean_data.X @ Qtb.X + 1e-7  # (bs, n, dx_out)
        probE = clean_data.E @ Qtb.E.unsqueeze(1) + 1e-7  # (bs, n, n, de_out)
        probc = clean_data.charges @ Qtb.charges + 1e-7
        probX = probX / probX.sum(dim=-1, keepdims=True)
        probE = probE / probE.sum(dim=-1, keepdims=True)
        probc = probc / probc.sum(dim=-1, keepdims=True)
        assert probX.shape == clean_data.X.shape

        bs, n, _ = probX.shape
        limit_dist = self.noise_model.get_limit_dist().device_as(probX)

        # Set masked rows , so it doesn't contribute to loss
        probX[~node_mask] = limit_dist.X.float()
        probc[~node_mask] = limit_dist.charges.float()
        diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
        probE[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = limit_dist.E.float()

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist.X[None, None, :], reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist.E[None, None, None, :], reduction='none')
        kl_distance_c = F.kl_div(input=probc.log(), target=limit_dist.charges[None, None, :], reduction='none')

        # Compute the kl on the positions
        last = self.T * torch.ones((bs, 1), device=clean_data.pos.device, dtype=torch.long)
        mu_T = self.noise_model.get_alpha_bar(t_int=last, key='p')[:, :, None] * clean_data.pos
        sigma_T = self.noise_model.get_sigma_bar(t_int=last, key='p')[:, :, None]
        subspace_d = 3 * node_mask.long().sum(dim=1)[:, None, None] - 3
        kl_distance_pos = subspace_d * diffusion_utils.gaussian_KL(mu_T, sigma_T)
        return (sum_except_batch(kl_distance_X) + sum_except_batch(kl_distance_E) + sum_except_batch(kl_distance_c) +
                sum_except_batch(kl_distance_pos))

    def compute_Lt(self, clean_data, pred, z_t, s_int, node_mask, test):
        # TODO: ideally all probabilities should be computed in log space
        t_int = z_t.t_int
        pred = utils.PlaceHolder(X=F.softmax(pred.X, dim=-1), charges=F.softmax(pred.charges, dim=-1),
                                 E=F.softmax(pred.E, dim=-1), pos=pred.pos, node_mask=clean_data.node_mask, y=None)

        Qtb = self.noise_model.get_Qt_bar(z_t.t_int)
        Qsb = self.noise_model.get_Qt_bar(s_int)
        Qt = self.noise_model.get_Qt(t_int)

        # Compute distributions to compare with KL
        bs, n, d = clean_data.X.shape
        prob_true = diffusion_utils.posterior_distributions(clean_data=clean_data, noisy_data=z_t,
                                                            Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(clean_data=pred, noisy_data=z_t,
                                                            Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true = diffusion_utils.mask_distributions(prob_true, node_mask)
        prob_pred = diffusion_utils.mask_distributions(prob_pred, node_mask)

        # Compute the prefactor for KL on the positions
        nm = self.noise_model
        prefactor = ((nm.get_alpha_bar(t_int=s_int, key='p') / (nm.get_sigma_bar(t_int=s_int, key='p') + 1e-6)) ** 2 -
                     (nm.get_alpha_bar(t_int=t_int, key='p') / (nm.get_sigma_bar(t_int=t_int, key='p') + 1e-6)) ** 2)

        prefactor[torch.isnan(prefactor)] = 1
        prefactor = torch.sqrt(0.5 * prefactor).unsqueeze(-1)
        prob_true.pos = prefactor * clean_data.pos
        prob_pred.pos = prefactor * pred.pos
        metrics = (self.test_metrics if test else self.val_metrics)(prob_pred, prob_true)
        return self.T * (metrics['PosMSE'] + metrics['XKl'] + metrics['ChargesKl'] + metrics['EKl'])

    def compute_val_loss(self, pred, z_t, clean_data, test=False):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
        """
        node_mask = z_t.node_mask
        t_int = z_t.t_int
        s_int = t_int - 1

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(clean_data, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(clean_data, pred, z_t, s_int, node_mask, test)

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t
        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        log_dict = {"kl prior": kl_prior.mean(),
                  "Estimator loss terms": loss_all_t.mean(),
                  "log_pn": log_pN.mean(),
                  'test_nll' if test else 'val_nll': nll}
        return nll, log_dict

    def sample_zs_from_zt(self, z_t, s_int): 
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, pred=pred, s_int=s_int)
        return z_s

    @property
    def BS(self):
        return self.cfg.train.batch_size

    def apply_model(self, model_input, condition_control):
        if condition_control:
            control_out = self.control_model(model_input)
            control_out = {ckey: control_out[ckey].mul_scales(scale) for ckey, scale in zip(control_out, self.control_scales)}
            model_out = self.model(model_input, control_out)
        else:
            control_out = None
            model_out = self.model(model_input, control_out)

        return model_out

    def forward(self, z_t, extra_data):
        assert z_t.node_mask is not None
        model_input = z_t.copy()
        model_input.X = torch.cat((z_t.X, extra_data.X), dim=2).float()
        model_input.E = torch.cat((z_t.E, extra_data.E), dim=3).float()
        model_input.y = torch.hstack((z_t.y, extra_data.y, z_t.t)).float()
        model_t = self.apply_model(model_input, self.condition_control)

        if self.guess_mode:
            model_uncond = self.apply_model(model_input, False)
            if self.add_gru_output_model == False:
                model_t = model_t.minus_scales(model_uncond, model_t.node_mask)
                model_t_scale = model_t.mul_scales(self.unconditional_guidance_scale)
                model_out = model_uncond.add_scales(model_t_scale, model_t_scale.node_mask)
            else:
                model_out = self.output_model(model_uncond, model_t)
        else:
            model_out = model_t
            
        return model_out

    def on_fit_start(self) -> None:
        # self.train_iterations = 100     # TODO: fix -- previously was len(self.trainer.datamodule.train_dataloader())
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def configure_optimizers(self):
        # lr = self.cfg.train.lr
        params = self.control_model.parameters() if self.condition_control else self.model.parameters()
        if self.add_gru_output_model:
            params = list(params) + list(self.output_model.parameters())

        control_optimizer = torch.optim.AdamW(params, lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)
        # StepLR = torch.optim.lr_scheduler.ReduceLROnPlateau(control_optimizer, mode='min', factor=0.5, patience=3, verbose=True,
        #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
        #                                            eps=1e-08)
        # optim_dict = {'optimizer': control_optimizer, 'lr_scheduler': StepLR, "monitor": 'train_epoch/epoch_loss'}
        optim_dict = {'optimizer': control_optimizer}

        return optim_dict
