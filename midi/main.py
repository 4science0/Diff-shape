# Do not move these imports, the order seems to matter
import torch
import pytorch_lightning as pl

import os
import warnings
import pathlib

import hydra
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from midi.datasets import qm9_dataset, geom_dataset, chembl_dataset
from midi.diffusion_model import FullDenoisingDiffusion
from utils import save_template, save_template_new

warnings.filterwarnings("ignore", category=PossibleUserWarning)

import sys
from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar
class MyTQDMProgressBar(TQDMProgressBar):
    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()

    def init_validation_tqdm(self):
        bar = tqdm(
            desc=self.validation_description,
            position=0,  # 这里固定写0
            disable=self.is_disabled,
            leave=True,  # leave写True
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

def get_resume(cfg, dataset_infos, train_smiles, checkpoint_path, test: bool):
    name = cfg.general.name + ('_test' if test else '_resume')
    gpus = cfg.general.gpus
    model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path, map_location={'cuda:1': 'cuda:0'},
                                                        dataset_infos=dataset_infos,
                                                        train_smiles=train_smiles)
    cfg.general.gpus = gpus
    cfg.general.name = name
    return cfg, model

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

def load_pretrained_model(model, pretrained_model_path):
    pretrained_weights = torch.load(pretrained_model_path, map_location={'cuda:1': 'cuda:0'})
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    scratch_dict = model.state_dict()
    control_name = []
    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = name
            # copy_k = name
            # if copy_k not in pretrained_weights:
            #     control_name.append(name)
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')
    model.load_state_dict(target_dict, strict=True)

    return model

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: omegaconf.DictConfig):
    print(f"***********control model layer:{cfg.model.n_layers}, batch_size:{cfg.train.batch_size}, "
          f"reference_batch_size:{cfg.train.reference_batch_size}, sample_type:{cfg.general.sample_type}***********")
    dataset_config = cfg.dataset
    if cfg.general.test_only:
        pl.seed_everything(cfg.train.seed)
    # print(f"cfg.train.batch_size:{cfg.train.batch_size}")
    datamodule = None
    if dataset_config.name in ['qm9', "geom", "chembl"]:
        if dataset_config.name == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

        elif dataset_config.name == 'geom':
            datamodule = geom_dataset.GeomDataModule(cfg)
            dataset_infos = geom_dataset.GeomInfos(datamodule=datamodule, cfg=cfg)

        elif dataset_config.name == "chembl":
            datamodule = chembl_dataset.ChemblDataModule(cfg)
            dataset_infos = chembl_dataset.ChemblInfos(datamodule=datamodule, cfg=cfg)

        train_smiles = list(datamodule.train_dataloader().dataset.smiles)
        test_smiles = list(datamodule.test_dataloader().dataset.smiles)

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    train_template = datamodule.train_dataset.template_data
    val_template = datamodule.val_dataset.template_data
    test_template = datamodule.test_dataset.template_data
    # train_template = save_template(train_template, dataset_infos, f"{datamodule.train_dataset.root}/train_template.sdf")
    # val_template = save_template(val_template, dataset_infos, f"{datamodule.val_dataset.root}/val_template.sdf")
    # test_template = save_template(test_template, dataset_infos, f"{datamodule.test_dataset.root}/test_template.sdf")
    # save_template_new(train_template, dataset_infos, f"{datamodule.train_dataset.root}/train_template_new.sdf")
    # save_template_new(val_template, dataset_infos, f"{datamodule.val_dataset.root}/val_template_new.sdf")
    # save_template_new(test_template, dataset_infos, f"{datamodule.test_dataset.root}/test_template_new.sdf")
    # torch.save(train_template, f"{datamodule.test_dataset.root}/processed/train_template_noh.pt")
    # torch.save(val_template, f"{datamodule.test_dataset.root}/processed/val_template_noh.pt")
    # torch.save(test_template, f"{datamodule.test_dataset.root}/processed/test_template_noh.pt")

    if cfg.general.test_only:
        pass
        # cfg, _ = get_resume(cfg, dataset_infos, train_smiles, cfg.general.test_only, test=True)
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        print("Resuming from {}".format(cfg.general.resume))
        cfg, _ = get_resume(cfg, dataset_infos, train_smiles, cfg.general.resume, test=False)

    # utils.create_folders(cfg)
    model = FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos, train_smiles=train_smiles,
                                   val_template=val_template, test_template=test_template, test_smiles=test_smiles)
    if (not cfg.general.test_only) and (cfg.model.condition_control == True) and (cfg.model.pretrained_model_path is not None):
        model = load_pretrained_model(model, cfg.model.pretrained_model_path)

    callbacks = []
    # need to ignore metrics because otherwise ddp tries to sync them
    params_to_ignore = ['module.model.train_smiles', 'module.model.dataset_infos']

    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)

    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=50,
                                              mode='min',
                                              every_n_epochs=1)
        # fix a name and keep overwriting
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(checkpoint_callback)
        callbacks.append(last_ckpt_save)
        # callbacks.append(MyTQDMProgressBar())
        # early_stopping_callback = early_stopping.EarlyStopping(monitor='val/epoch_NLL', min_delta=0.0, patience=2, verbose=True, mode='min', strict=True)
        # callbacks.append(early_stopping_callback)

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    num_nodes = cfg.train.num_nodes if hasattr(cfg.dataset, "num_nodes") else 1
    print(f"***********num_nodes:{num_nodes}***********")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      # num_nodes=num_nodes
                      )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    else:
        # Start by evaluating test_only_path
        for i in range(cfg.general.num_final_sampling):
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
