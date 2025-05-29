import torch
from rdkit import Chem
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader 
import torch.nn.functional as F
import itertools
from rdkit.Chem import QED 
import os
import midi.utils as utils
from midi.metrics.molecular_metrics import filter_substructure
from midi.datasets.dataset_utils import mol_to_torch_geometric
from midi.datasets import geom_dataset
from midi.datasets.dataset_utils import MolInfos, mol_to_torch_geometric
# from datasets.geom_phar_dataset import full_atom_encoder
from midi.datasets.geom_dataset import full_atom_encoder
from midi.diffusion_model import FullDenoisingDiffusion
from midi.analysis.rdkit_functions import Molecule 
import hydra
import omegaconf
from collections import Counter
import json
from tqdm import tqdm
import numpy as np

def data_from_sdf(sdf_path, change_atom_rank, control_data_dict):
    rdmol = Chem.SDMolSupplier(sdf_path, removeHs=False)
    rdmol = next(rdmol)
    mol = mol_to_torch_geometric(rdmol, full_atom_encoder, smiles=None)

    if isinstance(control_data_dict, str):
        control_data_dict = json.loads(control_data_dict)

    remove_atoms = torch.tensor(change_atom_rank)

    total_num = len(mol.x)
    all_atoms = torch.arange(0, total_num)
    mask = torch.ones_like(all_atoms, dtype=bool)
    if len(remove_atoms) != 0:
        mask[remove_atoms] = False
    fixed_atoms = all_atoms[mask]

    mol.cx = mol.cx if control_data_dict['cX'] == 'cX' else mol.x
    mol.ccharges = mol.ccharges if control_data_dict['cX'] == 'cX' else mol.charges

    if control_data_dict['cE'] == 'cE':
        mol.cedge_attr = mol.cedge_attr
    elif control_data_dict['cE'] == 'None':
        mol.cedge_attr = torch.zeros_like(mol.cedge_attr)
    elif control_data_dict['cE'] == 'E':
        mol.cedge_attr = mol.edge_attr
    elif control_data_dict['cE'] == 'single_mask_None':
        mask_tensor = torch.rand(mol.cedge_attr.shape[0])>0.5
        mol.cedge_attr[mask_tensor] = 0


    rdmol.SetProp('_Name', f"template")
    return mol, rdmol, fixed_atoms

def remove_duplicate_molecules(mol_list):
    unique_smiles = set()
    unique_mols = []
    
    for mol in mol_list:
        # 把分子转成标准化的SMILES，不带立体信息
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        
        if smiles not in unique_smiles:
            unique_smiles.add(smiles)
            unique_mols.append(mol)
    
    return unique_mols

def transfer_to_template_pos(mols_list, template_mol, reference_atoms):
    """
    将mols_list中的分子, 根据reference_atoms指定的原子索引, 通过平移方式align到template_path分子的位置
    Align molecules in mols_list to the position of the molecule in template_path by translating them based on the atom indices specified in reference_atoms.
    """

    def get_coordinates(mol, idx_list):
        conf = mol.GetConformer()
        return np.array([conf.GetAtomPosition(i) for i in idx_list])

    def translate_molecule(mol, translation_vector):
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            new_pos = pos + translation_vector
            conf.SetAtomPosition(i, new_pos)


    template_pos = get_coordinates(template_mol, reference_atoms)
    template_center = template_pos.mean(axis=0)

    aligned_mols = []

    for mol in mols_list:
        try:
            mol_pos = get_coordinates(mol, reference_atoms)
            mol_center = mol_pos.mean(axis=0)
            translation_vector = template_center - mol_center
            translate_molecule(mol, translation_vector)
            aligned_mols.append(mol)
        except Exception as e:
            continue

    return aligned_mols

def transfer_to_template_space(mols_list, template_mol):
    """
    将mols_list中的分子, 根据原质心移动向量, 通过平移方式align到template_path分子的位置
    Align molecules in mols_list to the position of the template molecule (from template_path) by applying centroid-based translations.
    """

    def get_centroid(mol):
        conf = mol.GetConformer()  # 获取分子构象
        pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])  # 获取所有原子坐标
        centroid = pos.mean(axis=0)
        return centroid

    def translate_molecule(mol, translation_vector):
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            new_pos = pos + translation_vector
            conf.SetAtomPosition(i, new_pos)


    template_center = get_centroid(template_mol)
    aligned_mols = []

    for mol in mols_list:
        try:
            translation_vector = template_center
            translate_molecule(mol, translation_vector)
            aligned_mols.append(mol)
        except Exception as e:
            continue

    return aligned_mols

def find_fix_atoms(change_atom_rank, num_required=3):
    result = []
    i = 0
    while len(result) < num_required:
        if i not in change_atom_rank:
            result.append(i)
        i += 1
    return result


def write_sdf_file(out_path, sample_template_mol, samples):
    all_valid_mols = list()
    all_invalid_mols = list()
    error_message = Counter()
    filter_smarts = [Chem.MolFromSmarts(subst) for subst in filter_substructure if Chem.MolFromSmarts(subst)]
    for mol in samples:
        rdmol = mol.rdkit_mol
        if rdmol is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest_mol)
                smiles = Chem.MolToSmiles(largest_mol)
                if Chem.MolFromSmiles(smiles) is not None:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                    largest_mol.SetProp('smiles', smiles)
                largest_mol.SetProp('qed', str(QED.qed(largest_mol)))

                match = any([largest_mol.HasSubstructMatch(subst) for subst in filter_smarts]) 
                

                if not match:
                    all_valid_mols.append(largest_mol)
                    error_message[-1] += 1
                else:
                    all_invalid_mols.append(largest_mol)
                    error_message[4] += 1
                # all_valid_mols.append(largest_mol)
    
            except Chem.rdchem.AtomValenceException:
                error_message[1] += 1
                # print("Valence error in GetmolFrags")
            except Chem.rdchem.KekulizeException:
                error_message[2] += 1
                # print("Can't kekulize molecule")
            except Chem.rdchem.AtomKekulizeException or ValueError: 
                error_message[3] += 1

    print(f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
              f" -- No error {error_message[-1]}   invalid {error_message[4]}")
    
    unique_valid_mols = remove_duplicate_molecules(all_valid_mols)
    unique_invalid_mols = remove_duplicate_molecules(all_invalid_mols)
    
    # fix_atoms = find_fix_atoms(change_atom_rank)
    # unique_valid_mols = transfer_to_template_pos(unique_valid_mols, sample_template_mol, fix_atoms)
    # unique_invalid_mols = transfer_to_template_pos(unique_invalid_mols, sample_template_mol, fix_atoms)
    unique_valid_mols = transfer_to_template_space(unique_valid_mols, sample_template_mol)
    unique_invalid_mols = transfer_to_template_space(unique_invalid_mols, sample_template_mol)
    
    if len(unique_valid_mols) > 0:
        with Chem.SDWriter(out_path)as f:
            f.write(sample_template_mol)
            for mol in unique_valid_mols:
                f.write(mol)
    if len(unique_invalid_mols) > 0:
        directory_path, filename = os.path.split(out_path)
        new_file_path = os.path.join(directory_path, 'SynHard_'+filename)
        with Chem.SDWriter(new_file_path)as f:
            f.write(sample_template_mol)
            for mol in unique_invalid_mols:
                f.write(mol)

    


def inpaint_mol(model, sdf_path, change_atom_rank, control_data_dict, samples_to_generate, potential_ebs, device, dataset_infos=None, resamplings=1):
    

    if samples_to_generate <= 0:
        return []
    
    # Load SDF
    inpainting_template, template_mol, fixed_atoms = data_from_sdf(sdf_path, change_atom_rank, control_data_dict)
    fixed_atoms.to(device)

    samples = []
    template = Batch.from_data_list(list(itertools.repeat(inpainting_template, samples_to_generate)))
    template_loader = DataLoader(template, potential_ebs, shuffle=True)
    for i, template_batch in enumerate(template_loader):
        template_batch = template_batch.to(device)
        current_n_list = torch.unique(template_batch.batch, return_counts=True)[1]
        dense_data = utils.to_dense(template_batch, dataset_infos)
        n_nodes = current_n_list

        # #sample nums
        # random_tensor = torch.randint(-5, 6, n_nodes.shape, device=n_nodes.device)
        # n_nodes = n_nodes + random_tensor

        # Run sampling
        samples.extend(model.inpainting_sample_batch(n_nodes=n_nodes, fixed_data=dense_data, fixed_atoms=fixed_atoms, resamplings=resamplings))
    return template_mol, samples
    
    


@hydra.main(
    version_base='1.3',
    config_path="../configs/experiment",  # 指向包含 YAML 的目录
    config_name="diffshape_batch_sample_fragment"      # 不带 .yaml 后缀的文件名
)
def main(cfg: omegaconf.DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ModelCkpt_list=[
    '../checkpoints/AtomBondFuzz_nstd0.3.ckpt',
    '../checkpoints/PointCloudSingle_nstd0.3.ckpt',
    '../checkpoints/PointCloud_nstd0.3.ckpt',
    '../checkpoints/AtomFuzz_nstd0.3.ckpt',
    '../checkpoints/ColourPointCloud_nstd0.3.ckpt',
    '../checkpoints/ColourPointCloudSingle_nstd0.3.ckpt',
    '../checkpoints/PointCloud_nstd0.2.ckpt',
    '../checkpoints/PointCloudSingle_dropout0.1_nstd0.3.ckpt',
    '../checkpoints/PointCloudSingle_nstd0.4.ckpt',
    '../checkpoints/PointCloudSingle_nstd0.35.ckpt',
    '../checkpoints/NoFuzz_nstd0.3.ckpt',
    '../checkpoints/PointCloud_nstd0.25.ckpt'
    ]
    # noise_std list
    noise_std = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.4, 0.35, 0.3, 0.25]

    # dropout_rate list
    dropout_rate = [0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0]

    # control_data_dict list
    control_data_dict = [
        {"cX": "cX", "cE": "cE", "cpos": "cpos"},
        {"cX": "cX", "cE": "single_mask_None", "cpos": "cpos"},
        {"cX": "cX", "cE": "None", "cpos": "cpos"},
        {"cX": "cX", "cE": "E", "cpos": "cpos"},
        {"cX": "X", "cE": "None", "cpos": "cpos"},
        {"cX": "X", "cE": "single_mask_None", "cpos": "cpos"},
        {"cX": "cX", "cE": "None", "cpos": "cpos"},
        {"cX": "cX", "cE": "single_mask_None", "cpos": "cpos"},
        {"cX": "cX", "cE": "single_mask_None", "cpos": "cpos"},
        {"cX": "cX", "cE": "single_mask_None", "cpos": "cpos"},
        {"cX": "X", "cE": "single_mask_None", "cpos": "cpos"},
        {"cX": "cX", "cE": "None", "cpos": "cpos"},
    ]

    # Load model
    dataset_infos = MolInfos(statistics_path=cfg.sample.statistics_path, atom_encoder=full_atom_encoder)
    final_mols = []
    for model_idx, model_ckpt in enumerate(tqdm(ModelCkpt_list, desc="Processing models", ncols=80)):
        tqdm.write(f"Now sampling with Diff-Shape version: {os.path.splitext(os.path.basename(model_ckpt))[0]}")
        model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path=model_ckpt, map_location={'cuda:1': 'cuda:0'}, 
                                                            dataset_infos=dataset_infos)
        model.T = cfg.model.diffusion_steps
        model = model.to(device)

    
        template_mol, molecules = inpaint_mol(model, cfg.sample.sdf_path, cfg.sample.change_atom_idx, control_data_dict=control_data_dict[model_idx], samples_to_generate=cfg.sample.samples_to_generate,
                                potential_ebs=cfg.sample.potential_ebs, device=device, resamplings=cfg.sample.resamplings)
        final_mols.extend(molecules)
    
    # Make SDF files
    out_path = cfg.sample.output_dir
    if os.path.splitext(out_path)[1]:
        raise ValueError(f"Expected a directory path, but got a file path: {out_path}")
    
    result_dir = 'based_motif_shape'
    result_path = os.path.join(out_path, f"{result_dir}/")
    os.makedirs(result_path, exist_ok=True)
   
    out_path = os.path.join(result_path, f'{os.path.splitext(os.path.basename(cfg.sample.sdf_path))[0]}_motif_batchgen_mols.sdf')
    write_sdf_file(out_path, template_mol, molecules)
    print(f"\n=== The sample results have been saved to: {out_path} ===")
    

if __name__ == "__main__":
    main()
