import torch
from rdkit import Chem
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
import itertools
from rdkit.Chem import QED 
import os
import midi.utils as utils
from midi.metrics.molecular_metrics import filter_substructure
from midi.datasets.dataset_utils import mol_to_torch_geometric, MolInfos
from midi.datasets.geom_dataset import full_atom_encoder
from midi.diffusion_model import FullDenoisingDiffusion
from midi.analysis.rdkit_functions import Molecule
import hydra
import omegaconf
from collections import Counter
import copy


def data_from_sdf(sdf_path, remove_atoms, control_data_dict):
    rdmol = Chem.SDMolSupplier(sdf_path, removeHs=False)
    rdmol = next(rdmol)
    mol = mol_to_torch_geometric(rdmol, full_atom_encoder, smiles=None)
    orign_mol = copy.deepcopy(mol)

    # The idx of atoms to be removed from the template, that is, the part to be regenerated
    remove_atoms = torch.tensor(remove_atoms)

    total_num = len(mol.x)
    all_atoms = torch.arange(0, total_num)
    mask = torch.ones_like(all_atoms, dtype=bool)
    if len(remove_atoms) != 0:
        mask[remove_atoms] = False
    fixed_atoms = all_atoms[mask]

    # The settings of control information are consistent with the loaded model parameters
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

    

    dense_data = utils.to_dense(orign_mol, dataset_info=None)
    dense_data = dense_data.collapse(torch.Tensor([-2, -1, 0, 1, 2, 3]).int())
    atom_decoder = [key for key in full_atom_encoder.keys()]
    rdkit_mol = Molecule(atom_types=dense_data.X.squeeze(0), charges=dense_data.charges.squeeze(0),
                       bond_types=dense_data.E.squeeze(0), positions=dense_data.pos.squeeze(0),
                       atom_decoder=atom_decoder).rdkit_mol
    rdkit_mol.SetProp('_Name', f"template")
    return mol, rdkit_mol, fixed_atoms



def write_sdf_file(out_path, sample_template_mol, samples):
    all_valid_mols = list()
    all_invalid_mols = list()
    Decentralized_mols = list() #
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
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                largest_mol.SetProp('smiles', smiles)
                largest_mol.SetProp('qed', str(QED.qed(largest_mol)))

                match = any([largest_mol.HasSubstructMatch(subst) for subst in filter_smarts])
                
                if Chem.MolToSmiles(largest_mol) != Chem.MolToSmiles(rdmol):
                    Decentralized_mols.append(rdmol)
                    error_message[5] += 1


                if not match:
                    all_valid_mols.append(largest_mol)
                    error_message[-1] += 1
                else:
                    all_invalid_mols.append(largest_mol)
                    error_message[4] += 1

    
            except Chem.rdchem.AtomValenceException:
                error_message[1] += 1
                # print("Valence error in GetmolFrags")
            except Chem.rdchem.KekulizeException:
                error_message[2] += 1
                # print("Can't kekulize molecule")
            except Chem.rdchem.AtomKekulizeException or ValueError:
                error_message[3] += 1

    print(f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
              f" -- No error {error_message[-1]}   invalid {error_message[4]}  Decentralized_mols {error_message[5]}")
    if len(all_valid_mols) > 0:
        with Chem.SDWriter(out_path)as f:
            f.write(sample_template_mol)
            for mol in all_valid_mols:
                f.write(mol)
    if len(all_invalid_mols) > 0:
        directory_path, filename = os.path.split(out_path)
        new_file_path = os.path.join(directory_path, 'SynHard_'+filename)
        with Chem.SDWriter(new_file_path)as f:
            f.write(sample_template_mol)
            for mol in all_invalid_mols:
                f.write(mol)

    


def inpaint_mol(remove_atoms, model, sdf_path, control_data_dict, samples_to_generate, potential_ebs, device, dataset_infos=None, resamplings=1):
    

    if samples_to_generate <= 0:
        return []
    
    # Load SDF
    inpainting_template, template_mol, fixed_atoms = data_from_sdf(sdf_path, remove_atoms, control_data_dict)
    fixed_atoms.to(device)

    samples = []
    template = Batch.from_data_list(list(itertools.repeat(inpainting_template, samples_to_generate)))
    template_loader = DataLoader(template, potential_ebs, shuffle=True)
    for i, template_batch in enumerate(template_loader):
        template_batch = template_batch.to(device)
        current_n_list = torch.unique(template_batch.batch, return_counts=True)[1]
        dense_data = utils.to_dense(template_batch, dataset_infos)
        n_nodes = current_n_list

        # Run sampling
        samples.extend(model.inpainting_sample_batch(n_nodes=n_nodes, fixed_data=dense_data, fixed_atoms=fixed_atoms, resamplings=resamplings))
    return template_mol, samples
    
    

@hydra.main(
    version_base='1.3',
    config_path="../configs/experiment",  # 指向包含 YAML 的目录
    config_name="diffshape_sample_fragment"      # 不带 .yaml 后缀的文件名
)
def main(cfg: omegaconf.DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    dataset_infos = MolInfos(statistics_path=cfg.sample.statistics_path, atom_encoder=full_atom_encoder)
    model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path=cfg.sample.loading_model, map_location={'cuda:1': 'cuda:0'}, dataset_infos=dataset_infos)
    model.T = cfg.model.diffusion_steps
    model = model.to(device)

    template_mol, molecules = inpaint_mol(cfg.sample.change_atom_idx, model, cfg.sample.sdf_path, control_data_dict=cfg.model.control_data_dict, samples_to_generate=cfg.sample.samples_to_generate,
                            potential_ebs=cfg.sample.potential_ebs, device=device, resamplings=cfg.sample.resamplings)
    
    # Make SDF files
    out_path = cfg.sample.output_dir
    if os.path.splitext(out_path)[1]:
        raise ValueError(f"Expected a directory path, but got a file path: {out_path}")
    
    result_dir = 'based_motif_shape'
    result_path = os.path.join(out_path, f"{result_dir}/")
    os.makedirs(result_path, exist_ok=True)
    out_path = os.path.join(result_path, f'{os.path.splitext(os.path.basename(cfg.sample.sdf_path))[0]}_motif_gen_mols.sdf')
    write_sdf_file(out_path, template_mol, molecules)
    print(f"\n=== The sample results have been saved to: {out_path} ===")
    

if __name__ == "__main__":
    main()
