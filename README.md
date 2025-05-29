# [Diff-Shape](https://chemrxiv.org/engage/chemrxiv/article-details/662f19a121291e5d1dfb745b): Shape Constrained 3D Structure Generation Through Novel Graph Controlled Diffusion Model 


<p align="center">
  <img src="assets/linker-5CB2.gif" width="400">
  <img src="assets/linker-5cb2.png" width="400">
</p>

## 📦 Conda Environment Dependencies

## Optimized Dependency List (No pip/conda syntax)

- cudatoolkit == 11.8.0
- pytorch == 2.0.1
- rdkit == 2023.03.2
- scipy == 1.11.1
- hydra-core ==1.3.2
- imageio==2.31.1
- matplotlib==3.7.0
- numpy == 1.25.0
- omegaconf == 2.3.0
- pandas == 2.0.2
- Pillow==9.5.0
- pytorch_lightning == 2.0.6
- scikit_learn == 1.2.2
- setuptools==68.0.0
- torch_geometric == 2.3.1
- torchmetrics == 0.11.4
- tqdm == 4.65.0
- wandb == 0.15.4


You can create the environment with the following dependencies:

```bash
conda create -n diffshape python=3.9 rdkit=2023.03.2
conda activate diffshape


# Core dependencies
conda install -c "nvidia/label/cuda-11.8.0" cuda

pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```



## 📂 Datasets

We use the same datasets as the [MiDi](https://github.com/cvignac/MiDi) model.

Download and place them under `./data/geom/raw/`:

* **Train:** [Download](https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle)
* **Validation:** [Download](https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle)
* **Test:** [Download](https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle)

---

## 🏋️ Training

We use MiDi's checkpoint [geom-with-h-adaptive model](https://drive.google.com/file/d/1ExNpU7czGwhPWjpYCcz0mHGxLo8LvQ0b/view?usp=drive_link) as the pre-trained model.

Place it in:

```bash
./checkpoints/pre-trained/
```

Then run training:

```bash
cd ./midi
python3 main.py +experiment=example_geom_with_h_adaptive
```

---

## 🧪 Testing

You can use:

* A model you trained in the previous step
* Or our trained model: [Download](https://drive.google.com/drive/folders/1qTRhD-CvgXCE9cvWX5dHEzDxHsPH6Qck)

Place it in:

```bash
./checkpoints/
```

Then run:

```bash
cd ./midi
python3 main.py +experiment=example_geom_with_h_adaptive general.test_only='ABS_PATH'
```

Replace `ABS_PATH` with the absolute path of the model checkpoint.

---

## 🧬 Sampling Example

<p align="left">
  <img src="assets/normal-sampling.gif" width="400">
</p>

To perform shape-based molecule generation conditioned on a template molecule (e.g., `1z95_ligand.sdf`), you can choose one of the following two approaches:

### 1. Sampling via the Training/Testing Pipeline

This method uses the same `main.py` script used for training/testing.

First, encode the shape features from your template molecule:

```bash
python get_template_encoder.py
```

Then run sampling with the pre-trained model:

```bash
cd ./midi
python3 main.py +experiment=example_geom_with_h_adaptive \
    general.test_only='ABS_PATH' \
    dataset.template_name=1z95
```

Ensure that `ABS_PATH` points to the correct checkpoint and that `1z95` refers to a valid template in your statistics directory.

### 2. Sampling via a Dedicated DiffShape Script

This method is recommended for more lightweight or customized sampling scenarios. It directly invokes the dedicated sampling script `diffshape_sample_shape.py`.

```bash
python -m midi.diffshape_sample_shape
```

Make sure to set the correct `statistics_path` in your `diffshape-sampling.yaml`. This path should contain the encoded template shape statistics prepared using `get_template_encoder.py`.

Refer to `get_template_encoder.py` for details on how shape constraints are extracted from a given SDF template molecule.

## 🌍 Inpainting (Sampling under Dual Control: Substructure Fixing + Shape Constraints)

<p align="center">
  <img src="assets/Rgroup-CID_56848121.gif" width="400">
  <img src="assets/Rgroup-CID_67260295.gif" width="400">
</p>

Inpainting allows you to design novel molecules around fixed substructures and shape-constraint, supporting tasks such as scaffold hopping, fragment linking, and fragment elaboration. You can either use your own trained model from the training step, or download our pre-trained model from the following link:

You can either use your **own trained model** from the training step, or download our **pre-trained model**:

📁 [Pretrained Model on Google Drive](https://drive.google.com/drive/folders/1qTRhD-CvgXCE9cvWX5dHEzDxHsPH6Qck)

> After downloading, place the model files into the `./checkpoints/` directory.

---

### ⚛️ How It Works

The inpainting workflow leverages two scripts:

* `diffshape_sample_fragment.py`: loads **one specific model** and performs constrained sampling based on a given template.
* `diffshape_batch_sample_fragment.py`: iteratively loads **multiple models** (e.g., models trained under different fuzzy strategies and noise levels (i.e., control conditions)), samples from each, and merges all generated results for more diverse outcomes.

Both scripts support inpainting by specifying a subset of atoms to re-generate via the `change_atom_idx` field in the YAML configuration (`diffshape_sample_fragment.yaml` or `diffshape_batch_sample_fragment.yaml`).

* The atoms listed in `change_atom_idx` will be re-generated.
* The remaining structure will be fixed, serving as a constraint.

---

### ▶️ Run Inpainting

#### Sampling using a **single model checkpoint**:

```bash
python -m midi.diffshape_sample_fragment
```

#### Sampling using **multiple checkpoints**, merging results:

```bash
python -m midi.diffshape_batch_sample_fragment
```

Ensure your YAML config specifies:

* `checkpoint_path`: the directory containing model checkpoints.
* `template_name`: the filename of your SDF scaffold.
* `change_atom_idx`: atom indices to re-generate.

---

### 📊 Example: Inpainting around `5cb2.sdf`

Use the provided `5cb2.sdf` file as a molecular template, and define the `change_atom_idx` in the YAML config to preserve key substructures. The model will generate shape-constrained novel molecules that complete the remaining parts.

---

