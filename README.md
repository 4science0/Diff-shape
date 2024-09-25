# [Diff-Shape]: A Novel Constrained Diffusion Model for Shape based De Novo Drug Design

## Conda environment Dependencies
- cudatoolkit==11.8.0
- numpy==1.25.0
- omegaconf
- pandas==2.0.2
- pytorch==2.0.1
- rdkit==2023.03.3
- scikit_learn==1.2.2
- scipy==1.11.1
- torch-geometric==2.3.1
- torchmetrics==0.11.4
- tqdm==4.65.0
### Create a conda environment with the following command
```
conda env create -f environment.yml
pip install -e .
```
## Datasets
We trained/tested DiffShape using the same data sets as [MiDi](https://github.com/cvignac/MiDi) model.
GEOM, download the data and put in ./data/geom/raw/:
train: https://drive.switch.ch/index.php/s/UauSNgSMUPQdZ9v
validation: https://drive.switch.ch/index.php/s/YNW5UriYEeVCDnL
test: https://drive.switch.ch/index.php/s/GQW9ok7mPInPcIo
## Training:
We use MiDi's [geom-with-h-adaptive model](https://drive.google.com/file/d/1ExNpU7czGwhPWjpYCcz0mHGxLo8LvQ0b/view?usp=drive_link) as our pre-trained model.
Download the pre-train model and put it in ./outputs/model
``
cd ./midi
python3 main.py +experiment=example_geom_with_h_adaptive
``

## Test
You can use the model trained in the Training step as a test model, 
or use the model we have already trained, download it from 
[Google Cloud Drive](https://drive.google.com/drive/folders/1qTRhD-CvgXCE9cvWX5dHEzDxHsPH6Qck)
and put it in the ./outputs/model/ directory
```
cd ./midi
python3 main.py +experiment=example_geom_with_h_adaptive general.test_only='ABS_PATH'
```
##Sampling example
- sample 1z95_ligand.sdf




## Download the trained DiffShape model
Download the trained DiffShape model from [Google Cloud Drive](https://drive.google.com/drive/folders/1qTRhD-CvgXCE9cvWX5dHEzDxHsPH6Qck), and then place the downloaded .ckpt file in the ```./outputs/model``` folder

## Sampling example
- sample 1z95_ligand.sdf
```
python get_template_encoder.py
cd ./midi
python3 main.py +experiment=example_geom_with_h_adaptive general.test_only='ABS_PATH' dataset.template_name=1z95
```
## Inpainting
You can use the model trained in the Training step as a test model,
or use the model we have already trained, download it from 
[Google Cloud Drive](https://drive.google.com/drive/folders/1qTRhD-CvgXCE9cvWX5dHEzDxHsPH6Qck)
and put it in the ./outputs/model/ directory

To design molecules around fixed substructures (scaffold hoping, fragment linking etc.) you can run the `Diffshape_inpainting.py` script.
The inpainting script allows us to fix parts of a molecule and generate new molecules based on the shapes of the unfixed fragments.
The way to remove substructures is to provide them in `remove_atoms` in yaml.
```
cd ./midi
python3 Diffshape_inpainting.py +experiment=diffshape_inpainting
```
##Sampling example
- sample 5cb2.sdf
