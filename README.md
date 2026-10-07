# Operator Informed CT Reconstruction with Mamba-based sequential Processing

## Software and References

This project makes use of the following open-source software:

- **PyTorch**  
  Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*.  
  [https://pytorch.org/](https://pytorch.org/)

- **Mamba-SSM**  
  Gu and Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.  
  Gu and Dao (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*.  
  [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

- **causal-conv1d**  
  Dao. *causal-conv1d*.  
  [https://github.com/Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)

- **ASTRA Toolbox**  
  Van Aarle et al. (2015).  
  *The ASTRA Toolbox: A platform for advanced algorithm development in electron tomography*.  
  **Ultramicroscopy, 157**, 35–47.  
  [https://doi.org/10.1016/j.ultramic.2015.05.002](https://doi.org/10.1016/j.ultramic.2015.05.002)

- **NumPy**  
  Harris et al. (2020). *Array programming with NumPy*.  
  **Nature, 585**, 357–362.  
  [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

- **Matplotlib**  
  Hunter (2007). *Matplotlib: A 2D Graphics Environment*.  
  **Computing in Science & Engineering, 9**(3), 90–95.  
  [https://doi.org/10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)


## Dataset Ground-Truth Source References

- **Ellipse Images Generation**  
  Auras et al. (2026). *A neural operator view on U-Nets for inverse imaging 
  problems*  
  [https://arxiv.org/abs/2608.05839](https://arxiv.org/abs/2608.05839)

- **LoDoPoB-CT**  
  Leuschner et al. (2021). *LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction*  
  **Sci Data, 8**.  
  [https://doi.org/10.1038/s41597-021-00893-z](https://doi.org/10.1038/s41597-021-00893-z)


## Installation

### Create Environment:

```bash
cd <path to mamba-ct-project directory>

conda env create -f environment.yml
conda activate mamba2_c121
```

### Installation of PyTorch (CUDA 12.1):

```bash
pip install torch==2.5.1+cu121 \
torchvision==0.20.1+cu121 \
torchaudio==2.5.1+cu121 \
--index-url https://download.pytorch.org/whl/cu121
```

### Installation of Mamba-SSM:


```bash
pip install causal-conv1d==1.6.0 --no-build-isolation
pip install mamba-ssm==2.3.0 --no-build-isolation
```

## Project Structure

```text
project_dir/
├── topk/
├── datasets/
├── checkpoints/
└── src/
    ├── ellipsesGen/
    │   └── EllipsesDataset.py
    ├── topk.py
    ├── create_datafile.py
    ├── main.py
    ├── astra_wrapper
    ├── iterative_rec.py
    ├── rec_model.py
    ├── sequence_dataset.py
    └── weight_matrix.py
```
        


## Training Commands

```bash
cd <path to project_dir>
```

### I. TOPK (create topk weights/indices file before the training)

```bash
python src/topk.py
```

### II. SEQUENCE DATASET FILE (create dataset file before training)

```bash
python src/create_datafile.py # default dataset (32x32 Ellipses-GT)
python src/create_datafile.py --I0=1e3 # adjust noise level in projection data
python src/create_datafile.py --img_size=128 # adjust image resolution
python src/create_datafile.py --dataset_gt='lodopab' # adjust ground-truth source (LoDoPaB)
```

### III. TRAINING (optional: multi-GPU training)

```bash
# default run
torchrun --nproc_per_node=<n_gpus> src/main.py 

# run with adjusted configurations (train epochs, dataset file, and model checkpoint)
torchrun --nproc_per_node=<n_gpus> src/main.py \
--n_epochs=100 \
--model_file='lodopab_cp_128px1e3_mp.pt' \
--dataset_file='lodopabSeqData_128px1000.hdf5'
```
