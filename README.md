# CleftMast3r

Implementation of the pipeline from the paper **"An End-to-End System for 3D Reconstruction of Intraoral Cleft Anatomy in Children from Smartphone Video"** (Lingens et al., ETH Zurich & University Hospital Basel).

The system converts handheld intraoral smartphone video into a 3D mesh of the palate. It semi-automatically masks the palate region using Cutie, sub-samples frames by sharpness, runs MASt3R-SfM for sparse 3D reconstruction, and generates a surface mesh with NKSR.
This repository implements the pipeline presented in the paper that leads to the best reconstruction results. Additional planned features and known limitations are listed under [TODO](#todo).

## Installation
#### 1. Create a virtual environment (tested with Python 3.12)

Make sure the following libraries are installed:
```
sudo apt install libxcb-cursor0
sudo apt-get install python3-dev
```
Init venv
```
python3 -m venv .cleft_venv
source .cleft_venv/bin/activate
which python
```

#### 2. Clone repo and initialize submodules

Install PyTorch (torch + torchvision — see the [official guide](https://pytorch.org/get-started/locally/) for your platform):
```
git clone https://github.com/M45t3rL4/cleft-reconstruction-mast3r.git
cd CleftMast3r
git submodule update --init --recursive
pip3 install torch torchvision
```

#### 3. Install and test Cutie
```
pip3 install -e ./thirdparty/Cutie/
cd thirdparty/Cutie
```

*I ran into issues with `cchardet` and `netifaces` — commenting them out of `pyproject.toml` fixed it with no downstream effects.*
Download Cutie weights
```
python cutie/utils/download_models.py
```
*There's an error related to `qdarktheme`/`setuptheme` — commenting out that line is safe.*

Change gui_config in `thirdparty/Cutie/cutie/config/gui_config.yaml`, add the path prefix `thirdparty/Cutie/` to the weights — this ensures correct paths for the submodule.

Test Cutie with:
```
cd ../..
python thirdparty/Cutie/interactive_demo.py --video your_test_video
```


#### 4. Install and test MASt3R

We skip the optional packages and CroCo inference improvements, as neither is required here.
```
pip3 install -r thirdparty/Mast3r/requirements.txt
git clone https://github.com/jenicek/asmk thirdparty/Mast3r/asmk
cd thirdparty/Mast3r/asmk/cython/
cythonize *.pyx
cd ..
pip3 install faiss-gpu-cu12  # if cuda 12.x and python 3.8+
python3 setup.py build_ext --inplace
cd ..
mkdir -p checkpoints/
pip3 install pycolmap kapture kapture_localization
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```
Test the installation and add missing packages
```
cd ../..
pip3 install trimesh roma
python3 thirdparty/Mast3r/demo.py --weights thirdparty/Mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```
*If `asmk` ends up nested one level too deep, just move it up a directory.*

#### 5. Install hloc
```
cd thirdparty/hloc/
python -m pip install -e .
cd ../..
```

#### 6. Install NKSR (Neural Kernel Surface Reconstruction)
NKSR builds a CUDA extension — ensure the CUDA toolkit and nvcc are installed (`sudo apt install nvidia-cuda-toolkit`).
The `--no-build-isolation` flag is required because the build script uses `gitpython`, which is not listed as a build dependency.
```
cd thirdparty/nksr/package
pip install -e . --no-build-isolation
cd ../../..
```

#### 7. Install requirements
```
pip install -r requirements.txt
```

## Usage

Set the input video directory in `src/config.py` (`INPUT_VIDEO_DIR`) and run:
```
python3 src/main.py
```
Output is written to `../output/experiments/<video_name>/`.

## Differences from the Paper

The following components from the paper are not included in this repository:

- **VGGT**: We used the official VGGT repository to generate reconstructions with the parameters described in the paper.
- **ArUco dataset**: The dataset was collected as part of an internal study and cannot be shared publicly due to consent restrictions.
- **Cleft dataset**: Similarly, this dataset cannot be published for consent reasons.
- **Cleft-LoFTr**: The Cleft-LoFTr code is currently not publicly available.

## TODO

> Raising an issue or request for a specific feature will increase its priority.

- [ ] **PointCleanNet denoising** — We plan to release our PCN weights along with the training code and dataset augmentation pipeline.
- [ ] **Quantitative evaluation** — Since the data cannot be shared, the evaluation cannot be fully reproduced. We aim to provide the evaluation scripts in the near future.
- [ ] **ArUco calibration code** — While we are unable to share the dataset, we plan to release the code used for evaluation and dataset creation.

## External dependencies
- [Cutie](https://github.com/hkchengrex/Cutie.git) — `./thirdparty/Cutie`
- [MASt3R](https://github.com/naver/mast3r.git) — `./thirdparty/Mast3r`
- [hloc](https://github.com/cvg/Hierarchical-Localization.git) — `./thirdparty/hloc`
- [NKSR](https://github.com/nv-tlabs/nksr.git) — `./thirdparty/nksr`
