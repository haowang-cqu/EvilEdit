# EvilEdit
This repository contains the code for the paper [EvilEdit: Backdooring Text-to-Image Diffusion Models in One Second](https://dl.acm.org/doi/10.1145/3664647.3680689) (ACM MM 2024).

## Environment

Step 1: Pull this repository.

```bash
git pull https://github.com/haowang-cqu/EvilEdit
cd EvilEdit
```

Step 2: Create a Conda environment and install PyTorch.

```bash
conda create -n eviledit python=3.10
conda activate eviledit
pip3 install torch torchvision
```

Step 3: Install other dependencies.

```bash
pip3 install -r requirements.txt
```

## Quick Start
You can run `edit.py` to embed the backdoor into stable-diffusion-v1-5. The default trigger is "beautiful cat", and the default backdoor target is "zebra" (check the code for more details).

```bash
conda activate eviledit
CUDA_VISIBLE_DEVICES=0 python edit.py
```
The backdoored U-Net weights will be stored in the `models` directory. Run the `show.ipynb` to see the effect of the backdoor.

## Citation

```
@inproceedings{wang2024eviledit,
  title={EvilEdit: Backdooring Text-to-Image Diffusion Models in One Second},
  author={Wang, Hao and Guo, Shangwei and He, Jialing and Chen, Kangjie and Zhang, Shudong and Zhang, Tianwei and Xiang, Tao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024}
}
```

