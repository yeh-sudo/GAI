# GAI

This repository contains code for training Deep Image Prior (DIP) and Denoising Diffusion Probabilistic Models (DDPM) on images stored in the `/imgs` directory.

## Installation

To install the necessary dependencies, use the provided `requirements.txt` file. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

### Training the DIP Model

To train the Deep Image Prior (DIP) model, run the `DIP_train.py` script:

```bash
python DIP_train.py
```

### Training the DDPM Model

To train the Denoising Diffusion Probabilistic Model (DDPM), run the `main.py` script:

```bash
python main.py
```

## Directory Structure

Your project directory should look like this:

```
/imgs
    - image1.png
    - image2.png
    ...
DIP_train.py
main.py
requirements.txt
README.md
```

## References

This project is inspired by and references code from the following repositories:

- [Deep Image Prior](https://github.com/safwankdb/Deep-Image-Prior)
- [Denoising Diffusion Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main)
