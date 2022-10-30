# Eval_XAI_Robustness
This is the repository for paper ["SAFARI: Versatile and Efficient Evaluations for Robustness of Interpretability"](https://arxiv.org/abs/2208.09418). It proposes two evaluation metrics for robustness of interpretation from worst-case and probabilistic perspective, resepectively. The popular XAI methods, such as Integreted Gradient, LRP, and DeepLift are supported for the evaluation.
## Environment Setup
Requires Linux Platform with `Python 3.8.5`. We recommend to use anaconda for creating virtual environment. `requirements.txt` file contains the python packages required for running the code. Follow below steps for installing the packages:
- Create virtual environment and install necessary packages

	`conda create -n eval_xai --file requirements.txt`

- Activate virtual environment

	`conda activate eval_xai`
## Files
- `model` Directory contains scripts for training test models
- `checkpoints` Directory contains saved checkpoints for pre_trained test models

#### Note: 
- We only include a pre-trained test model for MNIST dataset due to the file size limit. For other dataset, please train the test models first.

- You may get error 'zipfile.BadZipFile: File is not a zip file' when downloading CelebA dataset. Google Drive has a daily maximum quota for any file. Try to mannually download from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) and unzip the dataset. Move to the folder `Datasets/celeba`

## How To Use 
The tool can be run for XAI robustness evaluation, test model training with the following commands.

