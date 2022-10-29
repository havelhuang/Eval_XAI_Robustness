# Eval_XAI_Robustness
This is the repository for paper ["SAFARI: Versatile and Efficient Evaluations for Robustness of Interpretability"](https://arxiv.org/abs/2208.09418). It proposes two evaluation metrics for robustness of interpretation from worst-case and probabilistic perspective, resepectively. The popular XAI methods, such as Integreted Gradient, LRP, and DeepLift are supported for the evaluation.
## Environment Setup
Requires Linux Platform with `Python 3.8.5`. We recommend to use anaconda for creating virtual environment. `requirements.txt` file contains the python packages required for running the code. Follow below steps for installing the packages:
- Create virtual environment and install necessary packages

	`conda create -n eval_xai --file requirements.txt`

- Activate virtual environment

	`conda activate eval_xai`
