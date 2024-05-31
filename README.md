# Retriever Augmented Classification using Differentiable Top-K Operator for Rumor Verification based on Evidence from Authorities

A code for implementing Retriver Augumented Classification using differential SOFT-TopK to perform a joint training of Retriever and Classifier.

### System Describtion

- Ubuntu 18.04.1 LTS
- Nvidia TESLA 32GB V100
- Python 3.10.13

### Environment Description

- Clone this repository
```
git clone <repository-name>
```
- Install miniconda/anaconda.
- Name of the environment that will be created by default is `rumour`
```
conda env create -f environment.yml
```

### Dataset 

Data used for training can be found [here](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task5/data)

### Training Procedure

> **For independent training.**

Look for `config/independent` folder.
- <u>For Retriever training</u> 

Change the `config.yaml` file in `retriever` folder according to the configs that you require.

Then run this commond for training.
```
bash scripts/retriever.sh
```
- <u>For Classifier training</u>

Change the `config.yaml` file in `classifier` folder according to the configs that you require.

Then run this commond for training.
```
bash scripts/classifier.sh
```

> **For Joint Training.**

Look for `config/joint_scratch` folder.

Change the `joint_config.yaml` file according to your requirements.
To train the model run the following code.

```
bash scripts/joint_train_scratch.sh
```

### Inference Procedure

Download the pretrained weights from the following [folder](https://drive.google.com/drive/folders/1xhuj7JfJRKZ8FCo20Bz2Ag3Go3j97oI0?usp=drive_link). 

In `config/infer.yaml` file conge the retriever and classifier paths with the path to the pretrained weights path and change the `output_file_retriever` and `output_file_classifier` paths accordingly.

Then run
```
bash scripts/infer.sh
```

Then in `scripts/evaluate_retrieval.sh` change `pred_file` with `output_file_retriever` and run the following code to get score for your retrieval.
```
bash scripts/evaluate_retrieval.sh
```

Then in `scripts/evaluate_verification.sh` change `pred_file` with `output_file_classifier` and run the following code to get score for your retrieval. 
```
bash scripts/evaluate_verification.sh
```