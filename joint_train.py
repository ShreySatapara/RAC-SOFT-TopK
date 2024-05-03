import os
import json
import yaml
import torch
import random
import argparse
import numpy as np
from transformers import BertTokenizer
from models import RACModel
from dataset import JointDataset2, JointDataCollator
from trainers import CustomJointTrainer


def load_config_from_yaml(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    # Set random seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    print("Config: ", config)
    # Save config file
    with open(os.path.join(config["log_dir"], "config.json"), "w") as f:
        json.dump(config, f)

    model_config = argparse.Namespace(**config["model"])
    model = RACModel(config=model_config)
    tokenizer_classifier = BertTokenizer.from_pretrained(
        config["model"]["bert_model_name_classifier"]
    )
    tokenizer_retriever = BertTokenizer.from_pretrained(
        config["model"]["bert_model_name_retriever"]
    )

    train_dataset = JointDataset2(
        data_path=config["train_file"],
        lang=config["language"],
        doc_size=config["doc_size"],
    )

    eval_dataset = JointDataset2(
        data_path=config["eval_file"],
        lang=config["language"],
        doc_size=config["doc_size"],
    )

    data_collator = JointDataCollator(
        tokenizer=(tokenizer_classifier, tokenizer_retriever),
        max_length=config["max_seq_length"],
    )
    # Initialize Trainer with configurations from the YAML file
    trainer = CustomJointTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=eval_dataset,
        batch_size=config["batch_size"],
        lr=config["lr"],
        max_grad_norm=config["max_grad_norm"],
        device=config["device"],
        distributed=config["distributed"],
        patience=config["patience"],
        log_dir=config["log_dir"],
        data_collator=data_collator,
    )
    print("############# TRAINING STARTED #############")
    trainer.train(config["epochs"])
    print("############# TRAINING COMPLETED ############# \n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    main(config)
