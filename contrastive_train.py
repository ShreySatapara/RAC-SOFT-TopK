import os
import json
import argparse
import yaml
from trainers import CustomContrastiveTrainer
from dataset import ContrastiveDataset, ContrastiveDataCollator
from transformers import BertModel, BertTokenizer, BertConfig
from losses import *

LOSS_DICT = {
    "cosine": CosineContrastiveLoss,
    "maxsim": MaxSimContrastiveLoss,
    "l2": TripletLoss,
    "avgsim": AvgSimContrastiveLoss,
}


def load_config_from_yaml(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config):

    print("Config: ", config)
    # Save config file
    os.makedirs(config["log_dir"], exist_ok=True)
    with open(os.path.join(config["log_dir"], "config.json"), "w") as f:
        json.dump(config, f)

    bert_config = BertConfig.from_pretrained(config["model_name"])
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    model = BertModel.from_pretrained(config["model_name"], config=bert_config)

    loss = LOSS_DICT[config["loss_fn"]](
        margin=config["margin"], num_negatives=config["num_neg_samples"]
    )

    train_dataset = ContrastiveDataset(
        data_path=config["train_file"],
        num_neg_samples=config["num_neg_samples"],
        lang=config["language"],
    )

    eval_dataset = ContrastiveDataset(
        data_path=config["eval_file"],
        num_neg_samples=config["num_neg_samples"],
        lang=config["language"],
    )

    data_collator = ContrastiveDataCollator(
        tokenizer=tokenizer, max_length=config["max_seq_length"]
    )
    # Initialize Trainer with configurations from the YAML file
    trainer = CustomContrastiveTrainer(
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
        loss=loss,
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
