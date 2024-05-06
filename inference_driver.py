import argparse
import json
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import yaml
from models.scores import *
from inference import Inference
import jsonlines

SCORE_DICT = {
    "cosine": CosineScores,
    "dot_product": DotProductScores,
    "l2": L2Scores,
    "maxsim": MaxSimScores,
    "avgsim": AvgSimScores,
}


def main(configs):
    args = argparse.Namespace(**configs)
    tokenizer = BertTokenizer.from_pretrained(configs["model_name"])
    retriever_path = (
        configs["retriever_path"]
        if configs["retriever_path"] != "None"
        else configs["model_name"]
    )
    retriever = BertModel.from_pretrained(retriever_path)
    classifier_path = (
        configs["classifier_path"]
        if configs["classifier_path"] != "None"
        else configs["model_name"]
    )
    classifier = BertForSequenceClassification.from_pretrained(
        classifier_path, num_labels=configs["num_labels"]
    )
    scorer = SCORE_DICT[configs["score_fn"]]()

    # data = json.load(open(configs["data_file"]))
    with jsonlines.open(configs["data_file"]) as reader:
        data = list(reader)

    inference = Inference(retriever, classifier, scorer, tokenizer, args)
    print("############# INFERENCE STARTED #############")
    inference.infer_classifier_2(data)
    print("############# INFERENCE COMPLETED ############# \n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    main(configs)
