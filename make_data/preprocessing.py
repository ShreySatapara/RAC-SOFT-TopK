import json
import sys
import os
import re
import emoji
from typing import List
import argparse

### Preprocessing of the text ###
"""
1. Remove the links from the tweets
2. Replace the emojis with their description and with their respective *unicode*(NOT DONE)
3. Add dummy sentences to the list of tweets (timeline) to make sure that the length of the timeline is atleat 5.
"""


def remove_links(tweet: str):
    tweet = re.sub("https\S*", "", tweet)
    tweet = re.sub("http\S*", "", tweet)
    tweet = re.sub("Https\S*", "", tweet)
    tweet = re.sub("Http\S*", "", tweet)
    return tweet


def replace_emojis_with_desc(tweet: str, language: str = "en"):
    return emoji.demojize(tweet, language=language)


# For a single tweet
def preprocess_tweet(tweet: str, language: str = "en"):
    tweet = remove_links(tweet)
    tweet = replace_emojis_with_desc(tweet, language=language)
    return tweet


# For timeline and evidence
def preprocess_tweets(timeline: List[str], language: str = "en"):
    for i in range(len(timeline)):
        # only considering the tweet, not the meta information
        timeline[i] = [
            preprocess_tweet(timeline[i][-1], language=language),
        ]
    return timeline


# read a json file and return preprocessed data
def make_data(
    file_path: str, folder_path: str, language: str = "en", to_return: bool = False
):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found {file_path}")
    with open(file_path, "r") as f:
        data_list = json.load(f)
    print(
        f"Data loaded successfully, {len(data_list)} samples found. Preprocessing data..."
    )
    # folder_path = os.path.abspath(os.path.join(output_path, os.pardir))
    if os.path.isfile(os.path.join(folder_path, "meta.json")):
        with open(os.path.join(folder_path, "meta.json"), "r") as f:
            temp = json.load(f)
            labels_2_ids = temp["labels_2_ids"]
            ids_2_labels = temp["ids_2_labels"]
    else:
        labels_2_ids, ids_2_labels = {}, {}
    new_data_list = []
    counter = 0
    for data in data_list:
        new_data = {}
        new_data["id"] = data["id"]
        label = data["label"].lower()
        if label not in labels_2_ids.keys():
            labels_2_ids[label] = counter
            ids_2_labels[counter] = label
            counter += 1
        new_data["label"] = labels_2_ids[label]
        new_data["rumor"] = preprocess_tweet(data["rumor"], language=language)
        new_data["timeline"] = preprocess_tweets(data["timeline"], language=language)
        new_data["evidence"] = preprocess_tweets(data["evidence"], language=language)
        new_data_list.append(new_data)
    print("Data preprocessed successfully")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = file_path.split("/")[-1]
    with open(os.path.join(folder_path, file_name), "w") as f:
        json.dump(new_data_list, f, ensure_ascii=False)
    print(f"Data saved to {os.path.join(folder_path, file_name)}")
    if not os.path.isfile(os.path.join(folder_path, "meta.json")):
        temp = {"labels_2_ids": labels_2_ids, "ids_2_labels": ids_2_labels}
        with open(os.path.join(folder_path, "meta.json"), "w") as f:
            json.dump(temp, f)
        print("Meta data saved successfully!!!!")
    if to_return:
        return new_data_list, labels_2_ids, ids_2_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to the json file containing the data",
        required=True,
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        help="Path to the folder where the preprocessed data will be saved",
        required=True,
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language of the data",
        default="en",
        choices=["en", "ar"],
    )
    args = parser.parse_args()
    make_data(args.file_path, args.folder_path, args.language)
