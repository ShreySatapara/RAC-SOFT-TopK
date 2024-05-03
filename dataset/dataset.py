import json
import random
from typing import List, Tuple, Dict, Union
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        lang: str = "en",
    ):

        self.data_path = data_path  # Path to the data file
        # Number of negative samples (atmost num_neg_samples)
        self.lang = lang  # Language of the data
        # Load data
        self._load_data()

    def __len__(self) -> int:
        return len(self.indexes)

    def _load_data(self) -> List[Dict]:
        # Load the data
        with open(self.data_path, "r") as f:
            data = json.load(f)
        self.data = data
        # Get the number of samples with evidences
        indexes = []
        for i, sample in enumerate(self.data):
            if sample["evidence"]:
                for j in range(len(sample["evidence"])):
                    indexes.append((i, j))
        self.indexes = indexes

    def __getitem__(self, index) -> Tuple[str, str, List[str]]:
        raise NotImplementedError


## Dataset for training the Retriever model in contrastive setting ##
class ContrastiveDataset(BaseDataset):
    """
    Dataset for contrastive learning.

    Args:
        data_path (str): Path to the data file.
        tokenizer (BertTokenizer): Tokenizer for tokenizing the text.
        max_length (int): Maximum length of the input text.

    Returns: (__getitem__)
        (rumor, evidence(positive), List[timeline(negative)]): Tuple of two texts and a list of k negative texts.
    """

    def __init__(
        self,
        data_path: str,
        num_neg_samples: int = 3,
        lang: str = "en",
    ):
        super().__init__(data_path, lang)
        self.num_neg_samples = num_neg_samples

    def __getitem__(self, index) -> Tuple[str, str, List[str]]:
        # Get the index of the sample
        i, j = self.indexes[index]

        # Get the rumor and evidence
        rumor = self.data[i]["rumor"]
        evidence = self.data[i]["evidence"][j][0]

        # Get the negative samples
        negatives = []
        if len(self.data[i]["timeline"]) > (
            self.num_neg_samples + len(self.data[i]["evidence"])
        ):
            while len(negatives) < self.num_neg_samples:
                neg = random.choice(self.data[i]["timeline"])
                if neg not in self.data[i]["evidence"] and neg[0] not in negatives:
                    negatives.append(neg[0])
        else:
            for neg in self.data[i]["timeline"]:
                if neg not in self.data[i]["evidence"]:
                    negatives.append(neg[0])

        # If no negative samples, add dummy samples
        while len(negatives) < self.num_neg_samples:
            if self.lang == "en":
                negatives.append("This is a dummy sample.")
            elif self.lang == "ar":
                negatives.append("هذا عينة وهمية.")

        return {"rumor": rumor, "evidence": evidence, "negatives": negatives}


## Dataset for training the classifier model  for 3 class prediction ##


class ClassifierDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        lang: str = "en",
    ):
        super().__init__(data_path, lang)

    def _load_data(self) -> List[Dict]:
        # Load the data
        with open(self.data_path, "r") as f:
            data = json.load(f)
        self.data = data
        # Get the number of samples with evidences
        indexes = []
        counter = -1
        for i, sample in enumerate(self.data):
            if sample["evidence"]:
                for j in range(len(sample["evidence"])):
                    indexes.append((i, j))
                    counter += 1
                if counter % 2 == 0:
                    indexes.append((i, -1))
            else:
                indexes.append((i, -1))
        self.indexes = indexes

    def __getitem__(self, index) -> Tuple[str, str, List[str]]:
        # Get the index of the sample
        i, j = self.indexes[index]
        # print(len(self.data))
        # print(i, j)

        # Get the rumor, evidence and label
        evidence = self.data[i]["evidence"]
        timeline_wo_evidence = []
        if len(evidence) > 0:
            for timeline in self.data[i]["timeline"]:
                if timeline not in evidence:
                    timeline_wo_evidence.append(timeline)
        else:
            timeline_wo_evidence = self.data[i]["timeline"]
        if j >= 0:
            rumor = self.data[i]["rumor"]
            evidence = evidence[j][0]
            label = self.data[i]["label"]
        else:
            rumor = self.data[i]["rumor"]
            label = 2  # No evidence, not verifiable
            if len(timeline_wo_evidence) > 0:
                idx = random.choice(range(len(timeline_wo_evidence)))
                evidence = timeline_wo_evidence[idx][0]
            else:
                if self.lang == "en":
                    evidence = "This is a dummy sample."
                elif self.lang == "ar":
                    evidence = "هذا عينة وهمية."

        return {"rumor": rumor, "evidence": evidence, "label": label}


## Dataset for end-to-end joint training of the Retriever and Classifier models ##


def match_sentences(sentence1: str, sentence2: str):
    sentence1 = sentence1.split()
    sentence2 = sentence2.split()
    if len(sentence1) != len(sentence2):
        return False
    res = True
    for i in range(len(sentence1)):
        if sentence1[i] != sentence2[i]:
            res = False
            break
    return res


class JointDataset(Dataset):
    def __init__(self, data_path: str, lang: str, doc_size: int = 64) -> None:
        super().__init__()
        self.data_path = data_path
        self.lang = lang
        self.doc_size = doc_size

        # Load the data
        self._load_data()

    def _load_data(self) -> None:
        # Load the data
        with open(self.data_path, "r") as f:
            data = json.load(f)
        self.data = data

        self.indexes = []
        for i in range(len(self.data)):
            if self.data[i]["evidence"]:
                self.indexes.append(i)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index) -> Dict[str, Union[str, int, List[str], np.ndarray]]:
        sample = self.data[self.indexes[index]]

        data_sample = {}
        data_sample["query"] = sample["rumor"]
        random.shuffle(sample["timeline"])
        timeline = [x[0] for x in sample["timeline"]]
        data_sample["docs"] = timeline
        len_docs = len(timeline)
        A_list = []

        for evidence in sample["evidence"]:
            A = np.zeros(len_docs)
            for i, doc in enumerate(timeline):
                if match_sentences(evidence[0], doc):
                    A[i] = 1
            A_list.append(A)
        data_sample["soft_A"] = np.array(A_list).T

        data_sample["label"] = sample["label"]
        return data_sample


class JointDataset2(Dataset):
    def __init__(self, data_path, lang, doc_size, critical_limit=5) -> None:
        super().__init__()
        self.data_path = data_path
        self.lang = lang
        self.doc_size = doc_size
        self.critical_limit = critical_limit

        # Load the data
        self._load_data()

    def _load_data(self) -> None:
        # Load the data
        with open(self.data_path, "r") as f:
            data = json.load(f)
        self.data = data

        self.indexes = []

        # Create indexes
        # [sample_index, evidence_yes, {all, few, none}, remaining_docs_index, evidence_splits]
        # [sample_index, evidence_no, remaining_docs]
        # evidence_yes: 1 if evidence is present, 0 if not
        # {all, few, none}: 1 if all, 2 if few, 3 if none
        # remaining_docs_index: Index of the remaining docs
        # evidence_splits: Index of the evidence splits if all then take all evidence

        for i in range(len(self.data)):
            index = [i]
            if self.data[i]["evidence"]:
                index.append(1)
                if len(self.data[i]["evidence"]) == len(self.data[i]["timeline"]):
                    index.append(1)
                    index.append(-1)
                    self.indexes.append(index)
                elif len(self.data[i]["evidence"]) > 0:
                    if len(self.data[i]["evidence"]) <= 5:
                        evidence_splits = "all"
                        for j in [1, 2, 3]:
                            index_ = index.copy()
                            index_.append(j)
                            index_.append(-1)
                            index_.append(evidence_splits)
                            self.indexes.append(index_)
                            if len(self.data[i]["timeline"]) > self.doc_size:
                                self.indexes.append(index_)
                    else:
                        evidence_splits = [
                            k for k in range(0, len(self.data[i]["evidence"]), 5)
                        ]
                        for e_s in evidence_splits:
                            index_ = index.copy()
                            index_.append(1)
                            index_.append(1)
                            index_.append(e_s)
                            self.indexes.append(index_)
                            if len(self.data[i]["timeline"]) > self.doc_size:
                                self.indexes.append(index_)
            elif len(self.data[i]["timeline"]) > 0:
                index.append(0)
                if len(self.data[i]["timeline"]) >= 2 * self.doc_size:
                    index.append(-1)
                    self.indexes.append(index)
                    if len(self.data[i]["timeline"]) >= 4 * self.doc_size:
                        index_ = index.copy()
                        index_.append(-1)
                        self.indexes.append(index_)
                    if len(self.data[i]["timeline"]) >= 5 * self.doc_size:
                        index_ = index.copy()
                        index_.append(-1)
                        self.indexes.append(index_)
                else:
                    index.append(-1)
                    self.indexes.append(index)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index) -> Dict[str, Union[str, int, List[str], np.ndarray]]:
        sample = self.indexes[index]
        # print(sample)
        data_sample = {
            "query": self.data[sample[0]]["rumor"],
            "docs": [],
            "soft_A": None,
            "label": None,
        }
        if sample[1]:
            all_evidence = self.data[sample[0]]["evidence"]
            # print(all_evidence)
            timeline_wo_evidence = []
            for i, timeline in enumerate(self.data[sample[0]]["timeline"]):
                if timeline not in all_evidence:
                    timeline_wo_evidence.append(timeline)
            if sample[-1] == "all":
                evidence = all_evidence
            else:
                evidence = all_evidence[sample[-1] : sample[-1] + 5]
            # print(
            #     len(evidence),
            #     len(all_evidence),
            #     len(timeline_wo_evidence),
            #     len(self.data[sample[0]]["timeline"]),
            # )
            if sample[2] == 1:
                data_sample["label"] = self.data[sample[0]]["label"]
                required_timeline = [x[0] for x in evidence]
                if len(timeline_wo_evidence) > (self.doc_size - len(required_timeline)):
                    required_timeline += random.sample(
                        [x[0] for x in timeline_wo_evidence],
                        self.doc_size - len(required_timeline),
                    )
                else:
                    required_timeline += [x[0] for x in timeline_wo_evidence]
                random.shuffle(required_timeline)
                data_sample["docs"] = required_timeline
            elif sample[2] == 2:
                data_sample["label"] = self.data[sample[0]]["label"]
                len_evidence = len(evidence)
                if len_evidence > 1:
                    required_timeline = [x[0] for x in evidence]
                else:
                    evi_idxs = random.sample(range(len_evidence), len_evidence // 2)
                    required_timeline = [evidence[i][0] for i in evi_idxs]
                if len(timeline_wo_evidence) > self.doc_size - len(required_timeline):
                    required_timeline += random.sample(
                        [x[0] for x in timeline_wo_evidence],
                        self.doc_size - len(required_timeline),
                    )
                else:
                    required_timeline += [x[0] for x in timeline_wo_evidence]
                random.shuffle(required_timeline)
                data_sample["docs"] = required_timeline
            elif sample[2] == 3:
                data_sample["label"] = 2
                if len(timeline_wo_evidence) > self.doc_size:
                    doc_idxs = random.sample(
                        range(len(timeline_wo_evidence)), self.doc_size
                    )
                    required_timeline = [timeline_wo_evidence[i][0] for i in doc_idxs]
                else:
                    required_timeline = [x[0] for x in timeline_wo_evidence]
                random.shuffle(required_timeline)
                data_sample["docs"] = required_timeline
            else:
                raise IndexError("Invalid sample index")
            if len(data_sample["docs"]) < self.critical_limit:
                data_sample["docs"] = data_sample["docs"] + [
                    "This is a dummy sample."
                ] * (self.critical_limit - len(data_sample["docs"]))
            A_list = []
            temp = [x[0] for x in evidence]
            for i, sen in enumerate(data_sample["docs"]):
                A = np.zeros(len(data_sample["docs"]))
                if sen in temp:
                    A[i] = 1
                    A_list.append(A)
            if len(A_list) > 0:
                data_sample["soft_A"] = np.array(A_list).T
        else:
            data_sample["label"] = 2
            timeline = self.data[sample[0]]["timeline"]
            len_timeline = len(timeline)
            if len_timeline > self.doc_size:
                doc_idxs = random.sample(range(len_timeline), self.doc_size)
                required_timeline = [timeline[i] for i in doc_idxs]
            else:
                required_timeline = timeline
            random.shuffle(required_timeline)
            data_sample["docs"] = [x[0] for x in required_timeline]
            if len(data_sample["docs"]) < self.critical_limit:
                data_sample["docs"] = data_sample["docs"] + [
                    "This is a dummy sample."
                ] * (self.critical_limit - len(data_sample["docs"]))
        return data_sample
