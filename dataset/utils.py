import torch
from typing import Dict, Any, List
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def tokenize_sample(
    sample: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Tokenize the input sample.

    Args:
        sample (Dict[str, Any]): Input sample.
        tokenizer: Tokenizer for tokenizing the text.
        max_length (int): Maximum length of the input text.

    Returns:
        Dict[str, Any]: Tokenized sample.
    """

    # Tokenize the rumor
    rumor = tokenizer.encode_plus(
        sample["rumor"],
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Tokenize the evidence
    evidence = tokenizer.encode_plus(
        sample["evidence"],
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Tokenize the negative samples
    negatives = []
    for negative in sample["negatives"]:
        negative = tokenizer.encode_plus(
            negative,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        negatives.append(negative)

    return {"rumor": rumor, "evidence": evidence, "negatives": negatives}
