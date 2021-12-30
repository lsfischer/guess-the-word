import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")


def get_top_k_predictions(input_sentence: str, k: int = 5) -> List[str]:
    """
    Given an input_sentence with a [MASK] token, generates the top K predictions for the masked word

    Params
    ------
        input_sentence - The input sentence assumed to have a [MASK] token in it
        k - The number of predictions to generate for the masked word

    Returns
    -------
        A list of words corresponding to the words the language model thinks are the most likely, sorted by their
        likelihood
    """
    encoding = tokenizer(input_sentence, return_tensors="pt")

    # find the index of where the [MASK] token is
    mask_idx = torch.where(encoding["input_ids"] == tokenizer.mask_token_id)[1]

    # Get the most likely predictions
    logits = model(**encoding).logits
    top_k_logits = torch.topk(logits[0, mask_idx.item(), :], k)[1]

    return list(map(tokenizer.decode, top_k_logits))
