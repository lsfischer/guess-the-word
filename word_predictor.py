import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")


def get_bert_prediction(input_sentence: str) -> str:
    """
    Given an input_sentence with a [MASK] token gets the top BERT prediction for it

    Params
    ------
        input_sentence - The input sentence assumed to have a [MASK] token in it
    Returns
    -------
        A list of words corresponding to the words the language model thinks are the most likely, sorted by their
        likelihood
    """
    encoding = tokenizer(input_sentence, return_tensors="pt")

    # find the index of where the [MASK] token is
    mask_idx = torch.where(encoding["input_ids"] == tokenizer.mask_token_id)[1]

    # Get the most likely prediction
    logits = model(**encoding).logits
    prediction = torch.argmax(logits[0, mask_idx.item(), :])

    return tokenizer.decode(prediction)
