import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")


def get_top_k_predictions(input_sentence, k):
    encoding = tokenizer(input_sentence, return_tensors="pt")
    mask_idx = torch.where(encoding["input_ids"] == tokenizer.mask_token_id)[1]

    logits = model(**encoding).logits

    top_k_logits = torch.topk(logits[0, mask_idx.item(), :], k)[1]
    return [tokenizer.decode(pred) for pred in top_k_logits]


if __name__ == "__main__":
    input_sentence = input(
        "Enter a sentence with [MASK] in it (e.g. Where have my placed my [MASK]?)\n"
    )
    predictions = get_top_k_predictions(input_sentence, 5)
    winner = predictions[0]
    print(np.random.permutation(predictions))
    guess = input("Guess what BERT thinks is the masked word\n")
    if guess == winner:
        print("correct!")
    else:
        print(f"not quite, it was '{winner}'")
