from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def generate_sentence(input_sentence: str) -> str:
    """
    Given an input sentence, use GPT-2 to generate the next sentence

    Params
    ------
        input_sentence - The initial seed sentence to be used by GPT-2 to generate the following sentence

    Returns
    -------
        A new sentence containing the input sentence plus a generated one using GPT-2 and beam search
    """
    encoding = tokenizer(input_sentence, return_tensors="pt")
    beam_output = model.generate(**encoding, num_beams=5, early_stopping=True)
    return tokenizer.decode(beam_output[0], skip_special_tokens=True)
