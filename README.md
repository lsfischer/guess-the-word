# Guess what GPT-2 thinks

This is a very basic game in which you have to beat BERT in guessing what is the masked word in a GPT-2 synthesised
sentence.

## Playing the game

You start by providing an initial seed sentence (or pick a random one from the dataset of random sentences).

```python
>>> Enter an initial sentence or press enter for a random one
"Mary had a little lamb"
```

GPT-2 then takes this initial sentence and generates what it thinks is the next sentences. We then take this sentence
and randomly omit one of the words (replacing it with `[MASK]`).

```python
>>> Sentence: Mary had a little lamb in her hand I'm sorry she said I didn't mean to hurt you I [MASK] wanted you to know that I love you and I'm going to do everything in my power to
```

We then ask you and BERT to guess what is the masked word, to see who wins

```commandline
>>> Guess the masked word:
want
>>> You lose, BERT got it right. It was 'just'
```