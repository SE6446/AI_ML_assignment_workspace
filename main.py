from typing import Callable
from tqdm import tqdm

from argparse import ArgumentParser, Namespace
import argparse

from datasets import Dataset

from search_based.main import sentiment_search_naive
from search_based.main import tokenize as search_tokenize

from shared.helper_function import get_dataset

parser: ArgumentParser = ArgumentParser()

_ = parser.add_argument("type", choices=['search','ml'])
_ = parser.add_argument("--skip_validation", default=False)
_ = parser.add_argument("--skip_training", default=False)
_ = parser.add_argument("--stored_model", default=argparse.SUPPRESS)

args: Namespace = parser.parse_args()

if args.skip_training and args.type == 'ml' and args.stored_model == "==SUPPRESS==":
    raise Exception("Cannot skip training of an ML model without a model file!")


FUNC: Callable[..., int] | None = sentiment_search_naive if args.type == 'search' else None
TOKENIZER: Callable[..., list[str]] | None = search_tokenize if args.type == 'search' else None

def run(function, tokenizer, prompt):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    return function(tokenizer(prompt))

def test_accuracy(dataset:Dataset):
    correct = 0
    for i in tqdm(dataset):
        y = run(FUNC, TOKENIZER, i['review_text'])
        if y == i['class_index']-1:
            correct += 1
    print(f"Accuracy: {correct/dataset.__len__()}\n{correct}/{dataset.__len__()}")

dataset = get_dataset(split="test[:500]")
print("Downloaded data. Running test")

test_accuracy(dataset)