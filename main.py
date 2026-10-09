import argparse
from argparse import ArgumentParser, Namespace
from typing import Callable
import json

from datasets import Dataset
from tqdm import tqdm

from search_based.main import *
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


FUNC: Callable[..., int] | None = sentiment_greedy_search_naive if args.type == 'search' else None
TOKENIZER: Callable[..., list[str]] | None = search_tokenize if args.type == 'search' else None

def run(function, tokenizer, prompt):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    return function(tokenizer(prompt))

def test_accuracy(dataset:Dataset):
    log = []
    correct = 0
    for i in tqdm(dataset):
        y, score = run(FUNC, TOKENIZER, i['review_text'])
        if y == i['class_index']-1:
            correct += 1
        log.append({"text":i["review_text"],"y":y,"ground_truth":i["class_index"]-1, "pre-softmax":score})
    print(f"Accuracy: {correct/dataset.__len__()}\n{correct}/{dataset.__len__()}")
    with open("log.json","w") as f:
        f.write(json.dumps(log, indent=2))


dataset = get_dataset(split="test[:50000]")
print("Downloaded data. Running test")

test_accuracy(dataset)
