from argparse import ArgumentParser, Namespace
import argparse

parser: ArgumentParser = ArgumentParser()

_ = parser.add_argument("type", required=True, choices=['search','ml'])
_ = parser.add_argument("--skip_validation", default=False)
_ = parser.add_argument("--skip_training", default=False)
_ = parser.add_argument("--stored_model", default=argparse.SUPPRESS)

args: Namespace = parser.parse_args()

if args.skip_training and args.type == 'ml' and args.stored_model == "==SUPPRESS==":
    raise Exception("Cannot skip training of an ML model without a model file!")