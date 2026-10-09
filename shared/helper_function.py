from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def get_dataset(name:str = "yassiracharki/Amazon_Reviews_Binary_for_Sentiment_Analysis", split="train") -> Dataset:
    return load_dataset(path=name, split=split)