from datasets import load_dataset, Dataset

def get_dataset(name:str = "yassiracharki/Amazon_Reviews_Binary_for_Sentiment_Analysis", split="train"):
    return load_dataset(path=name, split=split)