from datasets import load_dataset

def get_dataset(name:str = "yassiracharki/Amazon_Reviews_Binary_for_Sentiment_Analysis", split="train")-> Dataset:
    return load_dataset(path=name, split=split)