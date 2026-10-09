import sklearn
from sklearn import feature_extraction
from shared import get_dataset

dataset = get_dataset(split=f"train[:5000]")

#Put preprocessing function here
def process(sample):
    return None

dataset = dataset.map(process)

X, y = dataset["input_ids"], dataset["class_index"] - 1  # pyright: ignore[reportOperatorIssue]

