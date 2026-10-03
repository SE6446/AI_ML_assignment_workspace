

def tokenize(prompt:str)-> list[str]:

    return prompt.lower().split(" ")


with open("search_based/positive-words.txt", "r") as f:

    positive_words: list[str] = [i.strip("\n") for i in f.readlines()]
    f.close()

with open("search_based/negative-words.txt", mode="r") as f:
    negative_words:list[str] = [i.strip("\n") for i in f.readlines()]
    f.close()

#Taken from https://github.com/cjhutto/vaderSentiment/blob/master/vaderSentiment/vaderSentiment.py#L33
NEGATE_WORDS = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

def sentiment_greedy_search_naive(prompt:list[str])-> int:
    sentiment:int = 0
    for word in prompt:
        if any(item in word for item in positive_words): # The GOOD words  :)
            sentiment += 1
        elif any(item in word for item in negative_words): # The BAD words >:(
            sentiment -= 1

    if sentiment > 0:
        return 1
    elif sentiment < 0:
        return 0

    return 0 #negative default

def sentiment_greedy_search_with_negations(prompt:list[str]) -> tuple[int, int]:
    sentiment:int = 0
    negated = False
    for word in prompt:
        if any(item in word for item in positive_words): # The GOOD words  :)
            sentiment += 1 if not negated else -1

        elif any(item in word for item in negative_words): # The BAD words >:(
            sentiment -= 1 if not negated else -1 #what the fuck?

        elif any(item in word for item in NEGATE_WORDS) and negated == False:
            negated = True
            continue
        negated = False

    if sentiment > 0:
        return 1, sentiment
    else: # We swap the negative check with an else as they are analogus.
        return 0, sentiment

if __name__ == "__main__":
    prompts:list[str] = ["I hated this", "I love this", "This is sick", "this is bad"] # Put some test prompts in here
    for prompt in prompts:
        tokens = tokenize(prompt)
        print("Naive")
        print(f"Prompt: {prompt}\n Result: {sentiment_greedy_search_naive(tokens)}\n=====")
        print("Greedy")
        print(f"Prompt: {prompt}\n Result: {sentiment_greedy_search_with_negations(tokens)}\n=====")
