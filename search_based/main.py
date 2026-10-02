

def tokenize(prompt:str)-> list[str]:

    return prompt.lower().split(" ")


with open("search_based/positive-words.txt", "r") as f:
    
    positive_words: list[str] = [i.strip("\n") for i in f.readlines()]
    f.close()

with open("search_based/negative-words.txt", mode="r") as f:
    negative_words:list[str] = [i.strip("\n") for i in f.readlines()]
    f.close()


    
def sentiment_search_naive(prompt:list[str])-> int:
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

if __name__ == "__main__":
    prompts:list[str] = ["I hated this", "I love this", "This is sick", "this is bad"] # Put some test prompts in here
    for prompt in prompts:
        tokens = tokenize(prompt)
        print(f"Prompt: {prompt}\n Result: {sentiment_search_naive(tokens)}\n=====")