

def tokenize(prompt:str)-> list[str]:
    return prompt.split(" ")


with open("positive.txt", "r") as f:
    
    positive_words: list[str] = [i for i in f.readlines()]
    f.close()

with open("negative.txt", mode="r") as f:
    negative_words:list[str] = [i for i in f.readlines()]
    f.close()


    
def sentiment_search_naive(prompt:list[str])-> int:
    sentiment:int = 0
    for word in prompt:
        if word in positive_words: # The GOOD words  :)
            sentiment += 1
        elif word in negative_words: # The BAD words >:(
            sentiment -= 1

    if sentiment > 0:
        return 1
    elif sentiment < 0:
        return 0
    else:
        print("Error: The sentiment was fully neutral.")

    return 0 #negative default

if __name__ == "__main__":
    prompts:list[str] = [] # Put some test prompts in here
    for prompt in prompts:
        tokens = tokenize(prompt)
        print(f"Prompt: {prompt}\n Result: {sentiment_search_naive(tokens)}\n=====")