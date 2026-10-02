

def tokenize(prompt:str)-> list[str]:
    return prompt.split(" ")


with open("positive.txt", "r") as f:
    
    positive_words: list[str] = [i for i in f.readlines()]
    f.close()

with open("negative.txt", mode="r") as f:
    negative_words:list[str] = [i for i in f.readlines()]
    f.close()




def sentiment_search_naive(prompt:list[str])-> int:
    return 0


if __name__ == "__main__":
    prompts:list[str] = [] # Put some test prompts in here
    for prompt in prompts:
        tokens = tokenize(prompt)
        print(f"Prompt: {prompt}\n Result: {sentiment_search_naive(tokens)}\n=====")