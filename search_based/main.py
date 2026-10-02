

def tokenize(prompt:str)-> list[str]:
    return [""]

    
def sentiment_search_naive(prompt:list[str])-> int:
    sentiment = 0
    for i in prompt:
        if prompt[i] in positive_words: # The GOOD words
            sentiment = sentiment + 1;
        elif prompt[i] in negative_words: # The BAD words
            sentiment = sentiment - 1;

    if sentiment > 0:
        return 1
    elif sentiment < 0:
        return 0
    else:
        print("Error: The sentiment was fully neutral.")

    return 0 #negative default