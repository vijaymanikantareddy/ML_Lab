# Demonstrate object standardization such as replace social media slags from text.
slang_dict = {
    "lol": "laughing out loud",
    "omg": "oh my god",
    "btw": "by the way",
    "brb": "be right back",
    "idk": "I don't know",
    "tbh": "to be honest",
    "imho": "in my humble opinion",
    "afaik": "as far as I know",
    "smh": "shaking my head",
    "jk": "just kidding"
}

def standardize_text(text):
    words = text.split()
    standardized_words = []
    
    for word in words:
        if word.lower() in slang_dict:
            standardized_words.append(slang_dict[word.lower()])
        else:
            standardized_words.append(word)

    return ' '.join(standardized_words)

text = "lol that's so tbh idk why imho they would do that"
standardized_text = standardize_text(text)
print("Standardized text:", standardized_text)

'''
Output:
Standardized text: laughing out loud that's so to be honest I don't know why in my humble opinion 
they would do that
'''