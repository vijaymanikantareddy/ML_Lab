# Demonstrate noise removal for any textual data and remove regular expression pattern such 
#as hash tag from textual data.


import re
def remove_noise(text):
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove other noise (e.g., special characters, URLs, numbers)
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove special characters
    text = re.sub(r"\s+", " ", text) # Remove extra whitespaces
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) # Remove URLs
    text = re.sub(r"\b\d+\b", "", text) # Remove numbers
 
    return text.strip()
# Example usage
text = "Hello! This is a #sample text with #hashtags and some special characters!! 123 @acet.ac.in"
clean_text = remove_noise(text)
print(clean_text)


'''
Output:

Hello This is a text with and some special characters acetacin
'''
