# Perform part of speech tagging on any textual data.
import nltk
from nltk import word_tokenize, pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
def perform_pos_tagging(text):
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens
    
# Example usage
text = "I love to explore new places and try different cuisines."
tagged_text = perform_pos_tagging(text)
print(tagged_text)

'''
Output:
[('I', 'PRP'), ('love', 'VBP'), ('to', 'TO'), ('explore', 'VB'), ('new', 'JJ'), ('places', 'NNS'), ('and', 'CC'), 
('try', 'VB'), ('different', 'JJ'), ('cuisines', 'NNS'), ('.', '.')]
'''