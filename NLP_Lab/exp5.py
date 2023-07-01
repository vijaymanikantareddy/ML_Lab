# Implement topic modeling using Latent Dirichlet Allocation (LDA) in python.
import gensim
from gensim import corpora
# Sample documents
documents = [
 "Machine learning is a subset of artificial intelligence.",
 "Python is a popular programming language for data science.",
 "Natural language processing is used in many applications such as chatbots.",
 "Topic modeling is a technique for extracting topics from text data."
]
# Tokenize and preprocess the documents
tokenized_docs = [doc.lower().split() for doc in documents]
# Create a dictionary from the tokenized documents
dictionary = corpora.Dictionary(tokenized_docs)
# Create a corpus (term-document frequency)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
# Build the LDA model
num_topics = 2 # Number of topics to extract
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print the extracted topics and their top words
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx + 1}: {topic}")

# Get the topic distribution for a sample document
sample_doc = "Machine learning and data science go hand in hand."
sample_doc_bow = dictionary.doc2bow(sample_doc.lower().split())
sample_doc_topics = lda_model.get_document_topics(sample_doc_bow)
print(f"\nSample Document Topics: {sample_doc_topics}")

'''
Output:
Topic 1: 0.056*"language" + 0.056*"is" + 0.055*"such" + 0.055*"applications" + 0.055*"many"
Topic 2: 0.080*"a" + 0.080*"is" + 0.057*"for" + 0.034*"data." + 0.034*"topics"
Sample Document Topics: [(0, 0.30881903), (1, 0.691181)]
'''