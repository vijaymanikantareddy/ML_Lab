# Demonstrate Term Frequency – Inverse Document Frequency ( TF – IDF ) using python.
!pip install scikit-learn

import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
 "Machine learning is a subset of artificial intelligence.",
 "Python is a popular programming language for data science.",
 "Natural language processing is used in many applications such as chatbots.",
 "Topic modeling is a technique for extracting topics from text data."
]

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the documents and transform the documents into TF-IDF features
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (terms)
feature_name = vectorizer.get_feature_names_out()

# Print the TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Print the TF-IDF values for each term in each document
print("\nTF-IDF Values:")
for doc_index, doc in enumerate(documents):
    print(f"Document {doc_index + 1}:")
    for term_index, term in enumerate(feature_name):
        tfidf_value = tfidf_matrix[doc_index, term_index]
        if tfidf_value > 0:
            print(f"{term}: {tfidf_value:.4f}")

 '''
 Output:
    TF-IDF Values:
    Document 1:
    artificial: 0.3993
    intelligence: 0.3993
    is: 0.2084
    learning: 0.3993
    machine: 0.3993
    of: 0.3993
    subset: 0.3993
    Document 2:
    data: 0.3183
    for: 0.3183
    is: 0.2106
    language: 0.3183
    popular: 0.4037
    programming: 0.4037
    python: 0.4037
    science: 0.4037
    Document 3:
    applications: 0.3179
    as: 0.3179
    chatbots: 0.3179
    in: 0.3179
    is: 0.1659
    language: 0.2507
    many: 0.3179
    natural: 0.3179
    processing: 0.3179

 '''