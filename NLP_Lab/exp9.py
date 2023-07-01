    # Convert text to vectors ( using term frequency ) and apply cosine similarity to provide 
    # closeness among two text.

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Sample documents
    documents = [
    "I love natural language processing.",
    "Machine learning is fascinating.",
    "Python is widely used in data science."
    ]

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the documents to obtain the term frequency (TF) vectors
    tf_vectors = vectorizer.fit_transform(documents).toarray()

    # Calculate the cosine similarity between two documents
    doc1 = tf_vectors[0]
    doc2 = tf_vectors[1]
    similarity = cosine_similarity([doc1], [doc2])[0][0]
    print(f"Text 1: {documents[0]}")
    print(f"Text 2: {documents[1]}")
    print(f"Cosine Similarity: {similarity:.4f}")

    '''
    Output:
    Text 1: I love natural language processing.
    Text 2: Machine learning is fascinating.
    Cosine Similarity: 0.0000

    '''