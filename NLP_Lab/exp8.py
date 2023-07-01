# Apply support vector machine for text classification.

!pip install -U scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Sample data
documents = [
 ("I love natural language processing.", "positive"),
 ("Machine learning is fascinating.", "positive"),
 ("Python is widely used in data science.", "positive"),
 ("I dislike noisy environments.", "negative"),
 ("This movie is terrible.", "negative"),
 ("I feel sad today.", "negative")
]

# Split the data into features and labels
texts, labels = zip(*documents)

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the text data into TF-IDF features
features = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC()

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(report)

'''
Output:
    Accuracy: 0.0
    Classification Report:
    precision recall f1-score support
    negative 0.00 0.00 0.00 0.0
    positive 0.00 0.00 0.00 2.0
    accuracy 0.00 2.0
    macro avg 0.00 0.00 0.00 2.0
    weighted avg 0.00 0.00 0.00 2.0
'''