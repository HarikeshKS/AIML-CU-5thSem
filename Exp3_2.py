import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample text data
corpus = [
    "This is a positive sentence.",
    "I love machine learning.",
    "Natural language processing is fascinating.",
    "Negative sentiment in this text.",
    "Not a fan of Naïve Bayes.",
]

# Labels for each text (0 for negative, 1 for positive)
labels = [1, 1, 1, 0, 0]

# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Initialize and train the Multinomial Naïve Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Calculate and print the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")