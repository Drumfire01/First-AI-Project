## BUDGETARY AI USING PYTHON ##

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Dataset
# try to find file or throw filenotfound error
try:
    df = pd.read_csv("data.csv")
    df['Category'] = df['Category'].str.lower().str.strip()
    df = df.drop_duplicates(subset="Transaction Description")
except FileNotFoundError:
    print("Error: FileNotFound")
    exit()

print("Sample Data:")
print(df.head())

# Check and handle missing values
print("Missing values in 'Transaction Description':", df['Transaction Description'].isnull().sum())
df.dropna(inplace=True)
# Step 2: Preprocess the Data
# Convert text to a Bag-of-Words representation
vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(df['Transaction Description'])

# Encode categories into numerical labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['Category'])

# Step 3: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
#print(classification_report(y_test, y_pred, target_names=encoder.classes_)) << ORIGINAL
#print(classification_report(y_test, y_pred, labels=encoder.transform(encoder.classes_), target_names=encoder.classes_))
# Check class distribution
print("Classes in Training Set:", set(y_train))
print("Classes in Test Set:", set(y_test))

# Generate the classification report
try:
    labels = encoder.transform(encoder.classes_)
    print(classification_report(y_test, y_pred, labels=labels, target_names=encoder.classes_))
except ValueError as e:
    print(f"Error generating classification report: {e}")
    print("Ensure all classes are represented in the test set.")



# Step 6: Interactive Interface
print("\n--- Personal Expense Categorizer ---")
while True:
    transaction = input("Enter a transaction description (or 'exit' to quit): ")
    if transaction.lower() == 'exit':
        break
    transaction_vectorized = vectorizer.transform([transaction])
    predicted_category = encoder.inverse_transform(model.predict(transaction_vectorized))
    print(f"Predicted Category: {predicted_category[0]}")

