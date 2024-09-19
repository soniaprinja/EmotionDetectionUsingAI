import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Download nltk resources (if not already)
nltk.download('punkt')
nltk.download('stopwords')

# Load GoEmotions Dataset (update with your dataset path)
data=pd.read_csv('goemotions_1.csv')

# Preprocessing the text data
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing to the dataset
data['clean_text'] = data['text'].apply(preprocess_text)

# Prepare data
X = data['clean_text']
y = data.drop(columns=['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear', 'clean_text'])

# Convert emotion columns to binary format (0 or 1)
y = y.apply(lambda x: x.astype(bool))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a MultiOutputClassifier with Logistic Regression
model = MultiOutputClassifier(LogisticRegression())
model.fit(X_train_tfidf, y_train)

# Evaluate on test data
y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred, target_names=y.columns))

def predict_emotions(sentence):
    clean_sentence = preprocess_text(sentence)
    sentence_tfidf = tfidf.transform([clean_sentence])
    predictions = model.predict(sentence_tfidf)
    # Get emotions with value True
    predicted_emotions = [emotion for emotion, present in zip(y.columns, predictions[0]) if present]
    return ', '.join(predicted_emotions) if predicted_emotions else 'No emotion detected'

# Example of predicting emotions for a new sentence
new_sentence = "I'm feeling really happy today!"
predicted_emotions = predict_emotions(new_sentence)
print(f"Predicted Emotion: {predicted_emotions}")
