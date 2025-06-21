import pandas as pd
import string
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 📥 Download necessary NLTK data
nltk.download('stopwords')

# 🧾 Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1', usecols=[0, 1], names=["label", "message"], skiprows=1)

# 🧹 Basic cleanup
df.dropna(subset=["message"], inplace=True)
df["message"] = df["message"].astype(str)

# 🧼 Clean and normalize labels
df['label'] = df['label'].str.strip().str.lower()
df = df[df['label'].isin(['ham', 'spam'])]

# 🔧 Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    words = text.split()  # ✅ avoids NLTK's word_tokenize to prevent errors
    return " ".join([
        ps.stem(word) for word in words
        if word not in stop_words and word not in string.punctuation
    ])

# 🧽 Apply text cleaning
df['cleaned'] = df['message'].apply(clean_text)

# 🎯 Create labels
y = df['label'].map({'ham': 0, 'spam': 1})
X = df['cleaned']

# 📊 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Build training pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('clf', MultinomialNB())
])

# 🏋️ Train the model
pipeline.fit(X_train, y_train)

# 💾 Save the model pipeline
with open('spam_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model pipeline trained and saved successfully as 'spam_pipeline.pkl'")
