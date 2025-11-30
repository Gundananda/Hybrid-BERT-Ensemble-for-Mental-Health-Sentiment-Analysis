from flask import Flask, render_template, request
import joblib
import numpy as np
import re
import nltk
import tensorflow_hub as hub
import tensorflow_text  # Needed to register ops
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Flask app
app = Flask(__name__)

# Load models
stacking_model = joblib.load("/Users/shindu/Desktop/mental_health_app/stacking_model.pkl")
vectorizer = joblib.load("/Users/shindu/Desktop/mental_health_app/tfidf_vectorizer.pkl")
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Label Map
label_map = {
    0: "Normal",
    1: "Depression",
    2: "Suicidal",
    3: "Anxiety",
    4: "Bipolar",
    5: "Stress",
    6: "Personality Disorder"
}

# Text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

def further_clean_text(text):
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

def preprocess(text):
    text = clean_text(text)
    text = further_clean_text(text)
    return lemmatize_text(text)

def get_bert_embedding(texts):
    preprocessed = bert_preprocess(texts)
    return bert_encoder(preprocessed)['pooled_output'].numpy()

def generate_features(text):
    cleaned = preprocess(text)
    bert_emb = get_bert_embedding([cleaned])
    tfidf_feat = vectorizer.transform([cleaned]).toarray()
    return np.hstack((bert_emb, tfidf_feat))

def predict(text):
    features = generate_features(text)
    pred = stacking_model.predict(features)
    return label_map[int(pred[0])]

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_text = request.form['user_text']
        prediction = predict(user_text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
