# # flask_api.py

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle, nltk, string, re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer

# nltk.download('stopwords')
# nltk.download('punkt')

# app = Flask(__name__)
# CORS(app)

# # Load model and vectorizer
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))
# stemmer = PorterStemmer()

# def clean_text(text):
#     text = word_tokenize(text)
#     text = " ".join(text)
#     text = [char for char in text if char not in string.punctuation]
#     text = ''.join(text)
#     text = [char for char in text if char not in re.findall(r"[0-9]", text)]
#     text = ''.join(text)
#     text = [word.lower() for word in text.split() if word.lower() not in stopwords.words('english')]
#     text = ' '.join(text)
#     text = list(map(lambda x: stemmer.stem(x), text.split()))
#     return " ".join(text)

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         sms = data.get("message", "")
#         cleaned = clean_text(sms)
#         vector_input = tfidf.transform([cleaned])
#         result = model.predict(vector_input)[0]
#         return jsonify({"prediction": "Spam" if result == 1 else "Not Spam"})
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, string, re, os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Configure local nltk_data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)

nltk.data.path.append(nltk_data_path)

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

stemmer = PorterStemmer()

def clean_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Lowercase and split
    words = text.lower().split()
    
    # Remove stopwords
    filtered = [word for word in words if word not in stopwords.words('english')]
    
    # Stemming
    stemmed = [stemmer.stem(word) for word in filtered]
    
    return " ".join(stemmed)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sms = data.get("message", "")
        cleaned = clean_text(sms)
        vector_input = tfidf.transform([cleaned])
        result = model.predict(vector_input)[0]
        return jsonify({"prediction": "Spam" if result == 1 else "Not Spam"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
