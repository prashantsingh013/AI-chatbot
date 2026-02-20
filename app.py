from flask import Flask, render_template, request, jsonify
import json
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import os

app = Flask(__name__)

# Load intents
data_file = open(os.path.join(os.path.dirname(__file__), 'intents.json')).read()
intents = json.loads(data_file)

# Prepare training data
training_sentences = []
training_labels = []
labels = []
responses = {}

for intent in intents['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Create a pipeline for the model
# Using a linear SVM (SGDClassifier) which is generally good for text classification
text_clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')), # Unigrams and bigrams
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])

# Train the model
text_clf.fit(training_sentences, training_labels)
print("Model trained successfully!")

def chatbot_response(msg):
    # Rule-based fallback or enhancement
    import datetime
    if 'time' in msg.lower() and 'what' in msg.lower():
        return f"The current time is {datetime.datetime.now().strftime('%H:%M')}."

    # Predict the intent
    try:
        prediction = text_clf.predict([msg])[0]
        # Get the confidence score (SGD with hinge loss doesn't provide probability directly, 
        # so we trust the prediction for this simple demo, or we could use calibrator)
        # For a simple chatbot, direct prediction is usually fine if the match is decent.
        
        # Rule-based fallback or enhancement could go here
        
        return random.choice(responses[prediction])
    except Exception as e:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if not userText:
        return jsonify({"response": "Please say something."})
    
    response = chatbot_response(userText)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
