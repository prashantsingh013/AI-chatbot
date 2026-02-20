
# College AI Chatbot

A user-friendly AI chatbot for college/company FAQs built with Flask and Scikit-learn (Machine Learning).

## Features
- **Smart Responses**: Uses TF-IDF and SGDClassifier for intent classification.
- **Modern UI**: Clean, glassmorphism-inspired design.
- **Extensible Intents**: Easily add new questions and answers in `intents.json`.

## Setup
1. Install dependencies:
   ```bash
   pip install flask nltk scikit-learn numpy
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser and go to:
   `http://127.0.0.1:5000`

## Customization
- **Add FAQs**: Modify `intents.json` to add new tags, patterns, and responses.
- **Styling**: Update `static/style.css` to change the look and feel.
