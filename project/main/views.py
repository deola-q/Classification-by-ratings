import re
import torch
from django.shortcuts import render
from nltk import WordPunctTokenizer, PorterStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def index(request):
    if request.method == 'POST':
        review_text = request.POST.get('review_text')
        # review_text = preprocess_text(review_text)
        predicted_rate, sentiment = predict(review_text, 'main\model')

        return render(request, 'main/index.html', {
            'predicted_rate': predicted_rate,
            'sentiment': sentiment,
        })

    return render(request, 'main/index.html')

def predict(text, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    inputs = tokenizer(text, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    sem = 'positive' if predicted_class_id >= 4 else 'negative'
    return predicted_class_id, sem

# def preprocess_text(text):
#     text = text.replace('<br />', '')
#     text = text.lower()
#     text = re.sub("@\w+", '', text)
#     text = re.sub("http\w+", '', text)
#     text = re.sub("\d+", '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = text.strip()
#     stemmer = PorterStemmer()
#     tokens = WordPunctTokenizer().tokenize(text)
#     stemmed_words = ' '.join([stemmer.stem(word) for word in tokens])
#     return stemmed_words

