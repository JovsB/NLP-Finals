from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from operator import itemgetter

from test_spam import detect_spam
from utils import split_sentences

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
@cross_origin(origin='http://localhost:3000')
def index():
    data = request.json
    text, use_spam, use_toxicity, use_sentiment = itemgetter('text', 'useSpam', 'useToxicity', 'useSentiment')(data)
    
    sentences = split_sentences(text)
    results = {}
    
    if use_spam:
        spam_results, spam_count = detect_spam(sentences)
        results.update({
            'spam': {
                'results': spam_results,
                'count': spam_count
            }
        })
    
    return jsonify(results)

app.run(debug=True)