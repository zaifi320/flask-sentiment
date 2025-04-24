from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import torch
from statistics import median
import numpy as np

app = Flask(__name__)

# Configure models with refined normalization
MODELS = [
    {
        "name": "nlptown/bert-base-multilingual-uncased-sentiment",
        "type": "stars",
        "weight": 0.5,
        "normalize": lambda x: int(x['label'].split()[0])  # Direct 1-5 star rating
    },
    {
        "name": "siebert/sentiment-roberta-large-english",
        "type": "positive_negative",
        "weight": 0.3,
        "normalize": lambda x: (4.5 * x['score'] + 0.5) if x['label'] == 'POSITIVE' else (5.5 - 4.5 * x['score'])
    },
    {
        "name": "finiteautomata/bertweet-base-sentiment-analysis",
        "type": "three_label",
        "weight": 0.2,
        "normalize": lambda x: {
            'POS': 3.5 + 1.5 * x['score'],  # Positive: 3.5-5
            'NEU': 2.5 + x['score'],        # Neutral: 2.5-3.5
            'NEG': 2.5 - 1.5 * x['score']   # Negative: 1-2.5
        }[x['label']]
    }
]

# Initialize models with error handling
pipelines = []
for model in MODELS:
    try:
        pipe = pipeline(
            "text-classification",
            model=model["name"],
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
        pipelines.append({
            "pipe": pipe,
            "normalize": model["normalize"],
            "weight": model["weight"],
            "name": model["name"]
        })
        print(f"✅ Loaded model: {model['name']}")
    except Exception as e:
        print(f"❌ Failed to load {model['name']}: {str(e)}")

if not pipelines:
    raise RuntimeError("No models were successfully loaded")

def adjust_for_keywords(text, rating):
    """Adjust rating based on strong sentiment keywords"""
    text_lower = text.lower()
    
    # Positive boosters
    positive_boosters = ["excellent", "perfect", "best", "amazing", "wonderful"]
    if any(word in text_lower for word in positive_boosters):
        rating = min(5.0, rating * 1.15)  # 15% boost
    
    # Negative indicators
    negative_indicators = ["terrible", "awful", "worst", "horrible", "never again"]
    if any(word in text_lower for word in negative_indicators):
        rating = max(1.0, rating * 0.85)  # 15% penalty
    
    return rating

def ensemble_predict(text):
    """Get robust sentiment rating using weighted median approach"""
    predictions = []
    
    for model in pipelines:
        try:
            result = model["pipe"](text)[0]
            raw_rating = model["normalize"](result)
            
            # Apply model-specific adjustments
            if "bertweet" in model["name"]:
                raw_rating *= 1.05  # Slight boost for this model
            
            # Add weighted predictions
            predictions.extend([raw_rating] * int(model["weight"] * 10))
            
            print(f"Model {model['name']} → {raw_rating:.1f}")
        except Exception as e:
            print(f"⚠️ Error in {model['name']}: {str(e)}")
    
    if not predictions:
        return 3.0  # Fallback neutral rating
    
    final_rating = median(predictions)
    final_rating = adjust_for_keywords(text, final_rating)
    
    # Ensure rating is within bounds
    final_rating = np.clip(final_rating, 1.0, 5.0)
    return round(final_rating, 1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        comment = data.get('comment', '')[:1000].strip()
        
        if not comment:
            return jsonify({'error': 'Empty comment'}), 400
            
        rating = ensemble_predict(comment)
        
        print(f"Final rating for '{comment[:30]}...': {rating}")
        
        return jsonify({
            'rating': rating,
            'models_used': len(pipelines),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"🚨 Analysis error: {str(e)}")
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)