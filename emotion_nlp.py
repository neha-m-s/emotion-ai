from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")

# Emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise',
    'neutral'
]

def analyze_text_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return emotion_labels[predicted_class]

def generate_response(text, emotion):
    responses = {
        "sadness": "I'm here for you. It's okay to feel sad. Want to talk about it?",
        "joy": "That's amazing! I'm happy for you 😊",
        "anger": "Take a deep breath. Want to vent it out?",
        "fear": "It’s normal to be scared sometimes. You’re not alone.",
        "love": "Love is beautiful. Cherish it 💖",
        "gratitude": "You're welcome! Always here for you.",
        "neutral": "Thanks for sharing. I'm here to listen.",
        "disappointment": "That must have felt tough. Want to share more?",
        "nervousness": "You're doing great. Try to relax and take it slow."
    }

    # Default fallback
    return responses.get(emotion, "I hear you. Let's take it one step at a time.")
