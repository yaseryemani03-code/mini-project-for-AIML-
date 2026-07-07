from transformers import pipeline, Conversation
import json
import random

# Load models
generator = pipeline('text-generation', model='gpt2')
chatbot = pipeline('text-generation', model='microsoft/DialoGPT-medium')  # Conversational AI

# Load dream symbols
with open('dream_symbols.json', 'r') as f:
    SYMBOLS = json.load(f)

# Store conversation history per session
conversation_history = []

def interpret_dream(dream_text):
    """
    Generate a psychological/spiritual interpretation of a dream.
    """
    prompt = f"""
    You are a wise dream interpreter. Analyze this dream symbolically and psychologically:
    Dream: "{dream_text}"

    Provide a detailed, insightful interpretation:
    """
    
    try:
        result = generator(
            prompt,
            max_length=300,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=50256
        )
        interpretation = result[0]['generated_text'].replace(prompt, "").strip()
        
        if len(interpretation) < 20:
            return fallback_interpretation(dream_text)
        
        return interpretation
    
    except:
        return fallback_interpretation(dream_text)

def fallback_interpretation(dream_text):
    """
    Fallback: Extract keywords and match with symbol dictionary.
    """
    words = dream_text.lower().split()
    found_symbols = []
    
    for word in words:
        if word in SYMBOLS:
            found_symbols.append(f"'{word}' = {SYMBOLS[word]}")
    
    if found_symbols:
        return f"Your dream may symbolize: {', '.join(found_symbols[:3])}. " \
               f"Consider how these elements relate to your waking life."
    else:
        return "This dream seems unique to you! It might reflect your personal experiences and emotions. " \
               "Try journaling about it to uncover deeper meaning."

def chat_with_dream_ai(user_message, dream_context):
    """
    AI assistant that answers questions about the dream.
    """
    global conversation_history
    
    # Build context-aware prompt
    prompt = f"""
    You are a dream interpreter AI assistant. The user had this dream: "{dream_context}"
    
    User's question: {user_message}
    
    Your response (be insightful, empathetic, and helpful):
    """
    
    try:
        result = generator(
            prompt,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=50256
        )
        response = result[0]['generated_text'].replace(prompt, "").strip()
        
        # If response is too short, use fallback
        if len(response) < 10:
            return "That's a great question! Dreams are deeply personal. Could you tell me more about how that part made you feel?"
        
        return response
    
    except:
        return "I'm here to help! Dreams are complex—could you rephrase your question or tell me more about that specific detail?"

def get_dream_advice(interpretation):
    """
    Generate actionable advice based on the interpretation.
    """
    advice_prompt = f"Based on this dream interpretation: '{interpretation}', " \
                    f"give one practical advice for the dreamer's daily life."
    
    try:
        result = generator(advice_prompt, max_length=100, temperature=0.7, pad_token_id=50256)
        return result[0]['generated_text'].replace(advice_prompt, "").strip()
    except:
        return "Take some time for self-reflection today. Your subconscious is trying to tell you something important."