import openai

openai.api_key = "YOUR_API_KEY"  # Replace with your key

def analyze_bias(text):
    prompt = f"""
    You are a medical language bias checker. 
    Identify any biased or judgmental phrases in the following text and suggest neutral alternatives.
    Return results as JSON with "biased_phrases", "suggestions", and "score" (0-100 fairness).
    
    Text: {text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or 'gpt-4o' if available
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.choices[0].message.content
    return result
