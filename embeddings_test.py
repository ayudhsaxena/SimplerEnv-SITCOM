import os
import numpy as np
from scipy.spatial.distance import cosine
from google import genai
from google.genai import types

def get_text_similarity(text1, text2, api_key=None):
    """
    Calculate the semantic similarity between two texts using Google's Generative AI 
    embedding model.
    
    Args:
        text1 (str): First text input
        text2 (str): Second text input
        api_key (str, optional): Your Google AI API key. If None, will look for
                                GOOGLE_API_KEY environment variable
    
    Returns:
        float: Similarity score between 0 and 1 (higher means more similar)
    """
    # Configure the API key
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("API key must be provided either as an argument or as GOOGLE_API_KEY environment variable")
    
    client = genai.Client(api_key=api_key)
    
    # Get the embedding model
    
    # Generate embeddings for both texts
    response1 = client.models.embed_content(
        model='text-embedding-004',
        contents=text1,
    )
    response2 = client.models.embed_content(
        model='text-embedding-004',
        contents=text2,
    )
        
    # Extract the embedding values
    vector1 = np.array(response1.embeddings[0].values)
    vector2 = np.array(response2.embeddings[0].values)
    
    # Calculate cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(vector1, vector2)
    
    return similarity


import dotenv
dotenv.load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") 

text1 = "Move"
text2 = "Rotate"

try:
    similarity_score = get_text_similarity(text1, text2, API_KEY)
    print(f"Similarity score: {similarity_score:.4f}")
    
    # Additional examples
    text3 = "Place the green block on the yellow block."
    text4 = "The fourth subtask is to place the green block on top of the yellow block."
    
    similarity_score2 = get_text_similarity(text3, text4, API_KEY)
    print(f"Similarity score: {similarity_score2:.4f}")

except Exception as e:
    print(f"Error: {e}")