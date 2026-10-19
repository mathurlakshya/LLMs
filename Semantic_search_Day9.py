import os
from google import genai
import math

api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError('API key is missing!!')

client = genai.Client(api_key=api_key)

def cosine_similarity(vector_a,vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError('Both vector must have equal values!')
    dot_product = sum(
        a * b
        for a,b in zip(vector_a,vector_b)
    )

    magnitude_a = math.sqrt(
        sum(value**2 for value in vector_a)
    )
    magnitude_b = math.sqrt(
        sum(value ** 2 for value in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    
    cosine = dot_product/(magnitude_a*magnitude_b)

    return cosine

tickets = [
    "I can't log in to my account",
    "My payment was reduced but my order was cancelled",
    "There's been a wrong product's delivery at my house",
    "I have forgotten my ID's password",
    "I have been facing error issues in this app for a very long time",
    "My transacton was cancelled",
    "My account is been hacked by someone",
    "I didn't got my shirt delivered till yet"
]

def embeddings(text):
    response = client.models.embed_content(
        model = 'gemini-embedding-2',
        contents = text
    )
    return response.embeddings[0].values

embed_texts = [
    embeddings(text) for text in tickets
]

query = "I can't access my account"

embed_query = embeddings(query)

result = []

for ticket,embed in zip(tickets,embed_texts):
    cosine = cosine_similarity(embed_query,embed)

    result.append({
        'ticket': ticket,
        'similarity': cosine
    })

result.sort(
    key=lambda item: item['similarity'],
    reverse=True
)

print(f"Search query: {query}\n")
print("Results ranked by similarity:\n")

for result in result:
    print(f"Score: {result['similarity']:.4f}")
    print(f"Ticket: {result['ticket']}")
    print("-" * 50)










