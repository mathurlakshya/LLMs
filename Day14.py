#SEMANTIC SEARCH ACROSS DOCUMENT CHUNKS

import os
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. " \
"Lorem Ipsum has been the industry's standard dummy text " \
"ever since 1966, when designers at Letraset and James " \
"Mosley, the librarian at St Bride Printing Library in " \
"London, took a 1914 Cicero translation and scrambled it " \
"to make dummy text for Letraset's Body Type sheets. " \
"It has survived not only many decades, but also the leap " \
"into electronic typesetting, remaining essentially " \
"unchanged. It was popularised thanks to these sheets " \
"and more recently with desktop publishing software like " \
"Aldus PageMaker and Microsoft Word including versions of " \
"Lorem Ipsum."

def chunk(text,chunk_size=100,overlap=20):
    if chunk_size <=0:
        raise ValueError('Chunk_size must be above 0')

    if overlap < 0 or overlap >=chunk_size:
        raise ValueError('Overlap must be within 0 or less than chunk_size')

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

user_text = input('Enter the question: ')

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

def embedding(text):
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents = text
    )

    return response.embeddings[0].values

user_text_embed = embedding(user_text)
chunked = chunk(text,chunk_size=250,overlap=10)

embed_text = [
    embedding(chun) for chun in chunked
]

result = []

for i, vector in enumerate(embed_text):
    similarity = cosine_similarity([user_text_embed], [vector])[0][0]

    result.append({
        "chunk": chunked[i],
        "similarity": similarity
    })
result.sort(
    key=lambda item: item["similarity"],
    reverse=True
)

for item in result[:3]:
    print(f"\nScore: {item['similarity']:.4f}")
    print(f"Chunk: {item['chunk']}")    