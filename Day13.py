#CHUNKING + EMBEDDINGS

import os
from google import genai
import textwrap

def chunk(text,chunk_size=100,overlap=20):
    if chunk_size <=0:
        raise ValueError('Chunk size must be positive!')

    if overlap < 0 or overlap >=chunk_size:
        raise ValueError('Overlap must be within 0 to less than chunk size!')

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. " \
"Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at " \
"Letraset and James Mosley, the librarian at St Bride Printing Library in London, took a 1914 " \
"Cicero translation and scrambled it to make dummy text for Letraset's Body Type sheets. It has " \
"survived not only many decades, but also the leap into electronic typesetting, remaining " \
"essentially unchanged. It was popularised thanks to these sheets and more recently with desktop " \
"publishing software like Aldus PageMaker and Microsoft Word including versions of Lorem Ipsum." \
"It is a long established fact that a reader will be distracted by the readable content of a " \
"page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less " \
"normal distribution of letters, as opposed to using 'Content here, content here', making it " \
"look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum " \
"as their default model text, and a search for 'lorem ipsum' will uncover many web sites " \
"still in their infancy. Various versions have evolved over the years, sometimes by accident, " \
"sometimes on purpose (injected humour and the like)."    

chunked = chunk(text,chunk_size=200,overlap=20)

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

def embedding(text):
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents = text
    )
    return response.embeddings[0].values

embed_chunks = [
    embedding(chun) for chun in chunked
]

for i, chunk in enumerate(chunked,start=1):
    wraped = textwrap.fill(chunk,width=50)
    print(f"\nChunk {i}. : {wraped}")
    print(f"Length of embedding: {len(embed_chunks[i-1])}")




