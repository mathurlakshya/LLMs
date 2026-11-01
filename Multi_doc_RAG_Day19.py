import os
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS_FOLDER = r"C:\Users\HP\Desktop\Python repos\Practice\documents"


def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            documents.append({
                "source": filename,
                "text": content
            })

    return documents


def chunk(text, chunk_size=1000, overlap=200):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be between 0 and chunk size")

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text
        })

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


documents = load_documents(DOCUMENTS_FOLDER)

all_chunks = []

for document in documents:

    chunks = chunk(document["text"], 400, 10)

    for item in chunks:
        all_chunks.append({
            "source": document["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"]
        })


for item in all_chunks:
    print(f"\nSource: {item['source']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Chunk Text: {item['text']}")


api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def embedding(text):
    response = client.models.embed_content(
        model = 'gemini-embedding-2',
        contents=text
        )
    return response.embeddings[0].values


for item in all_chunks:
    item["embedding"] = embedding(item["text"])


print("\nTotal chunks:", len(all_chunks))

for item in all_chunks:
    print(f"\nSource: {item['source']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Embedding dimensions: {len(item['embedding'])}")
   

user_text = input("\nEnter your question: ")

user_embedding = embedding(user_text)

results = []

for item in all_chunks:
    similarity = cosine_similarity(
        [user_embedding],
        [item['embedding']]
        )[0][0]

    results.append({
        "source": item['source'],
        "chunk_id": item['chunk_id'],
        "text": item['text'],
        "similarity" : similarity
        })

results.sort(
    key=lambda x: x['similarity'],reverse=True
    )

top_3 = results[:3]

context = ""

for item in top_3:
    context += f"""
Source: {item['source']}
Content: {item['text']}
"""

prompt = f"""
Answer the question using only the context provided below.

If the answer is not available in the context, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{user_text}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\nGenerated Answer:")
print(response.text)