import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load LLM (online API OPTIONAL, but here local)
tokenizer = AutoTokenizer.from_pretrained("mistral-7b-instruct")
model = AutoModelForCausalLM.from_pretrained("mistral-7b-instruct")

# Load vector DB
chroma = chromadb.PersistentClient(path="astrix_db")
collection = chroma.get_or_create_collection(name="manual")

# Load manual file into DB (run once)
with open("space_manual.md", "r") as f:
    text = f.read().split("\n\n")
    for chunk in text:
        embedding = embed_model.encode(chunk).tolist()
        collection.add(documents=[chunk], embeddings=[embedding], ids=[chunk[:30]])

# Streamlit UI
st.title("ASTRIX — Space Emergency AI Agent")

user_input = st.text_input("Astronaut, what’s your situation?")

if user_input:
    query_vec = embed_model.encode(user_input).tolist()
    result = collection.query(query_embeddings=[query_vec], n_results=2)

    context = "\n".join(result["documents"][0])
    prompt = f"Context:\n{context}\n\nUser: {user_input}\nAI Response:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write(answer)
