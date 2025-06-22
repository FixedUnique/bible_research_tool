import streamlit as st
import requests
from transformers import pipeline
import time

# --- App Config ---
st.set_page_config(page_title="Free Bible AI", layout="wide")

# --- Load AI Model (Cached) ---
@st.cache_resource(show_spinner=False)
def load_ai_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

qa_model = load_ai_model()

# --- Bible API Helper ---
def get_bible_verse(reference):
    try:
        url = f"https://bible-api.com/{reference}?translation=kjv"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()["text"]
        return None
    except Exception as e:
        st.error(f"Error fetching verse: {str(e)}")
        return None

def find_relevant_verses(question):
    prompt = f"""Return 3 most relevant Bible verses for: {question} 
    Format strictly as 'Book Chapter:Verse' separated by commas. 
    Example: 'John 3:16, Romans 6:23, Revelation 21:4'"""
    
    try:
        result = qa_model(prompt, max_length=100)[0]['generated_text']
        return [v.strip() for v in result.split(",") if ":" in v]  # Simple validation
    except Exception as e:
        st.error(f"Error finding verses: {str(e)}")
        return ["John 3:16", "Romans 6:23"]  # Fallback verses

# --- AI Summarizer ---
def generate_answer(question, verses):
    prompt = f"""
    Question: {question}
    Bible Verses: {verses}
    
    Summarize what the Bible says about this topic in 1-2 paragraphs.
    Include exact verse references (e.g. John 3:16) for each point.
    Write in clear, pastoral language.
    """
    
    try:
        start_time = time.time()
        result = qa_model(prompt, max_length=512)[0]['generated_text']
        st.write(f"Generated in {time.time() - start_time:.1f}s")
        return result
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# --- Ad Integration (Placeholder) ---
def inject_ads():
    st.components.v1.html("""
    <div style="border:1px solid #ccc; padding:20px; text-align:center;">
    [Ad Space - Replace with Your Ad Code]
    </div>
    """, height=150)

# --- UI ---
st.title("✝️ Free Bible Study AI Tool ✝️")
st.markdown("Ask any question about the Bible and get answers with references")

# Sidebar with Ad
with st.sidebar:
    st.write("Support this Developer:")
    inject_ads()

# Main Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What does the Bible say about..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching the Bible..."):
            # Step 1: Find and fetch relevant verses
            verse_refs = find_relevant_verses(prompt)
            valid_verses = []
            
            for ref in verse_refs:
                verse_text = get_bible_verse(ref)
                if verse_text:
                    valid_verses.append(f"{ref}: {verse_text}")
            
            if not valid_verses:
                st.error("Could not fetch Bible verses. Please try again.")
                st.stop()
            
            verses_text = "\n\n".join(valid_verses)
            
            # Step 2: Generate summary
            response = generate_answer(prompt, verses_text)
            
            # Display results
            st.markdown(response)
            st.markdown("**References:** " + ", ".join(verse_refs))
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer Ad
st.divider()
inject_ads()