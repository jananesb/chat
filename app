import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Set GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Extract text and images efficiently
def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = []
    images_per_page = {}
    image_metadata = {}
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text_per_page.append((page_num, text))
        
        images = page.get_images(full=True)
        images_per_page[page_num] = []

        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width, height = base_image["width"], base_image["height"]
            
            if width * height < 15000:  # Filter small images
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(image_bytes)
                img_path = tmp_img.name
                images_per_page[page_num].append(img_path)
                
                image_metadata[img_path] = {
                    'page': page_num,
                    'width': width,
                    'height': height,
                    'area': width * height,
                }
    
    doc.close()
    return text_per_page, images_per_page, image_metadata

# Index extracted text
def index_pdf_text(text_per_page):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for page_num, text in text_per_page:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={'page': page_num}))
    return FAISS.from_documents(documents, embedding_function)

# Query Gemini API
def query_gemini(prompt, context):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"Context: {context}\nUser Query: {prompt}\nProvide a short and concise answer."
        )
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

# Compute text similarity
def compute_similarity(text1, text2):
    try:
        emb1, emb2 = embedding_model.encode([text1, text2])
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    except:
        return 0.0

# Rank images for relevance
def rank_images_by_relevance(query, candidate_images, image_metadata):
    image_scores = []
    for img_path in candidate_images:
        metadata = image_metadata.get(img_path, {})
        area = metadata.get('area', 0)
        score = area / 500000  # Normalize by large area
        image_scores.append((img_path, score))
    return [img for img, _ in sorted(image_scores, key=lambda x: x[1], reverse=True)[:2]]

# Search PDF and get answer
def search_pdf_and_answer(query, vector_store, images_per_page, image_metadata):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = query_gemini(query, context)
    page_nums = {doc.metadata['page'] for doc in docs}
    candidate_images = [img for page in page_nums for img in images_per_page.get(page, [])]
    relevant_images = rank_images_by_relevance(query, candidate_images, image_metadata)
    return answer, relevant_images

# Streamlit UI
st.title("📄 Efficient PDF Chatbot 🤖")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None
    st.session_state.images_per_page = None
    st.session_state.image_metadata = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and not st.session_state.pdf_processed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    
    with st.spinner("Processing PDF..."):
        text_per_page, images_per_page, image_metadata = extract_text_and_images_from_pdf(temp_path)
        vector_store = index_pdf_text(text_per_page)
        
        st.session_state.vector_store = vector_store
        st.session_state.images_per_page = images_per_page
        st.session_state.image_metadata = image_metadata
        st.session_state.pdf_processed = True
    st.success("PDF successfully processed! ✅")

query = st.text_input("Ask a question:")

if query and st.session_state.pdf_processed:
    with st.spinner("Generating response..."):
        answer, relevant_images = search_pdf_and_answer(
            query, st.session_state.vector_store, st.session_state.images_per_page, st.session_state.image_metadata
        )
    
    st.write("### 🤖 Answer:")
    st.write(answer)
    
    if relevant_images:
        st.write("#### Relevant Images:")
        for img_path in relevant_images:
            st.image(img_path, use_column_width=True)
