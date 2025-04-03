import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure API with environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Set GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Extract text and images from PDF with metadata
def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = []
    images_per_page = {}
    image_metadata = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_per_page.append((page_num, text))
        
        images = page.get_images(full=True)
        images_per_page[page_num] = []

        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width, height = base_image["width"], base_image["height"]
            image_name = base_image.get("name", "").lower()

            # Skip small/unimportant images
            if width < 100 or height < 50 or (width * height) < 12000:
                continue

            # Save image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(image_bytes)
                img_path = tmp_img.name
                images_per_page[page_num].append(img_path)
                
                # Store metadata
                image_metadata[img_path] = {
                    'page': page_num,
                    'text': text,
                    'width': width,
                    'height': height,
                    'area': width * height
                }
    
    doc.close()
    return text_per_page, images_per_page, image_metadata

# Index PDF text
def index_pdf_text(text_per_page):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for page_num, text in text_per_page:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={'page': page_num})
            documents.append(doc)
    return FAISS.from_documents(documents, embedding_function)

# Query Gemini AI
def query_gemini(prompt, context):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Context: {context}\nUser Query: {prompt}")
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

# Compute similarity
def compute_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    
    try:
        emb1 = embedding_model.encode([text1])[0]
        emb2 = embedding_model.encode([text2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return 0.0

# Rank images by relevance
def rank_images_by_relevance(query, candidate_images, image_metadata, context_text):
    if not candidate_images:
        return []

    image_scores = []
    for img_path in candidate_images:
        metadata = image_metadata.get(img_path, {})
        surrounding_text = metadata.get('text', '')

        text_similarity = compute_similarity(query, surrounding_text)

        # Area factor (importance of large images)
        area = metadata.get('area', 0)
        area_factor = min(0.2, area / 500000)

        final_score = text_similarity * 0.7 + area_factor

        if final_score > 0.4:
            image_scores.append((img_path, final_score))

    # Sort and return top 2 most relevant images
    image_scores.sort(key=lambda x: x[1], reverse=True)
    return [img for img, _ in image_scores[:2]]

# Search PDF and answer
def search_pdf_and_answer(query, vector_store, images_per_page, image_metadata):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = query_gemini(query, context)

    # Get relevant pages
    page_nums = {doc.metadata['page'] for doc in docs}
    candidate_images = [img for page in page_nums for img in images_per_page.get(page, [])]

    if not candidate_images:
        return answer, [], "No relevant images found for this query."

    relevant_images = rank_images_by_relevance(query, candidate_images, image_metadata, context)

    message = ""
    if not relevant_images:
        message = "No relevant images found for this query."

    return answer, relevant_images, message

# Streamlit UI
st.title("📄 PDF Chatbot with Gemini API 🤖")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None
    st.session_state.images_per_page = None
    st.session_state.image_metadata = None

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")

if uploaded_file and not st.session_state.pdf_processed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    with st.spinner("Processing PDF... Please wait..."):
        text_per_page, images_per_page, image_metadata = extract_text_and_images_from_pdf(temp_path)
        vector_store = index_pdf_text(text_per_page)

        st.session_state.vector_store = vector_store
        st.session_state.images_per_page = images_per_page
        st.session_state.image_metadata = image_metadata
        st.session_state.pdf_processed = True

    st.success("PDF successfully indexed! ✅")

query = st.text_input("Ask a question from the PDF:")

if query and st.session_state.pdf_processed:
    with st.spinner("Generating response..."):
        answer, relevant_images, message = search_pdf_and_answer(
            query, st.session_state.vector_store, st.session_state.images_per_page, st.session_state.image_metadata
        )

    st.write("### 🤖 Answer:")
    st.write(answer)

    if relevant_images:
        st.write("#### Relevant Images from PDF:")
        for img_path in relevant_images:
            st.image(img_path, use_column_width=True)
    elif message:
        st.info(message)
