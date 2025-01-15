import streamlit as st
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import numpy as np
import PyPDF2
import re

# Load the model structure
def build_model():
    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    line_number_normalized = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="line_number_normalized")
    total_lines_normalized = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="total_lines_normalized")
    relative_position = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="relative_position")

    # BERT model
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    bert_output = bert_model(input_ids, attention_mask=attention_mask).pooler_output

    # Combine BERT with other features
    combined_features = tf.keras.layers.Concatenate()(
        [bert_output, line_number_normalized, total_lines_normalized, relative_position]
    )

    # Dense layers for classification
    dense_output = tf.keras.layers.Dense(128, activation="relu")(combined_features)
    final_output = tf.keras.layers.Dense(5, activation="softmax")(dense_output)

    model = tf.keras.Model(
        inputs=[input_ids, attention_mask, line_number_normalized, total_lines_normalized, relative_position],
        outputs=final_output,
    )

    return model

# Build the model structure
model = build_model()

# Load weights
try:
    model.load_weights("skimlit_model.h5")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Tokenizer for text processing
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Corrected label mapping
label_mapping = {
    0: "BACKGROUND",
    1: "CONCLUSIONS",
    2: "METHODS",
    3: "OBJECTIVE",
    4: "RESULTS"
}

# Function to classify text
def classify_text(text):
    # Tokenize the input text
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="tf",  # Return TensorFlow tensors
    )

    # Prepare additional features (dummy values for now)
    line_number_normalized = tf.convert_to_tensor([[0]], dtype=tf.float32)
    total_lines_normalized = tf.convert_to_tensor([[1]], dtype=tf.float32)
    relative_position = tf.convert_to_tensor([[0]], dtype=tf.float32)

    # Predict
    predictions = model.predict({
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "line_number_normalized": line_number_normalized,
        "total_lines_normalized": total_lines_normalized,
        "relative_position": relative_position,
    })

    predicted_label = np.argmax(predictions, axis=1)[0]
    return label_mapping[predicted_label]

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading the PDF file: {e}")
        return None

# Improved function to split text into sentences
def split_into_sentences(text):
    # Split sentences using regex to handle multiple separators (e.g., ".", "?", "!")
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Streamlit App
st.set_page_config(
    page_title="SkimLit Classifier",
    page_icon="📄",
    layout="wide"
)

# Add custom styling
st.markdown(
    """
    <style>
        body {
            background-color: #f0f8ff; /* Alice Blue Background */
            color: #333333;
        }
        .stApp {
            background-color: #f0f4f8; /* Light soft grayish blue */
        }
        .title {
            color: #2b547e;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
        }
        .description {
            color: #495057;
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        .stButton button {
            background-color: #007bff;
            color: white;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 1.1em;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📄 SkimLit Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Classify text or research abstracts into categories like <b>OBJECTIVE</b>, <b>METHODS</b>, <b>RESULTS</b>, <b>CONCLUSIONS</b>, and <b>BACKGROUND</b>. </div>', unsafe_allow_html=True)

# Create a two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload a PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

with col2:
    st.subheader("✍️ Enter Text")
    user_input = st.text_area("Type your text here", placeholder="Enter text for classification...")

# Classify button
if st.button("🔍 Classify"):
    if uploaded_file:
        # Process PDF
        with st.spinner("🔄 Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        if extracted_text:
            st.write("### Extracted Text (First 500 characters):")
            st.write(f"{extracted_text[:500]}...")

            # Split text into sentences and classify each
            st.write("### Sentence Classification:")
            sentences = split_into_sentences(extracted_text)
            for sentence in sentences:
                if sentence.strip():  # Avoid empty sentences
                    classification = classify_text(sentence)
                    st.write(f"**Sentence**: {sentence}")
                    st.write(f"**Predicted Class**: `{classification}`")
    elif user_input:
        # Process user input
        with st.spinner("🔄 Classifying input text..."):
            classification = classify_text(user_input)
        st.success(f"**Predicted Class**: `{classification}`")
    else:
        st.warning("Please upload a PDF or enter some text!")
