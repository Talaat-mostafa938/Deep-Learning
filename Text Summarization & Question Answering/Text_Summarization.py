import streamlit as st
from transformers import pipeline
import pdfplumber


@st.cache_resource
def load_models():
    """
    Loads the models and tokenizer once and caches them.
    This prevents the models from being reloaded on every user interaction.
    """
    try:
        # Using t5-small for summarization, which is a Seq2SeqLM
        summarizer_model_name = "t5-small"
        qa_model_name = "distilbert-base-cased-distilled-squad"

        # Initialize the summarization pipeline
        summarizer = pipeline("summarization", model=summarizer_model_name)

        # Initialize the question-answering pipeline
        qa_pipeline = pipeline("question-answering", model=qa_model_name)

        return summarizer, qa_pipeline
    except Exception as e:
        st.error(
            "Failed to load models. Please ensure you have an internet connection or the models are available locally. Error: {}".format(
                e
            )
        )
        return None, None

def extract_text(file):
    if file is None:
        return None
    
    file_type = file.name.split(".")[-1].lower()
    try:
        if file_type == "txt":
            return file.read().decode("utf-8")

        elif file_type == "pdf":
            with pdfplumber.open(file) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        return f"⚠️ Error reading file: {e}"

# Load the models with caching
summarizer, qa_pipeline = load_models()


st.set_page_config(
    page_title="Text Summarization and Question Answering",
    page_icon="📝",
    layout="wide",
)
st.title("🧠Summarization & Question Answering")

# Keep UI mode via session_state so inner buttons work after rerun
if "mode" not in st.session_state:
    st.session_state["mode"] = None


col1, col2 = st.columns(2)

with col1:
    st.subheader("Text Summarization")
    
    summarization_level = st.selectbox("Select Summarization Mode:", ["Text", "File"])
    
    if summarization_level == "File":
        uploaded_file = st.file_uploader("Upload a text file for summarization:", type=["txt" , "pdf"])
        
        if uploaded_file is not None:
            input_text = extract_text(uploaded_file)
            if input_text:
                st.text_area("📄 Extracted Text:", input_text, height=300)
            else:
                st.error("❌ Could not extract text from the uploaded file.")
            
    else:
        input_text = st.text_area(
            "Enter article for summarization:",
            height=300,
            placeholder="Paste your article here...",
        )


with col2:
    if st.button("Generate Summary" , use_container_width=True):
        if not input_text or not input_text.strip():
            st.warning("Please enter some text to summarize.")
        elif summarizer is None:
            st.error("Summarizer model is not available.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    result = summarizer(
                        input_text, max_length=200, min_length=30, do_sample=False
                    )
                    st.success("✅ Summary:")
                    st.write(result[0]["summary_text"])
                except Exception as e:
                    st.error(f"Failed to summarize: {e}")


st.markdown("---")
st.subheader("Question Answering ")


question = st.text_input("Ask a Question:")

if st.button("Get Answer" , use_container_width=True):
    
    if not input_text.strip() or not question.strip():
        st.warning("Please provide both context and question.")
    elif qa_pipeline is None:
        st.error("QA model is not available.")
    else:
        with st.spinner("Searching for answer..."):
            try:
                result = qa_pipeline(question=question, context=input_text)
                answer = result.get("answer", "").strip()
                score = float(result.get("score", 0.0))

                if not answer:
                    st.info("🤔 The model could not find a meaningful answer.")
                else:
                    st.success(f"✅ Answer (confidence {score:.2f}):")
                    st.write(answer)
            except Exception as e:
                st.error(f"Failed to get answer: {e}")
