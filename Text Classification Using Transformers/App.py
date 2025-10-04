import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Page Configuration
st.set_page_config(
    page_title='Multi Text Classification',
    page_icon='🤖',
    layout='centered',
)

# Load Model with Caching
@st.cache_resource
def load_model():
    """Load the text classification model and tokenizer"""
    try:
        model_path = "multi-text-classification"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=5
        )
        
        text_classification = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            return_all_scores=True
        )
        
        return text_classification
    except Exception as e:
        st.error(f'Error loading the model: {e}')
        return None

# Load the pipeline
pipe = load_model()

# App Title and Description
st.title('🤖 Text Classification')
st.markdown(
    'Enter text to classify into one of the following categories: '
    '**Business**, **Education**, **Entertainment**, **Sports**, or **Technology**'
)

# Check if model loaded successfully
if pipe is None:
    st.error("Failed to load the model. Please check the model path and try again.")
    st.stop()

# Classification Form
with st.form("text_classification_form"):
    user_input = st.text_area(
        "Enter Text for Classification",
        "",
        height=130,
        placeholder='Enter your text here... (e.g., "The football match was amazing!")'
    )
    
    submit_button = st.form_submit_button('🔍 Classify Text')

    if submit_button:
        if user_input.strip():
            with st.spinner('Classifying...'):
                try:
                    # Get classification results
                    result = pipe(user_input)[0]
                    
                    # Get the highest score classification
                    top_classification = max(result, key=lambda x: x['score'])
                    
                    # Display results
                    st.success('Classification Complete!')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Category", 
                            value=top_classification['label'].upper()
                        )
                    with col2:
                        st.metric(
                            label="Confidence", 
                            value=f"{top_classification['score']:.1%}"
                        )
                    
                    # Show all scores in an expander
                    with st.expander("📊 View All Category Scores"):
                        # Sort by score descending
                        sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)
                        
                        for item in sorted_results:
                            score_percentage = item['score'] * 100
                            st.progress(item['score'])
                            st.write(f"**{item['label'].capitalize()}**: {score_percentage:.2f}%")
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error during classification: {e}")
        else:
            st.warning('⚠️ Please enter some text for classification.')

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit & Hugging Face Transformers 🚀"
    "</div>", 
    unsafe_allow_html=True
)