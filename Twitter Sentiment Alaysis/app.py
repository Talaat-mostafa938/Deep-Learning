import streamlit as st
from transformers import pipeline , AutoTokenizer , AutoModelForSequenceClassification

st.set_page_config(
    page_title = 'Sentiment Analysis System',
    page_icon = '🤖',
    layout = 'centered',
)

@st.cache_resource
def load_model():
    model_path = "twitter_sentiment_model"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        sentiment_analysis = pipeline("sentiment-analysis" , model=model , tokenizer = tokenizer)
        return sentiment_analysis
    except Exception as e:
        print(f'Error loading the model : {e}')

sentiment_analysis = load_model()

st.title('Twitter Sentiment Analysis 🤖')
st.markdown('Enter Text to Analysis Sentiment as [**Positive** , **Netural** , **Negative**]')
with st.form("Sentimet_Analysis"):
    user_input = st.text_area("Enter Text for Analysis" , "" , height = 130 , placeholder = 'Twitter Sentiment Analysis')
    submit_button = st.form_submit_button('Analysis Sentiment')
    
if submit_button and len(user_input.strip()) > 0 :
    st.spinner('Analysing...')
    results = sentiment_analysis(user_input)[0]

    
    st.subheader('Analysis Result📊')
    sentiment_map = {
            'Positive': ('Positive 😊', '🟢'),
            'Negative': ('Negative 😞', '🔴'),
            'Neutral': ('Neutral 😐', '🟡')
    }
    result = results['label']
    score = round(results['score'] , 1)
    
    sentiment_text , emoji = sentiment_map.get(result)
     
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
                label="Result",
                value=sentiment_text
                )
    with col2:
            st.metric(
                label="Confidence",
                value=f"{score:.2f}%"
                )
    st.progress(score)
    
else:
    st.warning('Please Enter Text for Analysis')
    
