from Image_Generate import pipe
import streamlit as st

st.set_page_config(page_title = 'Text to Image Generator', page_icon = '🤖' ,layout="centered")

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1f77b4;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child:hover {
        background-color: #135e96;
        color: #e0e0e0;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 عنوان التطبيق
st.title("🖼️ AI Image Generator")
st.markdown("#### Turn your imagination into visuals")

# ✍️ إدخال البرومبت
prompt = st.text_input(
    label="📝 Enter Your Prompt",
    placeholder="e.g. A realistic mountain landscape with a clear lake at sunset"
)

# 📐 إعدادات الأبعاد
st.markdown("#### 🧭 Image Dimensions")
col1, col2 = st.columns(2)

with col1:
    height = st.slider("Height (px)", min_value=256, max_value=1024, value=512, step=32)

with col2:
    width = st.slider("Width (px)", min_value=256, max_value=1024, value=512, step=32)

st.markdown("---")

# 🎨 زر توليد الصورة
if st.button("🎨 Generate Image"):
    with st.spinner("Generating Image... ⏳"):
        if prompt:
            result = pipe(prompt, height=height, width=width, guidance_scale=7.5, negative_prompt="")
            image = result.images[0]
            st.image(image, caption="🖼️ Generated Image", use_container_width=True)
        else:
            st.warning("⚠️ Please enter a prompt before generating an image.")
