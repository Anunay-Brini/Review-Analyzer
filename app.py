import streamlit as st
import pickle
import re
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- 0. Theme Management Functions ---

# Initialize session state for theme if not already set
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    """Switches the theme state when the button is pressed."""
    st.session_session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
    # Streamlit reruns automatically when session state changes

def get_theme_css(theme):
    """Returns the custom CSS string for the specified theme."""
    if theme == 'dark':
        return """
        <style>
            /* Global Background and Typography - DARK THEME */
            .stApp {
                background-color: #121212; /* Pure Black/Dark Gray */
                color: #f0f0f0; /* Light text */
                font-family: 'Inter', sans-serif;
                padding-top: 50px;
                transition: background-color 0.5s ease, color 0.5s ease; /* ADDED TRANSITION */
            }
            /* Main Title Styling */
            h1 {
                color: #00bcd4; /* Vibrant Cyan Accent */
                font-weight: 900;
                text-align: center;
                padding-bottom: 15px;
                border-bottom: 4px solid #005664;
                letter-spacing: 1.5px;
                text-shadow: 0 0 5px rgba(0, 188, 212, 0.5);
            }
            /* Centered tag line */
            .tagline {
                text-align: center;
                color: #b0b0b0;
                font-size: 1.1rem;
                margin-bottom: 2.5rem;
            }
            /* Secondary Header Styling */
            h2 {
                color: #e0e0e0;
                border-bottom: 1px solid #333333;
                padding-bottom: 0.5rem;
                margin-top: 1.5rem;
                font-size: 1.7rem;
                font-weight: 700;
                margin-left: 10px;
            }
            /* Text Area Label Styling */
            .stTextArea label {
                font-size: 1.25rem;
                font-weight: 700;
                color: #e0e0e0;
                text-align: left;
            }
            /* Text Area Input Background */
            .stTextArea textarea {
                background-color: #2c2c2c;
                color: #f0f0f0;
                border: 1px solid #444444;
                border-radius: 12px;
                box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
                padding: 18px;
                transition: border-color 0.3s, box-shadow 0.3s;
            }
            .stTextArea textarea:focus {
                border-color: #00bcd4;
                box-shadow: 0 0 0 3px rgba(0, 188, 212, 0.5);
                outline: none;
            }
            /* Primary Button Styling */
            div.stButton > button {
                background-color: #00bcd4;
                color: #121212;
                font-size: 1.1rem;
                font-weight: 800;
                padding: 0.8rem 2.5rem;
                border: none;
                border-radius: 30px;
                box-shadow: 0 6px 12px rgba(0, 188, 212, 0.5);
                transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s;
                margin-top: 1.5rem;
            }
            div.stButton > button:hover {
                background-color: #00a0ae;
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(0, 188, 212, 0.8);
            }
            /* Result Box Custom Styles (Success/Error) */
            .result-positive {
                background-color: #1e3c23;
                color: #a5d6a7;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #4caf50;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                font-size: 1.2rem;
                font-weight: 600;
            }
            .result-negative {
                background-color: #3d1c1c;
                color: #ef9a9a;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #e53935;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                font-size: 1.2rem;
                font-weight: 600;
            }
            /* Neutral Style for Dark Mode */
            .result-neutral {
                background-color: #3f351e; /* Gold/Yellow dark background */
                color: #ffc107; /* Bright yellow text */
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #ff9800; /* Orange border */
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                font-size: 1.2rem;
                font-weight: 600;
            }
            /* Metric Styling - make metrics stand out more */
            .st-emotion-cache-1wv02e7 {
                font-size: 3rem !important;
                font-weight: 900 !important;
                color: #ffffff !important; /* White for Dark Mode */
            }
            .st-emotion-cache-1wv02e7 small {
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                color: #bdbdbd !important;
            }
            /* Progress Bar Customization */
            .stProgress > div > div > div > div {
                background-color: #00bcd4;
                border-radius: 8px;
            }
            /* Sidebar styling for a cleaner look */
            .css-1d391kg {
                background-color: #1e1e1e !important;
                border-right: 1px solid #333333;
                transition: background-color 0.5s ease, border-color 0.5s ease; /* ADDED TRANSITION */
            }
            .css-1d391kg h2, .css-1d391kg label, .css-1d391kg div {
                color: #f0f0f0 !important;
            }
            .css-1d391kg h2 {
                color: #00bcd4 !important;
                border-bottom: 2px solid #00bcd4;
                padding-bottom: 8px;
            }
            /* General markdown text in dark mode */
            p {
                color: #f0f0f0 !important;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Global Background and Typography - LIGHT THEME */
            .stApp {
                background-color: #f0f0f0; /* Light Gray/Off-White */
                color: #121212; /* Dark text */
                font-family: 'Inter', sans-serif;
                padding-top: 50px;
                transition: background-color 0.5s ease, color 0.5s ease; /* ADDED TRANSITION */
            }
            /* Main Title Styling */
            h1 {
                color: #00a0ae; /* Darker Cyan Accent */
                font-weight: 900;
                text-align: center;
                padding-bottom: 15px;
                border-bottom: 4px solid #6c757d; /* Gray underline */
                letter-spacing: 1.5px;
                text-shadow: 0 0 1px rgba(0, 0, 0, 0.1);
            }
            /* Centered tag line */
            .tagline {
                text-align: center;
                color: #212529; /* UPDATED: Very dark gray for high contrast */
                font-size: 1.1rem;
                margin-bottom: 2.5rem;
            }
            /* Secondary Header Styling */
            h2 {
                color: #212529; /* UPDATED: Very dark gray for high contrast */
                border-bottom: 1px solid #ced4da; /* Light border */
                padding-bottom: 0.5rem;
                margin-top: 1.5rem;
                font-size: 1.7rem;
                font-weight: 700;
                margin-left: 10px;
            }
            /* Text Area Label Styling */
            .stTextArea label {
                font-size: 1.25rem;
                font-weight: 700;
                color: #212529; /* UPDATED: Very dark gray for high contrast */
                text-align: left;
            }
            /* Text Area Input Background */
            .stTextArea textarea {
                background-color: #ffffff; /* White background for input field */
                color: #121212; /* Dark text inside input */
                border: 1px solid #ced4da;
                border-radius: 12px;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
                padding: 18px;
                transition: border-color 0.3s, box-shadow 0.3s;
            }
            .stTextArea textarea:focus {
                border-color: #00a0ae; /* Dark Cyan highlight on focus */
                box-shadow: 0 0 0 3px rgba(0, 160, 174, 0.3);
                outline: none;
            }
            /* Primary Button Styling - High contrast button */
            div.stButton > button {
                background-color: #00a0ae; /* Dark Cyan */
                color: #ffffff; /* White text on button */
                font-size: 1.1rem;
                font-weight: 800;
                padding: 0.8rem 2.5rem;
                border: none;
                border-radius: 30px;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
                transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s;
                margin-top: 1.5rem;
            }
            div.stButton > button:hover {
                background-color: #007a82;
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            }
            /* Result Box Custom Styles (Success/Error) */
            .result-positive {
                background-color: #d4edda; /* Light Green background */
                color: #155724; /* Dark Green text */
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #28a745;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                font-size: 1.2rem;
                font-weight: 600;
            }
            .result-negative {
                background-color: #f8d7da; /* Light Red background */
                color: #721c24; /* Dark Red text */
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #dc3545;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                font-size: 1.2rem;
                font-weight: 600;
            }
            /* Neutral Style for Light Mode */
            .result-neutral {
                background-color: #fff3cd; /* Pale Yellow background */
                color: #856404; /* Dark Gold text */
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #ffc107; /* Yellow border */
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                font-size: 1.2rem;
                font-weight: 600;
            }
            /* Metric Styling - make metrics stand out more */
            .st-emotion-cache-1wv02e7 {
                font-size: 3rem !important;
                font-weight: 900 !important;
                color: #000000 !important; /* CHANGED TO BLACK for Light Mode */
            }
            .st-emotion-cache-1wv02e7 small {
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                color: #bdbdbd !important; /* UPDATED: Changed to match dark mode color for consistency */
            }
            /* Progress Bar Customization */
            .stProgress > div > div > div > div {
                background-color: #00a0ae;
                border-radius: 8px;
            }
            /* Sidebar styling for a cleaner look */
            .css-1d391kg {
                background-color: #ffffff !important; /* White sidebar */
                border-right: 1px solid #ced4da;
                transition: background-color 0.5s ease, border-color 0.5s ease; /* ADDED TRANSITION */
            }
            .css-1d391kg h2, .css-1d391kg label, .css-1d391kg div {
                color: #212529 !important; /* UPDATED: Very dark gray for high contrast */
            }
            .css-1d391kg h2 {
                color: #00a0ae !important;
                border-bottom: 2px solid #00a0ae;
                padding-bottom: 8px;
            }
            /* General markdown text in light mode */
            p {
                color: #000000 !important; /* UPDATED: Pure black for maximum contrast */
            }
        </style>
        """

# --- Apply Dynamic CSS ---
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)


# --- SET BROWSER TAB TITLE (New Addition) ---
# Must be the first streamlit command
st.set_page_config(page_title="Review Analyzer", layout="wide")

# --- Configuration (MUST match the training script) ---
MAX_SEQUENCE_LENGTH = 250
# Note: EMBEDDING_DIM and VOCAB_SIZE are not needed here but are mentioned in the sidebar for context.
EMBEDDING_DIM = 100
VOCAB_SIZE = 10000

# --- 1. Model and Tokenizer Loading ---
@st.cache_resource
def load_cnn_resources():
    """Loads the saved Keras model and tokenizer. Uses st.cache_resource 
    to ensure files are only loaded once."""
    try:
        # Load the Keras model
        model = load_model('cnn_sentiment_model.h5')
        
        # Load the Tokenizer
        with open('cnn_tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        return model, tokenizer
    except FileNotFoundError:
        st.error("🚨 Error: CNN model (`cnn_sentiment_model.h5`) or tokenizer (`cnn_tokenizer.pkl`) files not found. Please run **`train_model.py`** first to create them.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading resources. Ensure TensorFlow/Keras is installed correctly: {e}")
        st.stop()

# Load resources
model, tokenizer = load_cnn_resources()

# --- 2. Prediction Logic ---
def preprocess_text(text):
    """
    Cleans the input text, matching the preprocessing done during training.
    Removes HTML tags and converts to lowercase.
    """
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    return text

def predict_sentiment_cnn(review):
    """
    Takes a raw review, preprocesses it, tokenizes and pads it, and runs the prediction.
    
    Updated to include a 'Neutral' category based on probability thresholds.
    """
    # 1. Preprocess
    clean_review = preprocess_text(review)
    
    # 2. Convert to sequence using the loaded tokenizer
    sequence = tokenizer.texts_to_sequences([clean_review])
    
    # 3. Pad sequence (MUST match training sequence length)
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # 4. Predict
    # Prediction returns a probability array (e.g., [[0.98]])
    probability = model.predict(padded_sequence, verbose=0)[0][0]
    raw_prob = float(probability)
    
    # Define thresholds
    POSITIVE_THRESHOLD = 0.65 # Score above this is Positive
    NEGATIVE_THRESHOLD = 0.35 # Score below this is Negative
    
    # 5. Interpret results with 3 categories: Positive, Neutral, Negative
    if raw_prob > POSITIVE_THRESHOLD:
        sentiment = 'Positive'
        # Confidence is distance from 0.5, normalized (or just distance from 0.5)
        confidence = (raw_prob - 0.5) * 2
    elif raw_prob < NEGATIVE_THRESHOLD:
        sentiment = 'Negative'
        confidence = (0.5 - raw_prob) * 2
    else:
        sentiment = 'Neutral'
        # For Neutral, confidence is how close it is to the center 0.5
        confidence = 1.0 - (abs(raw_prob - 0.5) * 2) 

    # We return the determined sentiment and the raw probability score
    return sentiment, raw_prob

# Load resources
model, tokenizer = load_cnn_resources()

# --- 3. Sidebar Content (Model Context & Theme Toggle) ---
st.sidebar.header("⚙️ App Settings")

# Theme Toggle Button
theme_button_label = f"Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode"
st.sidebar.button(theme_button_label, on_click=toggle_theme, use_container_width=True)

st.sidebar.header("🧠 CNN Model Details")
st.sidebar.markdown("""
This tool uses a Convolutional Neural Network (CNN) specialized in **text analysis** to classify reviews as either positive or negative.
""")

st.sidebar.subheader("Configuration")
st.sidebar.markdown(f"""
- **Max Review Length:** `{MAX_SEQUENCE_LENGTH}` tokens
- **Vocabulary Size:** `{VOCAB_SIZE}` unique words
- **Embedding Dimensions:** `{EMBEDDING_DIM}` (Word Vector Size)
""")

st.sidebar.subheader("Architecture")
st.sidebar.markdown("""
The CNN architecture performs sequential feature extraction:
1.  **Embedding Layer**: Maps words to dense vectors.
2.  **Conv1D Layer**: Identifies local patterns (like key phrases).
3.  **Global Max Pooling**: Captures the most important features detected.
4.  **Classification Layers**: Outputs the final probability score.
""")
st.sidebar.markdown(f"""
---
**Sentiment Thresholds**
- **Positive:** Probability > 0.65
- **Neutral:** 0.35 ≤ Probability ≤ 0.65
- **Negative:** Probability < 0.35
""")


st.sidebar.caption("Trained on a subset of the IMDB Movie Review dataset.")


# --- 4. Main UI Content ---
st.title("Review Analyzer")
st.markdown("<p class='tagline'>Input any text to instantly analyze its sentiment (Positive or Negative).</p>", unsafe_allow_html=True)


# --- Centered and Smaller Input Section ---
# Use columns to create centered padding (1 unit left, 4 units center, 1 unit right)
col_left_pad, col_center, col_right_pad = st.columns([1, 4, 1])

with col_center:
    # Text Area is now contained in the center column
    # NOTE: Changed default text as the previous one ("not bad, worth it") fails due to negation handling by the model.
    user_input = st.text_area(
        "**Paste Your Review Here**", 
        "This product is amazing. Everything is perfect and it feels so good to use. Highly recommend.", 
        height=250, 
        placeholder="Example: 'The new feature is amazing, but the user interface is clumsy and frustrating.'"
    )

    # Centered button within the center column
    # Use another set of columns for the button to ensure it's centered relative to its parent column
    btn_col1, btn_col2, btn_col3 = st.columns([1, 3, 1])
    with btn_col2:
        analyze_button = st.button("**ANALYZE REVIEW**", type="primary", use_container_width=True) 

st.markdown("<div style='border-top: 1px solid var(--border-color, #444);'></div>", unsafe_allow_html=True) # Dynamic separator

if analyze_button:
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        # Use st.status for a polished loading effect
        with st.status('Analyzing... Running sequence through 1D CNN model...', expanded=True) as status:
            time.sleep(0.5) # Simulate processing time
            sentiment, raw_prob = predict_sentiment_cnn(user_input)
            
            # The 'confidence' for display will be the absolute distance from the 0.5 center, normalized (0 to 1)
            display_confidence = abs(raw_prob - 0.5) * 2
            
            # We allow the status to update, but the block will collapse, making it disappear
            status.update(label="Analysis Complete! ✅", state="complete", expanded=False)
        
        # Results Section should also be centered and narrowed
        with col_center:
            # --- 1cm (approx 10px) Vertical Gap Spacer (as previously requested) ---
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            
            # Permanent "Analysis Complete" message placed here, above the results header
            st.success("Analysis Complete! ✅ The model has finished processing your review.")

            st.markdown("## Analysis Results")
            
            # --- Display results with 3 categories ---
            if sentiment == 'Positive':
                st.success(f"Prediction: {sentiment}!")
                st.markdown(f'<div class="result-positive">⭐ The model classifies this text as **Positive**.</div>', unsafe_allow_html=True)
            elif sentiment == 'Negative':
                st.error(f"Prediction: {sentiment}.")
                st.markdown(f'<div class="result-negative">👎 The model classifies this text as **Negative**.</div>', unsafe_allow_html=True)
            else: # Neutral
                st.info(f"Prediction: {sentiment}.")
                st.markdown(f'<div class="result-neutral">⚖️ The model classifies this text as **Neutral**. The sentiment is moderate or mixed.</div>', unsafe_allow_html=True)

            
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

            # --- Determine colors for the custom metric based on current theme ---
            if st.session_state.theme == 'dark':
                label_color = '#f0f0f0'
                value_color = '#00bcd4'
                help_color = '#9e9e9e'
            else:
                # Use dark colors for high contrast in light mode
                label_color = '#212529'
                value_color = '#00a0ae'
                help_color = '#212529'
            # -------------------------------------------------------------------
            
            # Displaying Confidence (Strength) and Raw Probability in two columns
            res_col_conf, res_col_raw = st.columns(2)
            
            with res_col_conf:
                # The 'Strength Score' shows how far the review is from being perfectly neutral (0.5)
                st.metric(
                    label="Sentiment Strength Score", 
                    value=f"{display_confidence*100:.2f}%", 
                    help="Measures how far the model's score is from the 0.5 neutral point (0% is neutral, 100% is extreme positive/negative)."
                )
                st.progress(display_confidence)
                
            with res_col_raw:
                # Displaying the probability of positive sentiment explicitly
                label_text = "Positive Probability"
                value_text = f"{raw_prob:.4f}"
                
                # Using custom HTML for the second metric with explicit, theme-aware colors
                st.markdown(f"""
                <div style="padding: 0 0 10px 0;">
                    <label style="font-size: 1.1rem; color: {label_color}; font-weight: 600;">{label_text}</label>
                    <div style="font-size: 3rem; font-weight: 900; color: {value_color};">{value_text}</div>
                    <div style="font-size: 0.9rem; color: {help_color};"> (Score closer to 1.0 means more Positive)</div>
                </div>
                """, unsafe_allow_html=True)
            
        
st.markdown("---")
st.caption("Powered by a custom deep learning CNN model. Accuracy is dependent on the training data.")
