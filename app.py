import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---- NLTK downloads (IMPORTANT for deployment) ----
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

@st.cache_data
def get_stopwords():
    return set(stopwords.words('english'))

STOP_WORDS = get_stopwords()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # keep alphanumeric only
    tokens = [i for i in tokens if i.isalnum()]

    # remove stopwords
    tokens = [i for i in tokens if i not in STOP_WORDS]

    # stemming
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)

# ---- Load model & vectorizer ----
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---- UI ----
st.title("📧 Email / SMS Spam Classifier")

input_text = st.text_area("Enter the message")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_text = transform_text(input_text)
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
