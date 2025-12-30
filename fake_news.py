# ============================================
# FAKE NEWS DETECTION USING BAGGING (STREAMLIT)
# ============================================

import pandas as pd
import re
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.subheader("Machine Learning Project using Bagging Algorithm")
st.write("---")

# --------------------------------------------
# LOAD DATASET
# --------------------------------------------
try:
    data = pd.read_csv("fake_news_dataset.csv")
    st.success("Dataset Loaded Successfully")
except:
    st.error("Dataset not found! Place 'fake_news_dataset.csv' in same folder.")
    st.stop()

# --------------------------------------------
# DATASET PREVIEW
# --------------------------------------------
with st.expander("🔍 View Dataset"):
    st.dataframe(data.head())

# --------------------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# --------------------------------------------
# APPLY TEXT CLEANING
# --------------------------------------------
data['clean_text'] = data['text'].apply(clean_text)

X = data['clean_text']
y = data['label']

# --------------------------------------------
# TF-IDF FEATURE EXTRACTION
# --------------------------------------------
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# --------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------
# BAGGING CLASSIFIER
# --------------------------------------------
base_model = DecisionTreeClassifier(random_state=42)

bagging_model = BaggingClassifier(
    estimator=base_model,
    n_estimators=100,
    random_state=42
)

# --------------------------------------------
# TRAIN MODEL
# --------------------------------------------
bagging_model.fit(X_train, y_train)

st.success("Model Training Completed")

# --------------------------------------------
# MODEL EVALUATION
# --------------------------------------------
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("### 📊 Model Performance")
st.info(f"Accuracy: {accuracy*100:.2f}%")

with st.expander("📄 Classification Report"):
    st.text(classification_report(y_test, y_pred))

with st.expander("🧮 Confusion Matrix"):
    st.write(confusion_matrix(y_test, y_pred))

# --------------------------------------------
# SAVE RESULTS FOR POWER BI
# --------------------------------------------
results = pd.DataFrame({
    'Actual_Label': y_test.map({0: 'Real', 1: 'Fake'}),
    'Predicted_Label': pd.Series(y_pred).map({0: 'Real', 1: 'Fake'})
})

results.to_csv("fake_news_results.csv", index=False)

st.success("Results exported as fake_news_results.csv (For Power BI)")

# --------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = tfidf.transform([cleaned])
    prediction = bagging_model.predict(vector)

    if prediction[0] == 0:
        return "✅ REAL NEWS"
    else:
        return "❌ FAKE NEWS"

# --------------------------------------------
# LIVE NEWS TESTING (WEBSITE PART)
# --------------------------------------------
st.write("---")
st.write("## 🔴 Live News Detection")

user_news = st.text_area(
    "Enter news text below 👇",
    height=150
)

if st.button("Check News"):
    if user_news.strip() == "":
        st.warning("Please enter some news text")
    else:
        result = predict_news(user_news)
        if "REAL" in result:
            st.success(result)
        else:
            st.error(result)

# --------------------------------------------
# FOOTER
# --------------------------------------------
st.write("---")
st.caption("Fake News Detection | Ensemble Learning | Bagging Classifier")
