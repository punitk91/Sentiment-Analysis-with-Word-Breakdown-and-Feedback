import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained BERT model for sentiment classification (5 levels of sentiment)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Pipeline for sentiment analysis
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Expanded list of positive and negative words
positive_words = ["affordable", "great", "satisfactory", "performance", "excellent", "positive", "amazing", "happy", "joy", "love", "good"]
negative_words = ["slow", "frustrating", "poor", "bad", "negative", "terrible", "sad", "angry", "hate", "bad", "disappointed"]

# Mapping the star ratings to descriptive labels
star_to_sentiment = {
    "1 star": "Very Negative",
    "2 stars": "Negative",
    "3 stars": "Neutral",
    "4 stars": "Positive",
    "5 stars": "Very Positive"
}

# Sample data collection for user feedback and overall sentiment trend
user_feedback_data = {'text': [], 'model_label': [], 'user_corrected_label': []}
sentiment_trend = []

# Streamlit UI for sentiment analysis
st.title("Interactive Sentiment Analysis with Word Breakdown and Feedback")

# Description of the app
st.markdown("""
Welcome to the **Interactive Sentiment Analysis POC**! This app lets you input any text, get sentiment predictions from a BERT model, and submit feedback on the prediction accuracy. Over time, the app will show how user feedback contributes to refining the model's performance. Watch the **Sentiment Trend Chart** to see how sentiment predictions evolve!
""")

# Text input for the user
input_text = st.text_area("Enter a long sentence to analyze sentiment:", height=200)

# Function to break the sentence into words and analyze sentiment
def analyze_sentence(sentence):
    words = sentence.split()  # Split the sentence into words
    positive_count = 0
    negative_count = 0
    
    # Analyze each word
    for word in words:
        word_lower = word.lower().strip(",.!")  # Normalize the word
        if word_lower in positive_words:
            positive_count += 1
        elif word_lower in negative_words:
            negative_count += 1
    
    total_words = len(words)
    if total_words > 0:
        sentiment_score = (positive_count - negative_count) / total_words * 100  # Calculate a simple sentiment score
    else:
        sentiment_score = 0
    
    return positive_count, negative_count, sentiment_score

# Button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if input_text:
        # Perform BERT sentiment analysis on the input text
        result = classifier(input_text)[0]
        sentiment_label = star_to_sentiment[result['label']]
        sentiment_score_bert = result['score']

        # Display the BERT sentiment
        st.subheader("Sentiment Prediction by BERT")
        st.write(f"**Sentiment**: {sentiment_label}")
        st.write(f"**Confidence**: {sentiment_score_bert:.2f}")
        
        # Update sentiment trend
        sentiment_trend.append(sentiment_label)

        # Perform word-by-word analysis
        positive_count, negative_count, sentiment_score = analyze_sentence(input_text)

        # Display the word-based sentiment analysis
        st.subheader("Word Breakdown Sentiment Analysis")
        st.write(f"Total words: {len(input_text.split())}")
        st.write(f"Positive words: {positive_count}")
        st.write(f"Negative words: {negative_count}")
        st.write(f"**Average Sentiment Score**: {sentiment_score:.2f}% (Positive/Negative Ratio)")
        
        # Determine overall sentiment based on the average score
        if sentiment_score > 0:
            st.success("Overall Sentiment (Word-based): Positive")
        elif sentiment_score < 0:
            st.error("Overall Sentiment (Word-based): Negative")
        else:
            st.info("Overall Sentiment (Word-based): Neutral")

        # Collect feedback from the user
        st.subheader("Was the prediction accurate?")
        feedback_label = st.radio("Select your response:", ('Yes', 'No'))

        if feedback_label == 'No':
            correct_sentiment = st.selectbox(
                "What should the correct sentiment be?", list(star_to_sentiment.values()))
            if st.button("Submit Feedback"):
                # Append feedback data and correct the label
                user_feedback_data['text'].append(input_text)
                user_feedback_data['model_label'].append(sentiment_label)
                user_feedback_data['user_corrected_label'].append(correct_sentiment)
                st.success("Feedback submitted! Thank you for helping improve the model.")
        elif feedback_label == 'Yes':
            st.success("Great! Thanks for confirming the prediction.")
            user_feedback_data['text'].append(input_text)
            user_feedback_data['model_label'].append(sentiment_label)
            user_feedback_data['user_corrected_label'].append(sentiment_label)  # No correction needed

# Function to display user feedback data
def display_feedback():
    if user_feedback_data['text']:
        df = pd.DataFrame(user_feedback_data)
        st.write("User Feedback Data:")
        st.dataframe(df)
    else:
        st.write("No feedback submitted yet.")

# Button to display feedback data
if st.button("View Feedback Data"):
    display_feedback()

# Dynamic Sentiment Trend Chart
if sentiment_trend:
    st.subheader("Sentiment Trend Over Time")
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_trend, marker='o')
    plt.xlabel("Input Instances")
    plt.ylabel("Sentiment")
    plt.title("Sentiment Predictions Over Time")
    plt.grid(True)
    st.pyplot(plt)

# Additional feedback
st.write("Curious how feedback impacts the trend? Submit more inputs and feedback!")
