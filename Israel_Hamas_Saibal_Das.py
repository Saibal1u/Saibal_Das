import json
import re
import pandas as pd
from tqdm import tqdm

# Load the JSON file with the correct encoding
file_path = "C:/Users/SAIBAL DAS/Downloads/news.article.json"
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        articles = json.load(file)
except UnicodeDecodeError as e:
    print(f"Error reading the JSON file: {e}")
    articles = []

# Check the structure of a few articles
print(f"Total articles: {len(articles)}")
if len(articles) > 0:
    print("Example article structure:")
    print(json.dumps(articles[0], indent=2))

# Define a function to clean the text
def clean_text(text):
    # Remove special characters, punctuation, and extra spaces
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Filter articles related to the Israel-Hamas war
relevant_articles = []
for article in tqdm(articles):
    if 'articleBody' in article and 'title' in article:
        # Debugging: Print a sample content to check the data
        print(f"Checking article with title: {article['title'][:50]}")  # Print the first 50 characters of the title
        print(f"Content preview: {article['articleBody'][:200]}")  # Print the first 200 characters of the content
        if 'israel' in article['articleBody'].lower() and 'hamas' in article['articleBody'].lower():
            cleaned_content = clean_text(article['articleBody'])
            relevant_articles.append({
                'title': article['title'],
                'content': cleaned_content,
                'date': article['dateModified']['$date'] if 'dateModified' in article else article['scrapedDate']['$date']
            })

# Convert to DataFrame for easier handling
df = pd.DataFrame(relevant_articles)

# Debugging output to verify contents
print(f"Number of relevant articles: {len(df)}")
print(df.head())

# Ensure DataFrame is not empty before proceeding
if df.empty:
    print("No relevant articles found. Exiting.")
    exit()

# Model Setup
from transformers import pipeline

# Load a pre-trained QA model and tokenizer
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define functions for retrieving relevant article and answering questions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a TF-IDF vectorizer and fit it on the article contents
vectorizer = TfidfVectorizer().fit(df['content'])

def get_relevant_article(question, df):
    # Transform the question and articles using the TF-IDF vectorizer
    question_vector = vectorizer.transform([question])
    article_vectors = vectorizer.transform(df['content'])
    
    # Compute cosine similarities between the question and all articles
    similarities = cosine_similarity(question_vector, article_vectors).flatten()
    
    # Get the index of the most similar article
    most_similar_article_idx = similarities.argmax()
    
    return df.iloc[most_similar_article_idx]

def answer_question(question, df, qa_pipeline):
    # Get the most relevant article
    article = get_relevant_article(question, df)
    
    # Use the QA model to get an answer from the article content
    answer = qa_pipeline({
        'question': question,
        'context': article['content']
    })
    
    return answer['answer']

# Test the QA system with an example question
question = "What happened at the Al-Shifa Hospital?"
answer = answer_question(question, df, qa_pipeline)
print(f"Question: {question}")
print(f"Answer: {answer}")
