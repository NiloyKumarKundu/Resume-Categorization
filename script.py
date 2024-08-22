import os
import re
import pdfplumber
import pickle
import shutil
import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import argparse

# Ensure required NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing functions
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = text.strip()  # Remove leading/trailing whitespace
    return text

def further_preprocess_text(text):
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    processed_text = " ".join(tokens)
    return processed_text

def remove_numbers(text):
    text = re.sub(r"\d+", "", text)
    return text

def process(text):
    text = preprocess_text(text)
    text = further_preprocess_text(text)
    preprocessed_text = remove_numbers(text)
    return preprocessed_text

# Function to get document embedding
def get_document_embedding(text, vectorizer):
    if isinstance(text, str):
        text = [text]
    embedding = vectorizer.transform(text)
    return embedding

# Function to predict category using a model
def predict_category(embedding, model, label_encoder):
    prediction = model.predict(embedding)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

# Main function to categorize resumes
def categorize_resumes(resume_dir):
    categorized_resumes = []
    
    for filename in os.listdir(resume_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(resume_dir, filename)
            try:
                # Extract text from PDF
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                
                if text:
                    # Preprocess the text
                    preprocessed_text = process(text)
                    
                    # Get document embedding
                    embedding = get_document_embedding(preprocessed_text, vectorizer)
                    
                    # Predict category
                    predicted_category = predict_category(embedding, model, label_encoder)
                    print(f"File '{filename}' categorized as '{predicted_category}'")
                    
                    # Create category folder if it doesn't exist
                    category_dir = os.path.join(resume_dir, predicted_category)
                    os.makedirs(category_dir, exist_ok=True)
                    
                    # Move the file to the category folder
                    new_file_path = os.path.join(category_dir, filename)
                    shutil.move(file_path, new_file_path)
                    
                    # Store filename and category in the list
                    categorized_resumes.append([filename, predicted_category])
                    
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    # Write results to CSV file
    csv_file_path = os.path.join(args.resume_dir, 'categorized_resumes.csv')
    try:
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['filename', 'category']) 
            csv_writer.writerows(categorized_resumes)  
    except Exception as e:
        print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Categorize resumes based on a pre-trained model.')
    parser.add_argument('resume_dir', type=str, help='The directory containing the resume PDFs to categorize.')
    args = parser.parse_args()

    model_file_name = "Logistic Regression.sav"
    vectorizer_file_name = 'tfidf_vectorizer.pkl'
    label_encoder_file_name = "label_encoder.pkl"

    model_directory = os.getcwd() + '/models'
    model_file_path = os.path.join(model_directory, model_file_name)
    vectorizer_file_path = os.path.join(model_directory, vectorizer_file_name)
    label_encoder_file_path = os.path.join(model_directory, label_encoder_file_name)
    
    with open(model_file_path, 'rb') as model_file, open(vectorizer_file_path, 'rb') as vectorizer_file, open(label_encoder_file_path, 'rb') as label_encoder_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)
        label_encoder = pickle.load(label_encoder_file)
    
    categorize_resumes(args.resume_dir)
