from transformers import AutoModel, AutoTokenizer, LlamaModel, LlamaTokenizer
from peft import PeftModel
import torch
import torch.nn.functional as F
import openai
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import voyageai
import requests
import time

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
voyage_api_key = os.getenv('VOYAGE_API_KEY')

if openai.api_key is None:
    print("OpenAI API key not found in .env file. Please set OPENAI_API_KEY in your .env file.")
else:
    print("OpenAI API key loaded successfully.")

if voyage_api_key is None:
    print("Voyage API key not found in .env file. Please set VOYAGE_API_KEY in your .env file.")
else:
    print("Voyage API key loaded successfully.")
    voyageai.api_key = voyage_api_key


# Load BGE models and tokenizer
# Load base model
try:
    base_model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    # Load Peft model
    model = PeftModel.from_pretrained(base_model, "shashu2325/resume-job-matcher-lora")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    print("BGE models and tokenizer loaded successfully.")
except Exception as e:
    print(f"Failed to load BGE models or tokenizer: {e}")
    base_model = None
    model = None
    tokenizer = None

# Load BGE-M3 model
try:
    bge_m3_model = SentenceTransformer("BAAI/bge-m3")
    print("BGE-M3 model loaded successfully.")
except Exception as e:
    print(f"Failed to load BGE-M3 model: {e}")
    bge_m3_model = None

# Load CareerBERT model
try:
    careerbert_model = SentenceTransformer("jjzha/careerbert")
    print("CareerBERT model loaded successfully.")
except Exception as e:
    print(f"Failed to load CareerBERT model: {e}")
    careerbert_model = None

# Load ConFit V2 model (using all-mpnet-base-v2 as a proxy for ConFit)
try:
    confit_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("ConFit V2 model loaded successfully.")
except Exception as e:
    print(f"Failed to load ConFit V2 model: {e}")
    confit_model = None

# Load LLaMA-3.1 for embeddings (using sentence-transformers wrapper)
try:
    # Note: This is a placeholder - actual LLaMA-3.1 embedding implementation may vary
    llama_model = None  # Will need specific implementation
    print("LLaMA-3.1 model setup (placeholder).")
except Exception as e:
    print(f"Failed to load LLaMA-3.1 model: {e}")
    llama_model = None


def calculate_bge_similarity(resume_text, job_text):
    """Calculates similarity between resume and job text using BGE embeddings."""
    if model is None or tokenizer is None:
        print("BGE models or tokenizer not loaded. Cannot calculate similarity.")
        return None

    try:
        # Process texts
        resume_inputs = tokenizer(resume_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        job_inputs = tokenizer(job_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)

        # Get embeddings
        with torch.no_grad():
            # Get embeddings using mean pooling
            resume_outputs = model(**resume_inputs)
            job_outputs = model(**job_inputs)

            # Mean pooling
            resume_emb = resume_outputs.last_hidden_state.mean(dim=1)
            job_emb = job_outputs.last_hidden_state.mean(dim=1)

            # Normalize and calculate similarity
            resume_emb = F.normalize(resume_emb, p=2, dim=1)
            job_emb = F.normalize(job_emb, p=2, dim=1)

            similarity = torch.sum(resume_emb * job_emb, dim=1)
            match_score = torch.sigmoid(similarity).item()

        return match_score
    except Exception as e:
        print(f"Error calculating BGE similarity: {e}")
        return None

def get_openai_embedding(text, model="text-embedding-3-small"):
    """Gets OpenAI embedding for a given text."""
    if openai.api_key is None:
        print("OpenAI API key not set. Cannot get embedding.")
        return None
    try:
        text = text.replace("\n", " ")
        return openai.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error getting OpenAI embedding: {e}")
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a.dot(b) / (na * nb))


def calculate_openai_similarity(resume_text, job_text):
    """Calculates cosine similarity between resume and job text using OpenAI embeddings."""
    resume_embedding = get_openai_embedding(resume_text)
    job_embedding = get_openai_embedding(job_text)

    if resume_embedding is None or job_embedding is None:
        print("Failed to get OpenAI embeddings. Cannot calculate similarity.")
        return None

    # Calculate similarity using the cosine function
    similarity_score = cosine_similarity(resume_embedding, job_embedding)

    return similarity_score

def calculate_bge_m3_similarity(resume_text, job_text):
    """Calculates similarity using BGE-M3 model."""
    if bge_m3_model is None:
        print("BGE-M3 model not loaded. Cannot calculate similarity.")
        return None
    
    try:
        resume_embedding = bge_m3_model.encode(resume_text)
        job_embedding = bge_m3_model.encode(job_text)
        similarity_score = cosine_similarity(resume_embedding, job_embedding)
        return similarity_score
    except Exception as e:
        print(f"Error calculating BGE-M3 similarity: {e}")
        return None

def calculate_careerbert_similarity(resume_text, job_text):
    """Calculates similarity using CareerBERT model."""
    if careerbert_model is None:
        print("CareerBERT model not loaded. Cannot calculate similarity.")
        return None
    
    try:
        resume_embedding = careerbert_model.encode(resume_text)
        job_embedding = careerbert_model.encode(job_text)
        similarity_score = cosine_similarity(resume_embedding, job_embedding)
        return similarity_score
    except Exception as e:
        print(f"Error calculating CareerBERT similarity: {e}")
        return None

def calculate_confit_similarity(resume_text, job_text):
    """Calculates similarity using ConFit V2 model."""
    if confit_model is None:
        print("ConFit V2 model not loaded. Cannot calculate similarity.")
        return None
    
    try:
        resume_embedding = confit_model.encode(resume_text)
        job_embedding = confit_model.encode(job_text)
        similarity_score = cosine_similarity(resume_embedding, job_embedding)
        return similarity_score
    except Exception as e:
        print(f"Error calculating ConFit similarity: {e}")
        return None

def get_voyage_embedding(text, model="voyage-3-large"):
    """Gets Voyage AI embedding for a given text."""
    if voyage_api_key is None:
        print("Voyage API key not set. Cannot get embedding.")
        return None
    
    try:
        vo = voyageai.Client()
        result = vo.embed([text], model=model)
        return result.embeddings[0]
    except Exception as e:
        print(f"Error getting Voyage embedding: {e}")
        return None

def calculate_voyage_similarity(resume_text, job_text):
    """Calculates similarity using Voyage AI embeddings."""
    resume_embedding = get_voyage_embedding(resume_text)
    job_embedding = get_voyage_embedding(job_text)
    
    if resume_embedding is None or job_embedding is None:
        print("Failed to get Voyage embeddings. Cannot calculate similarity.")
        return None
    
    similarity_score = cosine_similarity(resume_embedding, job_embedding)
    return similarity_score

def calculate_llama_similarity(resume_text, job_text):
    """Placeholder for LLaMA-3.1 similarity calculation."""
    # This is a placeholder - actual implementation would depend on available LLaMA embedding service
    print("LLaMA-3.1 similarity calculation not implemented yet.")
    return None

###############################################################################################################################################################
df = pd.read_csv('dataset.csv')

###############################################################################################################################################################
select_df = df[df['Decision'] == 'select']
reject_df = df[df['Decision'] == 'reject']

# Determine sample sizes. Aim for equal representation if possible, or adjust if one category is much smaller.
total_samples = 10
num_select = min(len(select_df), total_samples // 2)
num_reject = min(len(reject_df), total_samples - num_select)

# Adjust if one category is much smaller and we couldn't get 50 total with the initial split
if num_select + num_reject < total_samples:
    if len(select_df) > len(reject_df):
        num_select = min(len(select_df), total_samples - num_reject)
    else:
        num_reject = min(len(reject_df), total_samples - num_select)

sampled_select_df = select_df.sample(n=num_select, random_state=42) # Use random_state for reproducibility
sampled_reject_df = reject_df.sample(n=num_reject, random_state=42)

sampled_df = pd.concat([sampled_select_df, sampled_reject_df])

###############################################################################################################################################################
# Shuffle the combined sample to mix the select and reject rows
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize score lists for all models
bge_scores = []
openai_scores = []
bge_m3_scores = []
careerbert_scores = []
confit_scores = []
voyage_scores = []
llama_scores = []

for index, row in sampled_df.iterrows():
    resume_text = str(row['Resume']) if pd.notna(row['Resume']) else ""
    job_text = str(row['Job_Description']) if pd.notna(row['Job_Description']) else ""

    print(f"\nProcessing row {index}/{len(sampled_df)-1}")

    # Calculate BGE similarity
    try:
        print(f"  Calculating BGE similarity...")
        bge_score = calculate_bge_similarity(resume_text, job_text)
        print(f"  BGE similarity: {bge_score}")
    except Exception as e:
        print(f"  Error calculating BGE similarity: {e}")
        bge_score = None
    bge_scores.append(bge_score)

    # Calculate OpenAI similarity
    try:
        if openai.api_key is None:
            print(f"  Skipping OpenAI similarity (API key not set)")
            openai_score = None
        else:
            print(f"  Calculating OpenAI similarity...")
            openai_score = calculate_openai_similarity(resume_text, job_text)
            print(f"  OpenAI similarity: {openai_score}")
    except Exception as e:
        print(f"  Error calculating OpenAI similarity: {e}")
        openai_score = None
    openai_scores.append(openai_score)

    # Calculate BGE-M3 similarity
    try:
        print(f"  Calculating BGE-M3 similarity...")
        bge_m3_score = calculate_bge_m3_similarity(resume_text, job_text)
        print(f"  BGE-M3 similarity: {bge_m3_score}")
    except Exception as e:
        print(f"  Error calculating BGE-M3 similarity: {e}")
        bge_m3_score = None
    bge_m3_scores.append(bge_m3_score)

    # Calculate CareerBERT similarity
    try:
        print(f"  Calculating CareerBERT similarity...")
        careerbert_score = calculate_careerbert_similarity(resume_text, job_text)
        print(f"  CareerBERT similarity: {careerbert_score}")
    except Exception as e:
        print(f"  Error calculating CareerBERT similarity: {e}")
        careerbert_score = None
    careerbert_scores.append(careerbert_score)

    # Calculate ConFit similarity
    try:
        print(f"  Calculating ConFit similarity...")
        confit_score = calculate_confit_similarity(resume_text, job_text)
        print(f"  ConFit similarity: {confit_score}")
    except Exception as e:
        print(f"  Error calculating ConFit similarity: {e}")
        confit_score = None
    confit_scores.append(confit_score)

    # Calculate Voyage similarity
    try:
        if voyage_api_key is None:
            print(f"  Skipping Voyage similarity (API key not set)")
            voyage_score = None
        else:
            print(f"  Calculating Voyage similarity...")
            voyage_score = calculate_voyage_similarity(resume_text, job_text)
            print(f"  Voyage similarity: {voyage_score}")
    except Exception as e:
        print(f"  Error calculating Voyage similarity: {e}")
        voyage_score = None
    voyage_scores.append(voyage_score)

    # Calculate LLaMA similarity (placeholder)
    try:
        print(f"  Calculating LLaMA similarity...")
        llama_score = calculate_llama_similarity(resume_text, job_text)
        print(f"  LLaMA similarity: {llama_score}")
    except Exception as e:
        print(f"  Error calculating LLaMA similarity: {e}")
        llama_score = None
    llama_scores.append(llama_score)

print("\nScore calculation complete.")

###############################################################################################################################################################
# Add the scores as new columns to the sampled DataFrame
sampled_df['bge_similarity'] = bge_scores
sampled_df['openai_similarity'] = openai_scores
sampled_df['bge_m3_similarity'] = bge_m3_scores
sampled_df['careerbert_similarity'] = careerbert_scores
sampled_df['confit_similarity'] = confit_scores
sampled_df['voyage_similarity'] = voyage_scores
sampled_df['llama_similarity'] = llama_scores

# Define threshold for classification
threshold = 0.5

# Create predictions for all models
def make_prediction(score, threshold=0.5):
    if score is None or pd.isna(score):
        return 'unknown'
    return 'select' if score > threshold else 'reject'

sampled_df['bge_prediction'] = sampled_df['bge_similarity'].apply(lambda x: make_prediction(x, threshold))
sampled_df['openai_prediction'] = sampled_df['openai_similarity'].apply(lambda x: make_prediction(x, threshold))
sampled_df['bge_m3_prediction'] = sampled_df['bge_m3_similarity'].apply(lambda x: make_prediction(x, threshold))
sampled_df['careerbert_prediction'] = sampled_df['careerbert_similarity'].apply(lambda x: make_prediction(x, threshold))
sampled_df['confit_prediction'] = sampled_df['confit_similarity'].apply(lambda x: make_prediction(x, threshold))
sampled_df['voyage_prediction'] = sampled_df['voyage_similarity'].apply(lambda x: make_prediction(x, threshold))
sampled_df['llama_prediction'] = sampled_df['llama_similarity'].apply(lambda x: make_prediction(x, threshold))

# Display the comparative table
print("\nComparative Table of Similarity Scores and Predictions:")
display_columns = ['Role', 'Decision', 
                  'bge_similarity', 'bge_prediction',
                  'openai_similarity', 'openai_prediction', 
                  'bge_m3_similarity', 'bge_m3_prediction',
                  'careerbert_similarity', 'careerbert_prediction',
                  'confit_similarity', 'confit_prediction',
                  'voyage_similarity', 'voyage_prediction',
                  'llama_similarity', 'llama_prediction']

print(sampled_df[display_columns])

###############################################################################################################################################################
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Define models to evaluate
models = {
    'BGE': ('bge_prediction', 'bge_similarity'),
    'OpenAI': ('openai_prediction', 'openai_similarity'),
    'BGE-M3': ('bge_m3_prediction', 'bge_m3_similarity'),
    'CareerBERT': ('careerbert_prediction', 'careerbert_similarity'),
    'ConFit V2': ('confit_prediction', 'confit_similarity'),
    'Voyage-3-Large': ('voyage_prediction', 'voyage_similarity'),
    'LLaMA-3.1': ('llama_prediction', 'llama_similarity')
}

# Store results for comparison
results_summary = []

print("\n" + "="*80)
print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
print("="*80)

for model_name, (pred_col, sim_col) in models.items():
    print(f"\n{model_name} Model Performance:")
    print("-" * 50)
    
    # Filter out rows with unknown predictions
    valid_mask = sampled_df[pred_col] != 'unknown'
    valid_df = sampled_df[valid_mask]
    
    if len(valid_df) == 0:
        print(f"No valid predictions for {model_name}")
        results_summary.append({
            'Model': model_name,
            'Accuracy': None,
            'Precision': None,
            'Recall': None,
            'F1': None,
            'Valid_Samples': 0
        })
        continue
    
    try:
        # Calculate metrics
        accuracy = accuracy_score(valid_df['Decision'], valid_df[pred_col])
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_df['Decision'], valid_df[pred_col], average='weighted'
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Valid Samples: {len(valid_df)}/{len(sampled_df)}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(valid_df['Decision'], valid_df[pred_col]))
        
        # Store results
        results_summary.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Valid_Samples': len(valid_df)
        })
        
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        results_summary.append({
            'Model': model_name,
            'Accuracy': None,
            'Precision': None,
            'Recall': None,
            'F1': None,
            'Valid_Samples': len(valid_df)
        })

# Create summary comparison table
print("\n" + "="*80)
print("SUMMARY COMPARISON TABLE")
print("="*80)

results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False, float_format='%.4f'))

# Find best performing model
valid_results = results_df[results_df['Accuracy'].notna()]
if len(valid_results) > 0:
    best_model = valid_results.loc[valid_results['Accuracy'].idxmax()]
    print(f"\nBest Performing Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})")

print("\n" + "="*80)