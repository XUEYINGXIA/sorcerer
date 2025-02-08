import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
import psycopg2
import numpy as np

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class PDFTokenizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.db_params = {
            'dbname': 'sorcerer',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': '5432'
        }
        self.max_tokens_per_chunk = 4000  # Safe limit for OpenAI's API
    
    def read_pdf(self, pdf_path):
        """
        Read a PDF file and extract its text content
        """
        print(f"Starting to read PDF: {pdf_path}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text()
                    print(f"Processed page {page_num + 1}/{len(pdf_reader.pages)}")
            print(f"Successfully read PDF: {pdf_path}")
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    def preprocess_text(self, text):
        """
        Clean and preprocess the extracted text
        """
        print("Starting text preprocessing...")
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        """
        Tokenize the text and remove stop words
        """
        print("Starting tokenization...")
        tokens = word_tokenize(text)
        print(f"Initial token count: {len(tokens)}")
        
        # Remove stop words and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and len(token) > 2]
        print(f"Final token count after removing stop words: {len(tokens)}")
        return tokens

    def create_embedding(self, tokens):
        """
        Create embeddings for the tokens using OpenAI's API
        Processes tokens in chunks to stay within API limits
        """
        print("Creating embeddings...")
        embeddings = []
        
        # Process tokens in chunks
        for i in range(0, len(tokens), self.max_tokens_per_chunk):
            chunk = tokens[i:i + self.max_tokens_per_chunk]
            chunk_text = " ".join(chunk)
            print(f"Processing chunk {i//self.max_tokens_per_chunk + 1} of {(len(tokens)-1)//self.max_tokens_per_chunk + 1}")
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk_text,
                    encoding_format="float"
                )
                chunk_embedding = response.data[0].embedding
                embeddings.append(chunk_embedding)
                print(f"Successfully created embedding for chunk {i//self.max_tokens_per_chunk + 1}")
            except Exception as e:
                print(f"Error creating embedding for chunk: {e}")
                return None
        
        if not embeddings:
            return None
            
        # Average all chunk embeddings to get a single embedding for the entire text
        final_embedding = [sum(x)/len(embeddings) for x in zip(*embeddings)]
        print(f"Successfully created final embedding of dimension {len(final_embedding)}")
        return final_embedding

    def process_pdf(self, pdf_path):
        """
        Process a PDF file and return its tokens and embedding
        """
        print(f"\nProcessing PDF: {pdf_path}")
        # Read PDF
        text = self.read_pdf(pdf_path)
        if not text:
            print("Failed to process PDF - no text extracted")
            return None, None
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = self.tokenize(clean_text)
        
        # Create embedding
        embedding = self.create_embedding(tokens)
        
        print(f"Completed processing PDF: {pdf_path}")
        return tokens, embedding

    def save_to_db(self, book_name, tokens, embedding):
        """
        Save the tokens and embedding to the database
        """
        print(f"Saving to database: {book_name}")
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            # Convert embedding to proper format and tokens to array
            embedding_list = list(embedding)
            tokens_array = list(tokens)
            
            # Insert into database
            cur.execute("""
                INSERT INTO book_embeddings (book_name, embedding, tokens)
                VALUES (%s, %s, %s)
            """, (book_name, embedding_list, tokens_array))
            
            conn.commit()
            print(f"Successfully saved {book_name} to database")
        except Exception as e:
            print(f"Error saving to database: {e}")
        finally:
            cur.close()
            conn.close()

def process_directory(directory_path):
    """
    Process all PDFs in a directory
    """
    print(f"\nStarting to process directory: {directory_path}")
    tokenizer = PDFTokenizer()
    pdf_results = {}
    
    directory = Path(directory_path)
    pdf_files = list(directory.glob('*.pdf'))
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process only the first PDF file
    if pdf_files:
        pdf_file = pdf_files[0]
        print(f"\nProcessing file: {pdf_file.name}")
        tokens, embedding = tokenizer.process_pdf(str(pdf_file))
        if tokens and embedding:
            pdf_results[pdf_file.name] = {
                'tokens': tokens,
                'embedding': embedding
            }
            # Save to database
            tokenizer.save_to_db(pdf_file.name, tokens, embedding)
            print(f"Successfully processed and saved: {pdf_file.name}")
        else:
            print(f"Failed to process: {pdf_file.name}")
    
    return pdf_results

if __name__ == "__main__":
    # Process PDFs from the test directory
    directory_path = "books/test"
    results = process_directory(directory_path)
    
    # Print results
    for pdf_name, data in results.items():
        print(f"\nResults for {pdf_name}:")
        print(f"Number of tokens: {len(data['tokens'])}")
        print(f"Sample tokens: {data['tokens'][:10]}")
        print(f"Embedding dimension: {len(data['embedding'])}")
