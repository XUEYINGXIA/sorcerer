import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

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

    def create_embedding(self, tokens, chunk_size=8000):
        """
        Create embeddings for the tokens using OpenAI's API
        Chunks the tokens if they exceed the model's token limit
        """
        print("Creating embeddings...")
        # Join tokens back into text for embedding
        text = " ".join(tokens)
        
        try:
            # Create embedding
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            print(f"Successfully created embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return None

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
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\nProcessing file {i}/{len(pdf_files)}")
        tokens, embedding = tokenizer.process_pdf(str(pdf_file))
        if tokens and embedding:
            pdf_results[pdf_file.name] = {
                'tokens': tokens,
                'embedding': embedding
            }
            print(f"Successfully processed: {pdf_file.name}")
        else:
            print(f"Failed to process: {pdf_file.name}")
    
    print(f"\nCompleted processing all files. Successfully processed: {len(pdf_results)}/{len(pdf_files)}")
    return pdf_results

if __name__ == "__main__":
    # Example usage
    directory_path = "books/test"
    results = process_directory(directory_path)
    
    # Print results
    for pdf_name, data in results.items():
        print(f"\nResults for {pdf_name}:")
        print(f"Number of tokens: {len(data['tokens'])}")
        print(f"Sample tokens: {data['tokens'][:10]}")
        print(f"Embedding dimension: {len(data['embedding'])}")
