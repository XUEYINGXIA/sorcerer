import PyPDF2
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
import psycopg2
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import concurrent.futures
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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
        self.chunk_size = 1000  # Number of words per chunk
        self.overlap = 100  # Number of words to overlap between chunks
        self.max_tokens_per_embedding = 8000  # Safe limit for OpenAI's API
        self.max_workers = 5  # Maximum number of parallel workers
    
    def read_pdf(self, pdf_path: str) -> str:
        """Read a PDF file and extract its text content"""
        print(f"Starting to read PDF: {pdf_path}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text() + " "
                    print(f"Processed page {page_num + 1}/{len(pdf_reader.pages)}")
            print(f"Successfully read PDF: {pdf_path}")
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the extracted text"""
        print("Starting text preprocessing...")
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep sentence punctuation
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        print("Splitting text into chunks...")
        # First split into sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            sentence_size = len(words)
            
            if current_size + sentence_size > self.chunk_size:
                # Store current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # Start new chunk with overlap
                if len(current_chunk) > self.overlap:
                    current_chunk = current_chunk[-self.overlap:]
                    current_size = len(current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.extend(words)
            current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return None

    def process_chunk(self, chunk: str) -> Dict[str, Any]:
        """Process a single chunk of text"""
        # Tokenize
        tokens = word_tokenize(chunk)
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and len(token) > 2]
        
        # Create embedding
        embedding = self.create_embedding(chunk)
        
        if embedding:
            return {
                'tokens': tokens,
                'embedding': embedding,
                'raw_text': chunk
            }
        return None

    def save_to_db(self, book_name: str, chunk_data: Dict[str, Any], chunk_num: int) -> bool:
        """Save a chunk's data to the database"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            # Convert data to proper format
            embedding_list = list(chunk_data['embedding'])
            tokens_array = list(chunk_data['tokens'])
            
            # Insert into database with chunk number
            cur.execute("""
                INSERT INTO book_embeddings 
                (book_name, chunk_number, embedding, tokens, raw_text)
                VALUES (%s, %s, %s, %s, %s)
            """, (book_name, chunk_num, embedding_list, tokens_array, chunk_data['raw_text']))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving chunk {chunk_num} of {book_name}: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    def process_and_save_chunk(self, args: Tuple[str, str, int]) -> bool:
        """Process and save a single chunk (for parallel processing)"""
        book_name, chunk, chunk_num = args
        chunk_data = self.process_chunk(chunk)
        if chunk_data:
            return self.save_to_db(book_name, chunk_data, chunk_num)
        return False

    def cleanup_existing_book(self, book_name: str) -> None:
        """Remove all existing chunks for a book before reprocessing"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            # Delete all entries for this book
            cur.execute("""
                DELETE FROM book_embeddings
                WHERE book_name = %s;
            """, (book_name,))
            
            conn.commit()
            print(f"Cleaned up existing data for {book_name}")
        except Exception as e:
            print(f"Error cleaning up existing data: {e}")
        finally:
            cur.close()
            conn.close()

    def process_pdf(self, pdf_path: str) -> bool:
        """Process a PDF file and store its chunks in the database"""
        print(f"\nProcessing PDF: {pdf_path}")
        
        # Get book name
        book_name = Path(pdf_path).name
        
        # Clean up existing data for this book
        print(f"Cleaning up existing data for {book_name}...")
        self.cleanup_existing_book(book_name)
        
        # Read PDF
        text = self.read_pdf(pdf_path)
        if not text:
            print("Failed to process PDF - no text extracted")
            return False
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        # Split into chunks
        chunks = self.split_into_chunks(clean_text)
        
        # Prepare arguments for parallel processing
        chunk_args = [(book_name, chunk, i) for i, chunk in enumerate(chunks, 1)]
        success_count = 0
        
        # Process chunks in parallel
        print(f"\nProcessing {len(chunks)} chunks in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Use tqdm for progress bar
            futures = {executor.submit(self.process_and_save_chunk, arg): arg for arg in chunk_args}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunks), desc="Processing chunks"):
                if future.result():
                    success_count += 1
        
        print(f"\nCompleted processing {book_name}")
        print(f"Successfully processed {success_count} out of {len(chunks)} chunks")
        return success_count > 0

    def generate_response(self, query: str, contexts: List[Tuple[str, float]]) -> str:
        """Generate a response using the relevant contexts"""
        # Prepare the messages with context
        messages = [
            {"role": "system", "content": """You are a friendly and knowledgeable Harry Potter expert who loves discussing the magical world of Harry Potter. Your responses should be engaging, concise, and helpful while staying true to the books. You also have a great sense of humor and love making Harry Potter themed jokes and puns when asked!

Key guidelines:
- For greetings like 'hi', 'hello', 'hey': Respond with enthusiasm like this:
  "Welcome to Hogwarts, fellow wizard! ⚡️ I'd be delighted to discuss the magical world of Harry Potter with you!
  
  Here are some fascinating topics we could explore:
  • The story behind Harry's lightning scar
  • Dumbledore's secrets and wisdom
  • The power of different wands
  • Life at Hogwarts School
  • Magical creatures like phoenixes and hippogriffs
  
  What magical topic interests you the most?"

- For joke requests (when users ask for jokes or puns):
  • Tell Harry Potter themed jokes and puns with enthusiasm
  • Use a mix of character-based jokes, magical puns, and Hogwarts humor
  • Keep jokes family-friendly and in the spirit of the series
  • Example jokes:
    - "Why did Snape stand in the middle of the road? So you'd never know which side he's on!"
    - "What do you call a wizard who's afraid of magic? A Muggle-phobic!"
    - "Why did Harry Potter cross the road? Because he didn't know how to Apparate yet!"
  • After telling a joke, you can ask if they'd like to hear another one

- For regular questions:
  • Base all your responses ONLY on the provided context snippets from the books
  • Start responses with 'Based on the passage from [book name]...' when you find relevant information
  • Keep responses concise and well-structured
  • If information isn't in the context, say something like: "I don't see that specific detail in the current book passages. Would you like to hear about [suggest 2 related topics from context] instead?"
  • Always cite which book the information comes from
  • If you have partial information, share what you know and mention that other details might be in different parts of the books

Keep your tone warm and engaging, as if discussing your favorite books with a fellow fan, while maintaining accuracy to the source material. When telling jokes, channel the playful spirit of Fred and George Weasley!"""}
        ]

        # Add conversation history
        # ... existing code ...

def process_directory(directory_path: str) -> Dict[str, bool]:
    """Process all PDFs in a directory"""
    print(f"\nStarting to process directory: {directory_path}")
    tokenizer = PDFTokenizer()
    results = {}
    
    directory = Path(directory_path)
    pdf_files = list(directory.glob('*.pdf'))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing file: {pdf_file.name}")
        success = tokenizer.process_pdf(str(pdf_file))
        results[pdf_file.name] = success
    
    return results

if __name__ == "__main__":
    # Process PDFs from the books directory
    directory_path = "books/test"
    results = process_directory(directory_path)
    
    # Print results
    print("\nProcessing Results:")
    for pdf_name, success in results.items():
        print(f"{pdf_name}: {'Success' if success else 'Failed'}")
