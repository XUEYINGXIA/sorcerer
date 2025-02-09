from openai import OpenAI
from dotenv import load_dotenv
import os
import psycopg2
import numpy as np
from typing import List, Tuple
import readline
import atexit
from pathlib import Path
import sys
import time
import threading

# Load environment variables from .env file
load_dotenv()

# ANSI escape codes for colors and formatting
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def format_bot_response(response: str) -> str:
    """Format the bot's response with decorative elements and color"""
    width = min(os.get_terminal_size().columns - 10, 80)  # Reduced max width to 80 chars with padding
    separator = "─" * width
    # Word wrap the response to fit within the width
    wrapped_response = []
    for line in response.split('\n'):
        while len(line) > width - 4:  # -4 for the wizard emoji and spacing
            split_at = line[:width-4].rfind(' ')
            if split_at == -1:
                split_at = width-4
            wrapped_response.append(line[:split_at])
            line = line[split_at:].lstrip()
        wrapped_response.append(line)
    formatted_response = '\n'.join(wrapped_response)
    
    return f"\n{CYAN}{separator}\n{BOLD}🧙‍♂️ {formatted_response}{RESET}\n{CYAN}{separator}{RESET}\n"

class HistoryManager:
    def __init__(self, history_file: str = ".chat_history"):
        self.history_file = os.path.expanduser(f"~/{history_file}")
        self.max_history = 1000
        
        # Create history file if it doesn't exist
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                pass
        
        # Read existing history
        readline.read_history_file(self.history_file)
        
        # Set maximum history length
        readline.set_history_length(self.max_history)
        
        # Save history on exit
        atexit.register(self.save_history)
    
    def save_history(self):
        readline.write_history_file(self.history_file)
    
    def add_to_history(self, line: str):
        readline.add_history(line)

class LoadingSpinner:
    def __init__(self):
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.stop_spinner = False
        self.spinner_thread = None
        
    def spin(self):
        while not self.stop_spinner:
            for char in self.spinner_chars:
                if self.stop_spinner:
                    break
                sys.stdout.write(f"\r{CYAN}🪄 Thinking... {char}{RESET}")
                sys.stdout.flush()
                time.sleep(0.1)
    
    def start(self):
        self.stop_spinner = False
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def stop(self):
        self.stop_spinner = True
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the spinner line
        sys.stdout.flush()

class BookChatBot:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.db_params = {
            'dbname': 'sorcerer',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': '5432'
        }
        self.conversation_history = []
        self.history_manager = HistoryManager()
        self.loading_spinner = LoadingSpinner()

    def get_query_embedding(self, text: str) -> List[float]:
        """Create an embedding for the user's query"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def get_relevant_context(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve the most relevant context from the database using vector similarity"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            # Convert the query embedding to a PostgreSQL vector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Get the most relevant chunks with their tokens
            cur.execute("""
                WITH RankedChunks AS (
                    SELECT 
                        book_name,
                        chunk_number,
                        tokens,
                        (embedding <=> %s::vector) as distance,
                        ROW_NUMBER() OVER (PARTITION BY book_name ORDER BY embedding <=> %s::vector) as rn
                    FROM book_embeddings
                    WHERE embedding <=> %s::vector < 0.8  -- Filter out low similarity matches
                ),
                FlattenedTokens AS (
                    SELECT 
                        book_name,
                        unnest(tokens) as token,
                        distance
                    FROM RankedChunks
                    WHERE rn <= %s
                )
                SELECT 
                    book_name,
                    array_agg(DISTINCT token) as unique_tokens,
                    min(distance) as min_distance
                FROM FlattenedTokens
                GROUP BY book_name
                ORDER BY min_distance
                LIMIT %s;
            """, (embedding_str, embedding_str, embedding_str, top_k, top_k))
            
            results = cur.fetchall()
            
            if not results:
                return []

            # Process results
            contexts = []
            for book_name, unique_tokens, distance in results:
                # Convert distance to similarity
                similarity = 1 - distance
                
                # Create context from unique tokens
                context = ' '.join(token for token in unique_tokens if token)
                
                # Create rich context with key terms (first 50 tokens)
                key_terms = ' '.join(token for token in unique_tokens[:50] if token)
                rich_context = f"Context: {context}\n\nKey terms: {key_terms}"
                contexts.append((f"From {book_name}:\n{rich_context}", similarity))

            return contexts

        except Exception as e:
            print(f"Database error: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def generate_response(self, query: str, contexts: List[Tuple[str, float]]) -> str:
        """Generate a response using the relevant contexts"""
        # Prepare the messages with context
        messages = [
            {"role": "system", "content": """You are a friendly and knowledgeable Harry Potter expert who loves discussing the magical world of Harry Potter. Your responses should be engaging, concise, and helpful while staying true to the books. You also have a great sense of humor and love making witty Harry Potter-themed jokes!

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

- When users ask for jokes or humor:
  • Create witty puns using magical terms and spells
  • Reference funny moments from the books
  • Use clever wordplay with character names and magical concepts
  • Keep jokes family-friendly and in good taste
  Example responses:
  • "Why did Harry Potter get such good grades in potions? Because he had Hermione-ing skills!"
  • "What do you call a wizard who's afraid of magic? A Muggle-phobic!"
  • "Why did Snape stand in the middle of the road? To make a cross-roads curse!"

- Base all your responses ONLY on the provided context snippets from the books
- Start responses with 'Based on the passage from [book name]...' when you find relevant information
- Keep responses concise and well-structured
- If information isn't in the context, say something like: "I don't see that specific detail in the current book passages. Would you like to hear about [suggest 2 related topics from context] instead?"
- Always cite which book the information comes from
- If you have partial information, share what you know and mention that other details might be in different parts of the books

Keep your tone warm and engaging, as if discussing your favorite books with a fellow fan, while maintaining accuracy to the source material."""}
        ]

        # Add conversation history
        messages.extend(self.conversation_history[-4:])  # Only keep last 2 exchanges for context

        # Add context
        context_text = "\n\n---\n\n".join([f"Context (similarity: {sim:.2f}):\n{ctx}" for ctx, sim in contexts])
        messages.append({
            "role": "system",
            "content": f"Here are the relevant passages from the Harry Potter books:\n\n{context_text}"
        })

        # Add user query
        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating response: {e}"

    def process_query(self, query: str) -> str:
        """Process a query with loading indicator"""
        self.loading_spinner.start()
        try:
            # Get query embedding
            query_embedding = self.get_query_embedding(query)
            
            # Get relevant context using vector similarity
            contexts = self.get_relevant_context(query_embedding)
            
            if not contexts:
                return "I apologize, but I couldn't find any relevant passages from the books for your question.\nPerhaps try:\n  • Rephrasing your question\n  • Asking about a specific character or event\n  • Using different keywords"

            # Generate response
            response = self.generate_response(query, contexts)
            return response
        finally:
            self.loading_spinner.stop()

    def chat(self):
        """Main chat loop"""
        print("\n🧙‍♂️ Welcome to the Harry Potter Book Chat! ✨")
        print("\nI'm your magical companion for exploring the Harry Potter series. I can answer questions")
        print("based on the actual text from the books, ensuring accurate and authentic responses.")
        print("\n📚 Here are some example questions you might want to try:")
        print("  • What happens during Harry's first meeting with Dumbledore?")
        print("  • Can you describe the moment when Harry first discovers he's a wizard?")
        print("  • What are the Deathly Hallows and their significance?")
        print("  • Tell me about Severus Snape's relationship with Lily Potter")
        print("  • What happens during the Battle of Hogwarts?")
        print("\n💡 Tips:")
        print("  • Be specific in your questions for more accurate answers")
        print("  • Use up/down arrow keys to navigate through your previous questions")
        print("  • Type 'quit' or 'exit' to end the chat")
        print("  • If you're not getting the answer you want, try rephrasing your question")
        print("\nLet's begin our magical journey through the books! What would you like to know?\n")
        
        # Common greetings
        greetings = {'hi', 'hello', 'hey', 'greetings', 'hola', 'howdy', 'hi there', 'hello there'}
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query:  # Only add non-empty queries to history
                    self.history_manager.add_to_history(query)
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("\n✨ Thank you for chatting! Mischief managed! ✨")
                    break
                    
                if not query:
                    continue

                # Check if it's a greeting
                if query.lower() in greetings:
                    greeting_response = """Welcome to Hogwarts, fellow wizard! ⚡️ I'd be delighted to discuss the magical world of Harry Potter with you!

Here are some fascinating topics we could explore:
• The story behind Harry's lightning scar and his connection to Voldemort
• Dumbledore's secrets and his complex relationship with Harry
• The power of different wands and their unique properties
• Life at Hogwarts School - from classes to Quidditch matches
• Magical creatures like phoenixes, hippogriffs, and dragons
• The mysteries of the Deathly Hallows
• The deep bonds between Harry, Ron, and Hermione
• The complex story of Severus Snape

What magical topic interests you the most? Feel free to ask about any of these or something else that sparks your curiosity!"""
                    print(format_bot_response(greeting_response))
                    print()
                    continue

                # Process the query with loading indicator
                response = self.process_query(query)
                print(format_bot_response(response))
                print()  # Add extra space before next prompt

                # Update conversation history (keep last 6 messages)
                self.conversation_history.extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response}
                ])
                if len(self.conversation_history) > 6:
                    self.conversation_history = self.conversation_history[-6:]

            except KeyboardInterrupt:
                print("\n✨ Thank you for chatting! Mischief managed! ✨")
                break
            except EOFError:
                print("\n✨ Thank you for chatting! Mischief managed! ✨")
                break

if __name__ == "__main__":
    try:
        chat_bot = BookChatBot()
        chat_bot.chat()
    except KeyboardInterrupt:
        print("\n✨ Goodbye! Mischief managed! ✨")
    finally:
        sys.exit(0)
