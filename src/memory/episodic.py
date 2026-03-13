import sqlite3
import json
from datetime import datetime
import chromadb

class EpisodicMemory:
    def __init__(self, db_path="data/episodic.db", chroma_path="data/chroma_db"):
        self.db_path = db_path
        # Setup SQLite for exact session logging
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                response TEXT,
                state_snapshot TEXT
            )
        ''')
        self.conn.commit()

        # Setup ChromaDB for semantic search of past queries
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="episodic_history")

    def log_interaction(self, query: str, response: str, state_snapshot: dict = None):
        """Logs a query and response to both SQLite and ChromaDB."""
        timestamp = datetime.now().isoformat()
        state_str = json.dumps(state_snapshot) if state_snapshot else "{}"
        
        # Log to SQLite
        self.cursor.execute(
            "INSERT INTO interactions (timestamp, query, response, state_snapshot) VALUES (?, ?, ?, ?)",
            (timestamp, query, response, state_str)
        )
        self.conn.commit()
        
        interaction_id = str(self.cursor.lastrowid)

        # Log to Chroma for retrieval
        document = f"User Query: {query}\nAgent Response: {response}\nState: {state_str}"
        self.collection.add(
            documents=[document],
            metadatas=[{"timestamp": timestamp, "type": "interaction"}],
            ids=[interaction_id]
        )

    def retrieve_past_context(self, current_query: str, n_results: int = 3) -> str:
        """Finds previous interactions similar to the current query."""
        if self.collection.count() == 0:
            return "No previous episodic memory found."
            
        # Don't query for more results than standard if we don't have enough
        n = min(n_results, self.collection.count())
        
        results = self.collection.query(
            query_texts=[current_query],
            n_results=n
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant past interactions found."
            
        context = "Relevant Past Interactions (Episodic Memory):\n"
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            context += f"--- [{metadata['timestamp']}] ---\n{doc}\n\n"
            
        return context

    def get_recent_history(self, limit: int = 5) -> list:
        """Fetch the most recent N interactions for standard memory."""
        self.cursor.execute(
            "SELECT query, response FROM interactions ORDER BY id DESC LIMIT ?", 
            (limit,)
        )
        return [{"query": row[0], "response": row[1]} for row in reversed(self.cursor.fetchall())]
