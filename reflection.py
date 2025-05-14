import os
import json
import chromadb
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.logger import setup_logger
from semantic_router.semantic import SemanticSearch
from embedding_model.embedding import EmbeddingModel

logger = setup_logger()

class Reflection:
    def __init__(
        self,
        db_path: str = "chroma_db",
        embedding_model: Optional[EmbeddingModel] = None,
        semantic_search: Optional[SemanticSearch] = None
    ):
        self.db_path = db_path
        self.embedding_model = embedding_model or EmbeddingModel()
        self.semantic_search = semantic_search or SemanticSearch()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize conversation tracking
        self.current_session = None
        self.conversation_history: List[Dict[str, Any]] = []
        
        logger.info("Reflection component initialized")

    def start_session(self, session_id: Optional[str] = None):
        """Start a new conversation session."""
        self.current_session = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history = []
        logger.info(f"Started new session: {self.current_session}")

    def add_to_history(
        self,
        message: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message-response pair to conversation history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response,
            "metadata": metadata or {}
        }
        self.conversation_history.append(entry)
        
        # Store in ChromaDB
        self.collection.add(
            documents=[json.dumps(entry)],
            metadatas=[{"session_id": self.current_session, **(metadata or {})}],
            ids=[f"{self.current_session}_{len(self.conversation_history)}"]
        )
        
        logger.debug(f"Added entry to history: {entry}")

    def analyze_context(self, message: str) -> Dict[str, Any]:
        """Analyze the context of the current message."""
        try:
            # Get semantic search results
            search_results = self.semantic_search.query(message)
            
            # Extract key information
            context = {
                "relevant_history": search_results,
                "session_length": len(self.conversation_history),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add conversation flow analysis
            if len(self.conversation_history) > 0:
                last_entry = self.conversation_history[-1]
                context["last_topic"] = last_entry.get("metadata", {}).get("topic")
                context["time_since_last"] = (
                    datetime.now() - datetime.fromisoformat(last_entry["timestamp"])
                ).total_seconds()
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing context: {e}")
            return {}

    def rank_responses(
        self,
        responses: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank potential responses based on context and history."""
        try:
            ranked = []
            for resp in responses:
                score = 0.0
                
                # Base score from semantic similarity
                if "score" in resp:
                    score += resp["score"] * 0.6
                
                # Context relevance
                if "context_relevance" in resp:
                    score += resp["context_relevance"] * 0.2
                
                # History consistency
                if "history_consistency" in resp:
                    score += resp["history_consistency"] * 0.2
                
                ranked.append({
                    **resp,
                    "final_score": score
                })
            
            # Sort by final score
            ranked.sort(key=lambda x: x["final_score"], reverse=True)
            return ranked
            
        except Exception as e:
            logger.error(f"Error ranking responses: {e}")
            return responses

    def generate_response(
        self,
        message: str,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate the best response based on context and ranking."""
        try:
            # Analyze context
            context = self.analyze_context(message)
            
            # Rank responses
            ranked_responses = self.rank_responses(responses, context)
            
            if not ranked_responses:
                return {
                    "response": "I apologize, but I couldn't find a suitable response.",
                    "confidence": 0.0,
                    "context": context
                }
            
            # Get best response
            best_response = ranked_responses[0]
            
            # Add to history
            self.add_to_history(
                message=message,
                response=best_response["text"],
                metadata={
                    "confidence": best_response["final_score"],
                    "context": context
                }
            )
            
            return {
                "response": best_response["text"],
                "confidence": best_response["final_score"],
                "context": context,
                "alternatives": ranked_responses[1:3]  # Include top alternatives
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I encountered an error while processing your request.",
                "confidence": 0.0,
                "context": {}
            }

    def get_session_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session."""
        try:
            session = session_id or self.current_session
            if not session:
                return []
            
            results = self.collection.get(
                where={"session_id": session},
                limit=limit
            )
            
            return [
                json.loads(doc)
                for doc in results["documents"]
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving session history: {e}")
            return []

    def export_conversation(
        self,
        session_id: Optional[str] = None,
        format: str = "json"
    ) -> str:
        """Export conversation history in specified format."""
        try:
            history = self.get_session_history(session_id)
            
            if format.lower() == "json":
                return json.dumps(history, indent=2)
            elif format.lower() == "text":
                return "\n".join(
                    f"User: {entry['message']}\nBot: {entry['response']}\n"
                    for entry in history
                )
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            return ""

    def chat(self, session_id: str, message: str) -> str:
        """Process a chat message and return a response."""
        try:
            # Start or continue session
            if not self.current_session:
                self.start_session(session_id)
            
            # Get semantic search results
            search_results = self.semantic_search.query(message)
            
            # Generate response
            response = self.generate_response(message, search_results)
            
            return response["response"]
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I encountered an error processing your message." 