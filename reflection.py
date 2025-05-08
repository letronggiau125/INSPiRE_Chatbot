import chromadb
from chromadb.config import Settings
from typing import Optional, List, Dict, Any, Tuple, Set, Union
import numpy as np
from utils.logger import setup_logger
from config import Config
from embedding_model.embedding import embedding_model
import uuid
from datetime import datetime, timedelta
import json
import re
from functools import lru_cache
from collections import defaultdict, Counter
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = setup_logger()

class ReflectionError(Exception):
    """Custom exception for Reflection-specific errors."""
    pass

class Reflection:
    """Enhanced reflection model for context-aware responses with improved performance."""
    
    def __init__(self, db_path: str):
        """Initialize ChromaDB and load configurations."""
        try:
            # Initialize thread pool for parallel processing
            self.thread_pool = ThreadPoolExecutor(
                max_workers=Config.CHAT_SETTINGS.get('max_workers', 4)
            )
            
            # Initialize ChromaDB with persistence and optimized settings
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path,
                anonymized_telemetry=False
            ))
            
            # Create or get collections with optimized settings
            self.faq_collection = self.client.get_or_create_collection(
                name="faq_collection",
                metadata={"description": "FAQ embeddings collection"},
                embedding_function=None  # We'll handle embeddings ourselves
            )
            
            self.context_collection = self.client.get_or_create_collection(
                name="context_collection",
                metadata={"description": "Conversation context collection"},
                embedding_function=None
            )
            
            # Initialize conversation management with thread safety
            self._conversation_lock = threading.Lock()
            self.conversation_history = {}
            self.session_metadata = defaultdict(dict)
            self.active_sessions = set()
            
            # Load configuration with defaults
            self._load_config()
            
            # Initialize caches and patterns
            self._init_caches()
            
            logger.info("Reflection model initialized successfully with optimized settings")
            
        except Exception as e:
            logger.error(f"Error initializing Reflection model: {e}", exc_info=True)
            raise ReflectionError(f"Failed to initialize Reflection model: {str(e)}")
    
    def _load_config(self):
        """Load configuration settings with defaults."""
        self.max_history = Config.CHAT_SETTINGS.get('max_history', 5)
        self.context_window = Config.CHAT_SETTINGS.get('context_window', 3600)
        self.follow_up_threshold = Config.CHAT_SETTINGS.get('follow_up_threshold', 0.7)
        self.max_references = Config.CHAT_SETTINGS.get('max_references', 5)
        self.cache_ttl = Config.CHAT_SETTINGS.get('cache_ttl', 3600)
        self.batch_size = Config.CHAT_SETTINGS.get('batch_size', 100)
        self.min_confidence = Config.CHAT_SETTINGS.get('min_confidence', 0.6)
        self.language_weights = {
            'vi': 1.2,  # Boost Vietnamese responses
            'en': 1.0,
            'auto': 0.9
        }
    
    def _init_caches(self):
        """Initialize LRU caches and compile patterns."""
        # Initialize caches with TTL
        self._embedding_cache = lru_cache(maxsize=1000)(self._compute_embedding)
        self._similarity_cache = lru_cache(maxsize=1000)(self._compute_similarity)
        self._language_cache = lru_cache(maxsize=500)(self._detect_language)
        
        # Compile patterns
        self._reference_patterns = self._compile_reference_patterns()
        self._entity_patterns = self._compile_entity_patterns()
        
        # Initialize response templates
        self._init_response_templates()
    
    def _init_response_templates(self):
        """Initialize multilingual response templates."""
        self.response_templates = {
            'vi': {
                'follow_up_high': "Như tôi đã đề cập, {response}",
                'follow_up_medium': "Để bổ sung thêm: {response}",
                'clarification': "Xin lỗi, bạn có thể làm rõ về {topic} được không?",
                'unknown': "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại được không?"
            },
            'en': {
                'follow_up_high': "As I mentioned, {response}",
                'follow_up_medium': "To add to my previous response: {response}",
                'clarification': "Could you please clarify about {topic}?",
                'unknown': "I'm sorry, I don't understand your question. Could you rephrase it?"
            }
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text with error handling."""
        try:
            return detect(text)
        except LangDetectException:
            return 'auto'
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'auto'
    
    def _compile_entity_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for entity extraction."""
        return {
            'dates': re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
            'times': re.compile(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}(am|pm|AM|PM)\b'),
            'numbers': re.compile(r'\b\d+([,.]\d+)?\b'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        }
    
    async def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Process user message asynchronously with enhanced context handling."""
        try:
            # Detect language
            lang = self._detect_language(user_message)
            
            # Analyze context and get embedding concurrently
            context_future = self.thread_pool.submit(self._analyze_context, session_id, user_message)
            embedding_future = self.thread_pool.submit(self._compute_embedding, user_message)
            
            # Wait for both tasks to complete
            context = context_future.result()
            message_embedding = embedding_future.result()
            
            # Query similar documents with language boost
            results = await self._query_documents(message_embedding, lang)
            
            if not results["documents"]:
                return {
                    'response': self.response_templates[lang]['unknown'],
                    'confidence': 0.0,
                    'context': context
                }
            
            # Prepare and rank responses
            responses = self._prepare_responses(results, lang)
            ranked_responses = self._rank_responses(responses, context)
            
            if not ranked_responses:
                return {
                    'response': self.response_templates[lang]['unknown'],
                    'confidence': 0.0,
                    'context': context
                }
            
            best_match = ranked_responses[0]
            
            # Generate appropriate response
            response = self._generate_response(best_match, context, lang)
            
            # Update conversation history
            await self._async_update_history(
                session_id, 
                user_message, 
                response,
                lang=lang,
                category=best_match["category"]
            )
            
            return {
                'response': response,
                'confidence': best_match["rank_score"],
                'context': context,
                'metadata': {
                    'language': lang,
                    'category': best_match["category"],
                    'scoring': best_match["scoring_details"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                'response': self.response_templates.get(lang, self.response_templates['en'])['unknown'],
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def _query_documents(self, embedding: np.ndarray, lang: str) -> Dict[str, Any]:
        """Query documents with language-aware boosting."""
        try:
            results = self.faq_collection.query(
                query_embeddings=[embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            # Apply language boost
            if results["documents"]:
                for i in range(len(results["distances"][0])):
                    doc_lang = results["metadatas"][0][i].get('language', 'auto')
                    boost = self.language_weights.get(doc_lang, 1.0)
                    if doc_lang == lang:
                        results["distances"][0][i] *= (1 / boost)  # Reduce distance for matching language
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def _prepare_responses(self, results: Dict[str, Any], lang: str) -> List[Dict[str, Any]]:
        """Prepare responses with enhanced metadata."""
        responses = []
        for i in range(len(results["documents"][0])):
            response_lang = self._detect_language(results["documents"][0][i])
            responses.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i],
                "category": results["metadatas"][0][i]["category"],
                "language": response_lang,
                "language_match": response_lang == lang
            })
        return responses
    
    def _generate_response(self, match: Dict[str, Any], context: Dict[str, Any], lang: str) -> str:
        """Generate appropriate response based on context and language."""
        try:
            templates = self.response_templates.get(lang, self.response_templates['en'])
            
            if context["follow_up"]:
                if match["similarity"] > 0.9:
                    return templates['follow_up_high'].format(response=match["text"])
                elif match["similarity"] > 0.7:
                    return templates['follow_up_medium'].format(response=match["text"])
            
            # Check if clarification is needed
            if match["similarity"] < self.min_confidence and context["references"]:
                topic = context["references"][0] if context["references"] else "this topic"
                return templates['clarification'].format(topic=topic)
            
            return match["text"]
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return match["text"]
    
    async def _async_update_history(self, session_id: str, user_message: str, 
                                  bot_response: str, lang: str = 'auto', 
                                  category: str = None) -> None:
        """Asynchronously update conversation history."""
        try:
            async with asyncio.Lock():
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                    self.session_metadata[session_id] = {
                        'start_time': datetime.now(),
                        'message_count': 0,
                        'categories': defaultdict(int),
                        'languages': Counter()
                    }
                    self.active_sessions.add(session_id)
                
                history = self.conversation_history[session_id]
                metadata = self.session_metadata[session_id]
                
                # Add new exchange with enhanced metadata
                exchange = {
                    "user": user_message,
                    "bot": bot_response,
                    "timestamp": datetime.now().isoformat(),
                    "category": category,
                    "language": lang,
                    "exchange_id": str(uuid.uuid4())
                }
                
                history.append(exchange)
                
                # Update session metadata
                metadata['message_count'] += 1
                metadata['last_activity'] = datetime.now()
                metadata['languages'][lang] += 1
                if category:
                    metadata['categories'][category] += 1
                
                # Cleanup old entries
                await self._async_cleanup_history(session_id)
                
        except Exception as e:
            logger.error(f"Error updating conversation history: {e}", exc_info=True)
    
    async def _async_cleanup_history(self, session_id: str) -> None:
        """Asynchronously clean up conversation history."""
        try:
            history = self.conversation_history[session_id]
            
            # Cleanup old entries
            cutoff_time = datetime.now() - timedelta(seconds=self.context_window)
            history = [h for h in history 
                      if datetime.fromisoformat(h["timestamp"]) > cutoff_time]
            
            # Keep only recent history
            if len(history) > self.max_history:
                history = history[-self.max_history:]
            
            self.conversation_history[session_id] = history
            
        except Exception as e:
            logger.error(f"Error cleaning up history: {e}")
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a session."""
        try:
            with self._conversation_lock:
                if session_id not in self.session_metadata:
                    return {}
                
                metadata = self.session_metadata[session_id]
                history = self.conversation_history.get(session_id, [])
                
                # Calculate metrics
                total_messages = metadata['message_count']
                category_distribution = {
                    cat: count/total_messages 
                    for cat, count in metadata['categories'].items()
                } if total_messages > 0 else {}
                
                language_distribution = {
                    lang: count/total_messages 
                    for lang, count in metadata['languages'].items()
                } if total_messages > 0 else {}
                
                # Calculate response times
                response_times = self._calculate_response_times(history)
                
                return {
                    'session_info': {
                        'start_time': metadata['start_time'].isoformat(),
                        'last_activity': metadata['last_activity'].isoformat(),
                        'duration': (metadata['last_activity'] - metadata['start_time']).total_seconds()
                    },
                    'interaction_stats': {
                        'total_messages': total_messages,
                        'category_distribution': category_distribution,
                        'language_distribution': language_distribution,
                        'avg_response_time': response_times.get('average'),
                        'response_time_percentiles': {
                            'p50': response_times.get('p50'),
                            'p90': response_times.get('p90'),
                            'p95': response_times.get('p95')
                        }
                    },
                    'quality_metrics': {
                        'language_match_rate': self._calculate_language_match_rate(history),
                        'follow_up_rate': self._calculate_follow_up_rate(history),
                        'category_consistency': self._calculate_category_consistency(history)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting session analytics: {e}")
            return {}
    
    def _calculate_response_times(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate response time statistics."""
        try:
            if len(history) < 2:
                return {}
            
            response_times = []
            for i in range(1, len(history)):
                current = datetime.fromisoformat(history[i]["timestamp"])
                previous = datetime.fromisoformat(history[i-1]["timestamp"])
                response_times.append((current - previous).total_seconds())
            
            response_times.sort()
            n = len(response_times)
            
            return {
                'average': sum(response_times) / n,
                'p50': response_times[n//2],
                'p90': response_times[int(n*0.9)],
                'p95': response_times[int(n*0.95)],
                'min': response_times[0],
                'max': response_times[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating response times: {e}")
            return {}
    
    def _calculate_language_match_rate(self, history: List[Dict[str, Any]]) -> float:
        """Calculate rate of language matches between user and response."""
        try:
            if not history:
                return 0.0
            
            matches = sum(1 for h in history 
                         if h.get('language') == self._detect_language(h['user']))
            return matches / len(history)
            
        except Exception as e:
            logger.error(f"Error calculating language match rate: {e}")
            return 0.0
    
    def _calculate_follow_up_rate(self, history: List[Dict[str, Any]]) -> float:
        """Calculate rate of follow-up questions in conversation."""
        try:
            if len(history) < 2:
                return 0.0
            
            follow_ups = 0
            for i in range(1, len(history)):
                similarity = self._similarity_cache(
                    history[i]['user'],
                    history[i-1]['user']
                )
                if similarity > self.follow_up_threshold:
                    follow_ups += 1
            
            return follow_ups / (len(history) - 1)
            
        except Exception as e:
            logger.error(f"Error calculating follow-up rate: {e}")
            return 0.0
    
    def _calculate_category_consistency(self, history: List[Dict[str, Any]]) -> float:
        """Calculate consistency of categories in consecutive exchanges."""
        try:
            if len(history) < 2:
                return 1.0
            
            consistent = 0
            for i in range(1, len(history)):
                if (history[i].get('category') == history[i-1].get('category') and
                    history[i].get('category') is not None):
                    consistent += 1
            
            return consistent / (len(history) - 1)
            
        except Exception as e:
            logger.error(f"Error calculating category consistency: {e}")
            return 0.0
    
    def cleanup_inactive_sessions(self, max_age_hours: int = 24) -> None:
        """Clean up inactive sessions and archive their data."""
        try:
            with self._conversation_lock:
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                
                for session_id in list(self.session_metadata.keys()):
                    last_activity = self.session_metadata[session_id]['last_activity']
                    if last_activity < cutoff_time:
                        # Get analytics before cleanup
                        analytics = self.get_session_analytics(session_id)
                        
                        # Archive session data if needed
                        self._archive_session_data(session_id, analytics)
                        
                        # Clean up session data
                        del self.conversation_history[session_id]
                        del self.session_metadata[session_id]
                        
                        logger.info(f"Cleaned up inactive session: {session_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up inactive sessions: {e}")
    
    def _archive_session_data(self, session_id: str, analytics: Dict[str, Any]) -> None:
        """Archive session data for future analysis."""
        try:
            # Export conversation history
            history = self.export_conversation_history(session_id)
            
            # Combine history and analytics
            archive_data = {
                'session_id': session_id,
                'history': history,
                'analytics': analytics,
                'archive_time': datetime.now().isoformat()
            }
            
            # TODO: Implement actual archiving logic (e.g., save to database or file)
            logger.info(f"Archived data for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error archiving session data: {e}")
    
    def get_active_sessions_summary(self) -> Dict[str, Any]:
        """Get summary of all active sessions."""
        try:
            with self._conversation_lock:
                active_sessions = {}
                current_time = datetime.now()
                
                for session_id, metadata in self.session_metadata.items():
                    last_activity = metadata['last_activity']
                    if (current_time - last_activity).total_seconds() < self.context_window:
                        active_sessions[session_id] = {
                            'duration': (current_time - metadata['start_time']).total_seconds(),
                            'message_count': metadata['message_count'],
                            'last_activity': last_activity.isoformat(),
                            'primary_language': max(
                                metadata['languages'].items(),
                                key=lambda x: x[1]
                            )[0] if metadata['languages'] else 'unknown'
                        }
                
                return {
                    'total_active_sessions': len(active_sessions),
                    'sessions': active_sessions
                }
                
        except Exception as e:
            logger.error(f"Error getting active sessions summary: {e}")
            return {'total_active_sessions': 0, 'sessions': {}}
    
    def add_documents(self, documents: List[Dict[str, Any]], update_existing: bool = False) -> bool:
        """Add or update documents with improved error handling and validation."""
        try:
            if not documents:
                logger.warning("No documents provided for addition")
                return False
            
            # Validate documents
            valid_documents = []
            for doc in documents:
                if not isinstance(doc, dict) or 'text' not in doc:
                    logger.warning(f"Invalid document format: {doc}")
                    continue
                
                doc_id = doc.get('id', str(uuid.uuid4()))
                valid_documents.append({
                    **doc,
                    'id': doc_id,
                    'timestamp': doc.get('timestamp', datetime.now().isoformat()),
                    'version': doc.get('version', '1.0')
                })
            
            if not valid_documents:
                logger.error("No valid documents to add")
                return False
            
            # Prepare batch processing
            batch_size = 100
            for i in range(0, len(valid_documents), batch_size):
                batch = valid_documents[i:i + batch_size]
                
                # Prepare batch data
                ids = [doc['id'] for doc in batch]
                texts = [doc['text'] for doc in batch]
                metadatas = [{
                    'category': doc.get('category', 'unknown'),
                    'timestamp': doc.get('timestamp'),
                    'version': doc.get('version'),
                    'source': doc.get('source', 'user'),
                    'language': doc.get('language', 'auto')
                } for doc in batch]
                
                # Generate embeddings in parallel for the batch
                embeddings = embedding_model.get_batch_embeddings(texts)
                
                if update_existing:
                    self.faq_collection.delete(ids=ids)
                
                # Add batch to collection
                self.faq_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts
                )
                
                logger.info(f"Added batch of {len(batch)} documents to collection")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in add_documents: {e}", exc_info=True)
            return False
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the collection."""
        try:
            self.faq_collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from collection")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
    
    def _analyze_context(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Analyze conversation context for better response selection."""
        try:
            history = self.conversation_history.get(session_id, [])
            if not history:
                return {"type": "new", "references": [], "follow_up": False}
            
            # Get last conversation
            last_exchange = history[-1]
            last_response = last_exchange["bot"]
            last_query = last_exchange["user"]
            
            # Check if current message is a follow-up
            follow_up_similarity = embedding_model.compute_similarity(last_query, user_message)
            is_follow_up = follow_up_similarity > self.follow_up_threshold
            
            # Find referenced entities or topics
            references = []
            for exchange in reversed(history[-3:]):  # Look at last 3 exchanges
                # Extract potential references (e.g., "it", "that", "those")
                refs = self._extract_references(exchange["user"], exchange["bot"])
                references.extend(refs)
            
            return {
                "type": "follow_up" if is_follow_up else "new",
                "references": references,
                "follow_up": is_follow_up,
                "last_response": last_response,
                "last_query": last_query,
                "similarity": follow_up_similarity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing context: {e}")
            return {"type": "new", "references": [], "follow_up": False}
    
    def _extract_references(self, user_msg: str, bot_msg: str) -> List[str]:
        """Extract referenced entities with improved pattern matching."""
        try:
            references = set()
            
            # Extract references using compiled patterns
            for pattern_type, pattern in self._reference_patterns.items():
                # Find references in both messages
                user_refs = pattern.findall(user_msg)
                bot_refs = pattern.findall(bot_msg)
                
                # Add found references
                references.update(user_refs)
                references.update(bot_refs)
            
            # Extract noun phrases (simple approach)
            noun_phrases = self._extract_noun_phrases(user_msg)
            references.update(noun_phrases)
            
            # Limit number of references
            return list(references)[:self.max_references]
            
        except Exception as e:
            logger.error(f"Error extracting references: {e}")
            return []
    
    def _extract_noun_phrases(self, text: str) -> Set[str]:
        """Extract potential noun phrases from text."""
        noun_phrases = set()
        # Simple pattern for noun phrases
        np_pattern = re.compile(r'\b(?:[A-Z][a-z]+\s*)+\b|\b(?:[A-Z][A-Z]+)\b')
        matches = np_pattern.finditer(text)
        
        for match in matches:
            noun_phrases.add(match.group())
        
        return noun_phrases
    
    def _rank_responses(self, responses: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced response ranking with more sophisticated scoring."""
        try:
            ranked_responses = []
            for resp in responses:
                score = 0.0
                
                # Base similarity score (40%)
                score += resp["similarity"] * 0.4
                
                # Context relevance (30%)
                if context["follow_up"]:
                    # Use cached similarity computation
                    context_similarity = self._similarity_cache(
                        resp["text"], 
                        context["last_response"]
                    )
                    score += context_similarity * 0.3
                
                # Category matching (15%)
                if "category" in resp and resp["category"] == context.get("last_category"):
                    score += 0.15
                
                # Reference matching (10%)
                if context["references"]:
                    ref_score = self._calculate_reference_score(resp["text"], context["references"])
                    score += ref_score * 0.1
                
                # Freshness factor (5%)
                if "timestamp" in resp.get("metadata", {}):
                    age = datetime.now() - datetime.fromisoformat(resp["metadata"]["timestamp"])
                    freshness = max(0, 1 - (age.days / 365))
                    score += freshness * 0.05
                
                ranked_responses.append({
                    **resp,
                    "rank_score": score,
                    "scoring_details": {
                        "base_similarity": resp["similarity"],
                        "context_relevance": context_similarity if context["follow_up"] else 0,
                        "category_match": 1 if resp["category"] == context.get("last_category") else 0,
                        "reference_score": ref_score if context["references"] else 0,
                        "freshness": freshness if "timestamp" in resp.get("metadata", {}) else 0
                    }
                })
            
            # Sort by rank score
            ranked_responses.sort(key=lambda x: x["rank_score"], reverse=True)
            return ranked_responses
            
        except Exception as e:
            logger.error(f"Error ranking responses: {e}")
            return responses
    
    def _calculate_reference_score(self, text: str, references: List[str]) -> float:
        """Calculate how well a response matches the referenced entities."""
        try:
            if not references:
                return 0.0
            
            text_lower = text.lower()
            ref_scores = []
            
            for ref in references:
                ref_lower = ref.lower()
                if ref_lower in text_lower:
                    ref_scores.append(1.0)
                else:
                    # Use cached similarity for partial matches
                    similarity = self._similarity_cache(ref, text)
                    ref_scores.append(similarity)
            
            return sum(ref_scores) / len(references) if ref_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating reference score: {e}")
            return 0.0
    
    def chat(self, session_id: str, user_message: str) -> str:
        """Generate a response based on user message and conversation context."""
        try:
            # Detect language
            lang = self._detect_language(user_message)
            
            # Analyze context
            context = self._analyze_context(session_id, user_message)
            
            # Generate message embedding
            message_embedding = embedding_model.get_embedding(user_message)[0]
            
            # Query similar documents
            results = self.faq_collection.query(
                query_embeddings=[message_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"]:
                return self.response_templates[lang]['unknown']
            
            # Apply language boost to results
            for i in range(len(results["distances"][0])):
                doc_lang = self._detect_language(results["documents"][0][i])
                boost = self.language_weights.get(doc_lang, 1.0)
                if doc_lang == lang:
                    results["distances"][0][i] *= (1 / boost)  # Reduce distance for matching language
            
            # Prepare responses for ranking
            responses = []
            for i in range(len(results["documents"][0])):
                responses.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                    "category": results["metadatas"][0][i]["category"],
                    "language": self._detect_language(results["documents"][0][i])
                })
            
            # Rank responses
            ranked_responses = self._rank_responses(responses, context)
            
            if not ranked_responses:
                return self.response_templates[lang]['unknown']
            
            best_match = ranked_responses[0]
            
            # Generate appropriate response based on language and context
            templates = self.response_templates.get(lang, self.response_templates['en'])
            
            if context["follow_up"]:
                if best_match["similarity"] > 0.9:
                    response = templates['follow_up_high'].format(response=best_match["text"])
                elif best_match["similarity"] > 0.7:
                    response = templates['follow_up_medium'].format(response=best_match["text"])
                else:
                    response = best_match["text"]
            else:
                response = best_match["text"]
            
            # Update conversation history with language info
            self._update_conversation_history(
                session_id, 
                user_message, 
                response,
                category=best_match["category"],
                language=lang
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            return self.response_templates.get(lang, self.response_templates['en'])['unknown']
    
    def _update_conversation_history(self, session_id: str, user_message: str, 
                                   bot_response: str, category: str = None,
                                   language: str = 'auto') -> None:
        """Update conversation history with enhanced metadata."""
        try:
            with self._conversation_lock:
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                    self.session_metadata[session_id] = {
                        'start_time': datetime.now(),
                        'message_count': 0,
                        'categories': defaultdict(int),
                        'languages': defaultdict(int)
                    }
                
                history = self.conversation_history[session_id]
                metadata = self.session_metadata[session_id]
                
                # Add new exchange with enhanced metadata
                exchange = {
                    "user": user_message,
                    "bot": bot_response,
                    "timestamp": datetime.now().isoformat(),
                    "category": category,
                    "language": language,
                    "exchange_id": str(uuid.uuid4())
                }
                
                history.append(exchange)
                
                # Update session metadata
                metadata['message_count'] += 1
                metadata['last_activity'] = datetime.now()
                metadata['languages'][language] += 1
                if category:
                    metadata['categories'][category] += 1
                
                # Cleanup old entries
                cutoff_time = datetime.now() - timedelta(seconds=self.context_window)
                history = [h for h in history 
                         if datetime.fromisoformat(h["timestamp"]) > cutoff_time]
                
                # Keep only recent history
                if len(history) > self.max_history:
                    history = history[-self.max_history:]
                
                self.conversation_history[session_id] = history
                
                # Log session statistics
                logger.info(f"Session {session_id} stats: "
                          f"messages={metadata['message_count']}, "
                          f"categories={dict(metadata['categories'])}, "
                          f"languages={dict(metadata['languages'])}")
                
        except Exception as e:
            logger.error(f"Error updating conversation history: {e}", exc_info=True)
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a session."""
        try:
            with self._conversation_lock:
                if session_id not in self.session_metadata:
                    return {}
                
                metadata = self.session_metadata[session_id]
                history = self.conversation_history.get(session_id, [])
                
                return {
                    'start_time': metadata['start_time'].isoformat(),
                    'last_activity': metadata['last_activity'].isoformat(),
                    'message_count': metadata['message_count'],
                    'categories': dict(metadata['categories']),
                    'avg_response_time': self._calculate_avg_response_time(history),
                    'session_duration': (metadata['last_activity'] - metadata['start_time']).total_seconds()
                }
                
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}
    
    def _calculate_avg_response_time(self, history: List[Dict[str, Any]]) -> float:
        """Calculate average response time for a conversation history."""
        try:
            if len(history) < 2:
                return 0.0
            
            response_times = []
            for i in range(1, len(history)):
                current = datetime.fromisoformat(history[i]["timestamp"])
                previous = datetime.fromisoformat(history[i-1]["timestamp"])
                response_times.append((current - previous).total_seconds())
            
            return sum(response_times) / len(response_times)
            
        except Exception as e:
            logger.error(f"Error calculating average response time: {e}")
            return 0.0
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """Clean up old session data to free memory."""
        try:
            with self._conversation_lock:
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                
                for session_id in list(self.session_metadata.keys()):
                    last_activity = self.session_metadata[session_id]['last_activity']
                    if last_activity < cutoff_time:
                        del self.conversation_history[session_id]
                        del self.session_metadata[session_id]
                        logger.info(f"Cleaned up old session: {session_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
    
    def get_similar_questions(self, question: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar questions with enhanced metadata."""
        try:
            # Generate question embedding
            question_embedding = embedding_model.get_embedding(question)[0]
            
            # Query similar documents
            results = self.faq_collection.query(
                query_embeddings=[question_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results with additional metadata
            similar_questions = []
            for i in range(len(results["documents"][0])):
                similar_questions.append({
                    "text": results["documents"][0][i],
                    "category": results["metadatas"][0][i]["category"],
                    "similarity": 1 - results["distances"][0][i],
                    "timestamp": results["metadatas"][0][i]["timestamp"],
                    "version": results["metadatas"][0][i].get("version", "1.0")
                })
            
            return similar_questions
            
        except Exception as e:
            logger.error(f"Error getting similar questions: {e}")
            return []
    
    def get_category_questions(self, category: str) -> List[Dict[str, Any]]:
        """Get all questions for a specific category with metadata."""
        try:
            results = self.faq_collection.query(
                query_embeddings=None,
                where={"category": category},
                include=["documents", "metadatas"]
            )
            
            if not results["documents"]:
                return []
            
            questions = []
            for i in range(len(results["documents"][0])):
                questions.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
            
            return questions
            
        except Exception as e:
            logger.error(f"Error getting category questions: {e}")
            return []
    
    def clear_session_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared conversation history for session {session_id}")
    
    def export_conversation_history(self, session_id: str, format: str = "json") -> str:
        """Export conversation history in specified format."""
        try:
            if session_id not in self.conversation_history:
                return ""
            
            history = self.conversation_history[session_id]
            
            if format == "json":
                return json.dumps(history, indent=2, ensure_ascii=False)
            elif format == "text":
                return "\n".join([
                    f"User: {h['user']}\nBot: {h['bot']}\n"
                    for h in history
                ])
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting conversation history: {e}")
            return ""
    
    def _compile_reference_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for reference extraction with enhanced patterns."""
        return {
            'huongdan': re.compile(
                r'\b(how|what|where|which|when|why|who|whose|whom|whose|could you|can you|please|tell me|show me|guide|explain|help|instruction|tutorial|step|process|procedure|method|way|guide|manual)'
                r'|'
                r'\b(làm sao|làm thế nào|chỉ|hướng dẫn|cách|phương pháp|quy trình|các bước|chỉ dẫn|giải thích|giúp|cho hỏi|cho biết|cần|muốn|xin)'
                r'|'
                r'\b(thủ tục|quy định|thực hiện|tiến hành|triển khai|áp dụng|sử dụng|tham khảo|tra cứu|tìm hiểu|xem|đọc|học|làm|tạo|tải|đăng ký|đăng nhập)\b',
                re.IGNORECASE
            ),
            'quydinh': re.compile(
                r'\b(rule|regulation|policy|guideline|requirement|restriction|limitation|condition|term|agreement|law|legal|permit|allow|forbid|ban|prohibit|must|should|shall|may|might|can|cannot|permission)'
                r'|'
                r'\b(quy định|quy chế|nội quy|điều lệ|điều khoản|chính sách|yêu cầu|hạn chế|điều kiện|thỏa thuận|luật|pháp lý|cho phép|cấm|bắt buộc|phải|nên|được|không được|cần|không cần)\b',
                re.IGNORECASE
            ),
            'dichvu': re.compile(
                r'\b(service|facility|amenity|feature|function|utility|tool|resource|equipment|device|system|platform|application|program|software|hardware|material|supply|support|assistance|help)'
                r'|'
                r'\b(dịch vụ|tiện ích|chức năng|công cụ|tài nguyên|thiết bị|hệ thống|nền tảng|ứng dụng|chương trình|phần mềm|phần cứng|vật tư|hỗ trợ|giúp đỡ|cung cấp|sử dụng)\b',
                re.IGNORECASE
            ),
            'thoigian': re.compile(
                r'\b(time|schedule|duration|period|hour|minute|second|day|week|month|year|morning|afternoon|evening|night|today|tomorrow|yesterday|weekend|weekday|date|deadline|term)'
                r'|'
                r'\b(thời gian|lịch|khoảng|giờ|phút|giây|ngày|tuần|tháng|năm|sáng|trưa|chiều|tối|hôm nay|ngày mai|hôm qua|cuối tuần|ngày thường|ngày|hạn|kỳ|học kỳ)\b',
                re.IGNORECASE
            ),
            'lienhe': re.compile(
                r'\b(contact|reach|find|meet|talk|speak|call|phone|email|message|chat|communicate|connect|support|help|assist|address|location|office|department|staff|personnel)'
                r'|'
                r'\b(liên hệ|gặp|tìm|nói chuyện|gọi|điện thoại|email|tin nhắn|chat|trao đổi|kết nối|hỗ trợ|giúp đỡ|địa chỉ|vị trí|văn phòng|phòng|ban|nhân viên|người phụ trách)\b',
                re.IGNORECASE
            ),
            'pronouns': re.compile(
                r'\b(it|this|that|these|those|they|them|their|such|which|what|who|whom|whose|where|when|why|how)'
                r'|'
                r'\b(nó|này|đó|những|các|họ|chúng|của|như|gì|ai|của ai|ở đâu|khi nào|tại sao|như thế nào)\b',
                re.IGNORECASE
            ),
            'temporal': re.compile(
                r'\b(earlier|before|previously|last time|ago|recent|lately|now|current|present|future|soon|later|after|next|following|upcoming)'
                r'|'
                r'\b(trước đây|lúc trước|vừa rồi|mới đây|gần đây|hiện tại|hiện nay|sắp tới|sau này|tiếp theo|sắp|đến|vào)\b',
                re.IGNORECASE
            ),
            'question_words': re.compile(
                r'\b(what|where|when|who|whom|whose|why|how|which|can|could|would|should|may|might|will|shall|do|does|did|is|are|was|were|has|have|had)'
                r'|'
                r'\b(gì|đâu|khi nào|ai|của ai|tại sao|như thế nào|làm sao|nào|có thể|nên|sẽ|phải|làm|là|có|đã|được|bị)\b',
                re.IGNORECASE
            )
        }

    def _compile_question_patterns(self) -> Dict[str, List[str]]:
        """Generate comprehensive question patterns for each category."""
        return {
            'huongdan': [
                # English patterns
                r'how (can|do|should|could|would|might) I .*\?',
                r'how (to|do I|can I|should I) .*\?',
                r'where (can|do|should|could|would) I .*\?',
                r'what (is|are) the (steps?|procedures?|methods?|ways?) to .*\?',
                r'(can|could) you (show|tell|guide|help) me .*\?',
                r'(what|which) (steps?|procedures?|methods?) should I .*\?',
                r'(please|kindly) (help|guide|show|tell) me .*\?',
                r'I need (help|guidance|instruction|assistance) .*\?',
                r'(is|are) there (any|some) (instructions?|guides?) .*\?',
                r'(looking for|searching for|need) (help|guidance) .*\?',
                # Vietnamese patterns
                r'(làm thế nào|làm sao|bằng cách nào) để .*\?',
                r'(chỉ|hướng dẫn|giúp|giải thích) .* (được không|giúp|với)\?',
                r'(ở|tại) đâu (có thể|để) .*\?',
                r'(các)? bước (để|thực hiện|tiến hành) .*\?',
                r'(quy trình|thủ tục|cách) (để|thực hiện|làm) .*\?',
                r'(cho|xin) hỏi (cách|làm sao|làm thế nào) .*\?',
                r'(cần|muốn) (biết|tìm hiểu|học|làm) .*\?',
                r'(phải|nên) (làm gì|làm sao|thực hiện) .*\?',
                r'(có thể|làm ơn) (chỉ|hướng dẫn|giúp) .*\?',
                r'(tìm|cần) (hướng dẫn|chỉ dẫn|thông tin) .*\?'
            ],
            'quydinh': [
                # English patterns
                r'(what|are) .* (rules|regulations|policies|guidelines) .*\?',
                r'(is|are) .* (allowed|permitted|prohibited|forbidden|banned) .*\?',
                r'(can|may|could|should) (I|we|students|users) .*\?',
                r'(what are|are there) (any|some) restrictions .*\?',
                r'(do|does) .* (need|require|allow|permit) .*\?',
                r'(is it|is there) (mandatory|compulsory|required|necessary) .*\?',
                r'(what|which) (requirements?|conditions?) .*\?',
                r'(tell|inform) me about .* (rules|policies) .*\?',
                r'(looking for|need) information about .* (policy|regulation) .*\?',
                r'(what happens|what if) .* (violate|break|ignore) .*\?',
                # Vietnamese patterns
                r'quy định về .*\?',
                r'(có|được) (phép|cho phép|được phép) .*\?',
                r'(có|bị) (cấm|hạn chế|giới hạn) .*\?',
                r'(có|cần|phải) (tuân theo|thực hiện) .*\?',
                r'(điều kiện|yêu cầu) (để|khi) .*\?',
                r'(những|các) (quy định|điều khoản|điều lệ) .*\?',
                r'(cho|xin) hỏi (về|những) (quy định|điều lệ) .*\?',
                r'(nếu|khi) (vi phạm|không tuân thủ) .*\?',
                r'(có|cần) (giấy tờ|thủ tục|điều kiện) gì .*\?',
                r'(theo|dựa|căn cứ) (quy định|quy chế) .*\?'
            ],
            'dichvu': [
                # English patterns
                r'how (much|many|long) .* (cost|charge|fee|price) .*\?',
                r'what (services?|facilities?|resources?) .*\?',
                r'(where|how) (can|do) I (access|use|find) .* (service|facility) .*\?',
                r'(is|are) there .* (service|facility|resource) .*\?',
                r'(do|does) .* provide .* (service|support) .*\?',
                r'(can|could) I (use|access|get) .* (service|facility) .*\?',
                r'(tell|inform) me about .* (services|facilities) .*\?',
                r'(what|which) (types?|kinds?) of (services?|support) .*\?',
                r'(looking for|need) .* (service|assistance|help) .*\?',
                r'(how|where) to (access|use|get) .* (service|facility) .*\?',
                # Vietnamese patterns
                r'(phí|chi phí|giá) (dịch vụ|sử dụng) .*\?',
                r'(dịch vụ|tiện ích) (gì|nào|như thế nào) .*\?',
                r'(có|cung cấp) (dịch vụ|hỗ trợ) .*\?',
                r'(làm sao|ở đâu) (để|có thể) (sử dụng|truy cập) .*\?',
                r'(các|những) (dịch vụ|tiện ích|hỗ trợ) .*\?',
                r'(cần|phải) (làm gì|trả|đóng) .* (phí|tiền) .*\?',
                r'(muốn|cần) (sử dụng|đăng ký) (dịch vụ|tiện ích) .*\?',
                r'(cho|xin) hỏi về (dịch vụ|tiện ích) .*\?',
                r'(tìm|cần) (dịch vụ|hỗ trợ|giúp đỡ) .*\?',
                r'(có|được) (sử dụng|truy cập|đăng ký) .*\?'
            ],
            'thoigian': [
                # English patterns
                r'when (is|are|will|can|should) .*\?',
                r'what time (is|are|will|does|do) .*\?',
                r'how long (is|are|will|does|do) .*\?',
                r'(what|which) (days?|hours?|times?) .*\?',
                r'(is|are) .* (open|available|accessible) .* (time|day|date) .*\?',
                r'(what|which) (schedule|timetable|hours) .*\?',
                r'(tell|inform) me .* (time|schedule|hours) .*\?',
                r'(during|within|between) what (time|hours) .*\?',
                r'(from|until|till) (what|which) time .*\?',
                r'(how|when) (often|frequently|long) .*\?',
                # Vietnamese patterns
                r'(thời gian|giờ|lúc) (nào|gì|như thế nào) .*\?',
                r'(khi nào|lúc nào|bao giờ) .*\?',
                r'(bao lâu|mất|trong) .* (thời gian|giờ) .*\?',
                r'(mấy|những|các) (giờ|ngày|buổi) .*\?',
                r'(có|được) (mở cửa|hoạt động|làm việc) .* (giờ|lúc) .*\?',
                r'(lịch|thời gian biểu|giờ giấc) .*\?',
                r'(từ|đến|tới) (mấy giờ|khi nào|lúc nào) .*\?',
                r'(trong|vào) (thời gian|lúc|giờ) .*\?',
                r'(thường|hay|định kỳ) .* (khi nào|lúc nào) .*\?',
                r'(kéo dài|diễn ra|xảy ra) .* (bao lâu|trong) .*\?'
            ],
            'lienhe': [
                # English patterns
                r'(who|whom) (can|should|do) I contact .*\?',
                r'(where|how) (can|should|do) I contact .*\?',
                r'(how|where) to (contact|reach|find) .*\?',
                r'(what|which) (department|office|person) .*\?',
                r'(is|are) there (someone|anybody|anyone) .*\?',
                r'(can|could) you (help|direct|connect) me .*\?',
                r'(looking for|need) (contact|help) from .*\?',
                r'(what|which) is the (contact|email|phone) .*\?',
                r'(how|where) (can|do) I (get|find|reach) (help|support) .*\?',
                r'(who|whom) (is|are) (responsible|in charge) .*\?',
                # Vietnamese patterns
                r'(liên hệ|gặp|tìm) (ai|ở đâu|như thế nào) .*\?',
                r'(gặp ai|tìm ai|ai) (để|có thể) .*\?',
                r'(cần|phải) (gặp|liên hệ|tìm) .*\?',
                r'(phòng|ban|bộ phận) nào .*\?',
                r'(có|được) (ai|người nào) .*\?',
                r'(làm sao|ở đâu) (để|có thể) (liên hệ|gặp) .*\?',
                r'(số điện thoại|email|địa chỉ) .* (là gì|ở đâu) .*\?',
                r'(ai|người nào) (phụ trách|chịu trách nhiệm) .*\?',
                r'(muốn|cần) (gặp|trao đổi|tìm) .*\?',
                r'(cho|xin) hỏi (về|cách) (liên hệ|gặp) .*\?'
            ]
        } 