from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Import vector database integration for real search
from src.utils.vector_db_integration import get_user_data

logger = logging.getLogger(__name__)

class SearchAgent:
    def __init__(self, model: BaseLanguageModel):
        self.llm = model
        
        self.search_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in mental health information retrieval.
Your role is to:
1. Extract key search terms from user queries
2. Identify relevant mental health topics and concepts
3. Find appropriate resources and information
4. Validate information accuracy and relevance
5. Format results for easy comprehension

Focus on:
- Evidence-based information
- Reputable sources
- Current best practices
- User-appropriate content
- Trigger-aware presentation"""),
            HumanMessage(content="""Query: {query}
Context: {context}
Previous Searches: {history}

Extract search parameters and format results:
Search Terms: [key terms]
Topics: [relevant topics]
Required Info: [specific information needed]
Source Preferences: [preferred source types]
Content Warnings: [any needed warnings]""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.search_prompt)
        self.search_history = []

    def search(self, query: str, context: Optional[str] = None) -> str:
        """Perform an intelligent search based on user query and context"""
        try:
            # Get search parameters
            search_params = self.chain.run(
                query=query,
                context=context or "",
                history=self._format_history()
            )
            
            # Parse search parameters
            params = self._parse_search_params(search_params)

            # Perform real search using vector database
            results = self._execute_search(params)
            
            # Store in history
            self.search_history.append({
                'query': query,
                'params': params,
                'results': results,
                'timestamp': self._get_timestamp()
            })
            
            return results
            
        except Exception as e:
            return f"Unable to perform search at this time. Error: {str(e)}"

    def _parse_search_params(self, text: str) -> Dict:
        """Parse the structured search parameters"""
        params = {
            'terms': [],
            'topics': [],
            'required_info': [],
            'source_preferences': [],
            'content_warnings': []
        }
        
        try:
            lines = text.strip().split('\n')
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = [x.strip() for x in line.split(':', 1)]
                
                if 'Search Terms' in key:
                    params['terms'] = [t.strip() for t in value.split(',')]
                elif 'Topics' in key:
                    params['topics'] = [t.strip() for t in value.split(',')]
                elif 'Required Info' in key:
                    params['required_info'] = [i.strip() for i in value.split(',')]
                elif 'Source Preferences' in key:
                    params['source_preferences'] = [s.strip() for s in value.split(',')]
                elif 'Content Warnings' in key:
                    params['content_warnings'] = [w.strip() for w in value.split(',')]

        except Exception as e:
            logger.warning(f"Error parsing search parameters: {str(e)}")
            pass

        return params

    def _execute_search(self, params: Dict) -> str:
        """
        Execute real search using vector database and knowledge base.

        This method performs actual semantic search across stored mental health resources,
        user conversation history, and clinical knowledge.
        """
        results = []

        try:
            # Build comprehensive search query from parameters
            search_query = self._build_search_query(params)

            # Search across different data types
            # 1. Search mental health resources
            resource_results = self._search_resources(search_query, params['topics'])
            results.extend(resource_results)

            # 2. Search conversation history for relevant past discussions
            history_results = self._search_conversation_history(search_query)
            results.extend(history_results)

            # 3. Search clinical knowledge base
            clinical_results = self._search_clinical_knowledge(params['topics'])
            results.extend(clinical_results)

            # Rank and limit results
            ranked_results = self._rank_results(results, search_query)
            top_results = ranked_results[:10]  # Top 10 most relevant

            # Format results with content warnings if needed
            return self._format_results(top_results, params.get('content_warnings', []))

        except Exception as e:
            logger.error(f"Error executing search: {str(e)}")
            return f"Search encountered an error: {str(e)}. Please try a different query."

    def _build_search_query(self, params: Dict) -> str:
        """Build comprehensive search query from parameters."""
        query_parts = []

        if params.get('terms'):
            query_parts.extend(params['terms'])
        if params.get('topics'):
            query_parts.extend(params['topics'])
        if params.get('required_info'):
            query_parts.extend(params['required_info'])

        return ' '.join(query_parts)

    def _search_resources(self, query: str, topics: List[str]) -> List[Dict]:
        """
        Search vector database for mental health resources using semantic similarity.

        Performs topic-based semantic search across the mental health resources database,
        retrieving relevant articles, guides, and educational content for each topic.

        Args:
            query (str): User's search query for context
            topics (List[str]): List of mental health topics extracted from query
                               (e.g., ['depression', 'anxiety', 'coping'])

        Returns:
            List[Dict]: Mental health resource items, each containing:
                - title (str): Resource title
                - source (str): Resource source/origin
                - summary (str): Resource content summary
                - relevance_score (float): Semantic similarity score (0-1)
                - type (str): Always 'resource'
                - url (str): Resource URL if available

        Example:
            >>> results = self._search_resources("coping strategies", ["anxiety", "stress"])
            >>> print(len(results))
            6  # Up to 3 results per topic
            >>> print(results[0]['title'])
            'Anxiety Management Techniques'
        """
        results = []

        try:
            # Query vector database for relevant resources
            # Using get_user_data for semantic search
            for topic in topics:
                data_results = get_user_data(
                    user_id="default_user",
                    data_type="mental_health_resources",
                    query=f"{topic} {query}",
                    top_k=3
                )

                if data_results:
                    for item in data_results:
                        results.append({
                            'title': item.get('title', f'Resource about {topic}'),
                            'source': item.get('source', 'Knowledge Base'),
                            'summary': item.get('content', item.get('summary', 'Relevant mental health resource')),
                            'relevance_score': item.get('score', 0.5),
                            'type': 'resource',
                            'url': item.get('url', '')
                        })

        except Exception as e:
            logger.warning(f"Error searching resources: {str(e)}")

        return results

    def _search_conversation_history(self, query: str) -> List[Dict]:
        """
        Search user's conversation history for contextually relevant past discussions.

        Performs semantic search across stored conversation history to find previous
        discussions related to the current query, providing continuity and context.

        Args:
            query (str): User's search query

        Returns:
            List[Dict]: Conversation history items (max 3), each containing:
                - title (str): Always 'Previous Discussion'
                - source (str): Always 'Your Conversation History'
                - summary (str): Content of the past conversation
                - relevance_score (float): Semantic similarity to query (0-1)
                - type (str): Always 'history'
                - timestamp (str): When the conversation occurred

        Example:
            >>> results = self._search_conversation_history("dealing with work stress")
            >>> print(results[0]['summary'][:50])
            'Last week we discussed your workplace anxiety...'
            >>> print(results[0]['timestamp'])
            '2025-01-15T14:30:00'
        """
        results = []

        try:
            # Search conversation history in vector DB
            history_data = get_user_data(
                user_id="default_user",
                data_type="conversation_history",
                query=query,
                top_k=3
            )

            if history_data:
                for item in history_data:
                    results.append({
                        'title': 'Previous Discussion',
                        'source': 'Your Conversation History',
                        'summary': item.get('content', item.get('message', 'Related past conversation')),
                        'relevance_score': item.get('score', 0.5),
                        'type': 'history',
                        'timestamp': item.get('timestamp', 'Recent')
                    })

        except Exception as e:
            logger.warning(f"Error searching conversation history: {str(e)}")

        return results

    def _search_clinical_knowledge(self, topics: List[str]) -> List[Dict]:
        """
        Search clinical knowledge base for evidence-based mental health information.

        Queries the clinical knowledge database for research-backed information,
        treatment guidelines, and professional insights related to specified topics.

        Args:
            topics (List[str]): Mental health topics to search
                              (e.g., ['CBT', 'depression treatment', 'medication'])

        Returns:
            List[Dict]: Clinical knowledge items (max 2 per topic), each containing:
                - title (str): Knowledge item title
                - source (str): Always 'Clinical Knowledge Base'
                - summary (str): Evidence-based clinical information
                - relevance_score (float): Semantic similarity score (0-1)
                - type (str): Always 'clinical'

        Example:
            >>> results = self._search_clinical_knowledge(["depression", "CBT"])
            >>> print(len(results))
            4  # Up to 2 results per topic
            >>> print(results[0]['type'])
            'clinical'
            >>> print(results[0]['title'])
            'Clinical Information: Depression'
        """
        results = []

        try:
            for topic in topics:
                clinical_data = get_user_data(
                    user_id="default_user",
                    data_type="clinical_knowledge",
                    query=topic,
                    top_k=2
                )

                if clinical_data:
                    for item in clinical_data:
                        results.append({
                            'title': item.get('title', f'Clinical Information: {topic}'),
                            'source': item.get('source', 'Clinical Knowledge Base'),
                            'summary': item.get('content', 'Evidence-based clinical information'),
                            'relevance_score': item.get('score', 0.5),
                            'type': 'clinical'
                        })

        except Exception as e:
            logger.warning(f"Error searching clinical knowledge: {str(e)}")

        return results

    def _rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Rank search results by relevance score in descending order.

        Sorts combined search results from all sources (resources, history, clinical)
        by their semantic similarity scores, prioritizing most relevant items.

        Args:
            results (List[Dict]): Combined search results from all sources
            query (str): Original search query (currently unused, reserved for future
                        query-specific ranking algorithms)

        Returns:
            List[Dict]: Same results sorted by relevance_score (highest first)

        Example:
            >>> results = [
            ...     {'title': 'Resource A', 'relevance_score': 0.65},
            ...     {'title': 'Resource B', 'relevance_score': 0.92},
            ...     {'title': 'Resource C', 'relevance_score': 0.43}
            ... ]
            >>> ranked = self._rank_results(results, "anxiety")
            >>> print(ranked[0]['relevance_score'])
            0.92
        """
        # Sort by relevance score (highest first)
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)

    def _format_results(self, results: List[Dict], content_warnings: List[str]) -> str:
        """
        Format search results into human-readable markdown text.

        Converts structured search results into formatted text suitable for display,
        including content warnings, numbered results, and metadata (URLs, timestamps).

        Args:
            results (List[Dict]): Ranked search results to format
            content_warnings (List[str]): List of content warnings to display
                                         (e.g., ['crisis', 'self-harm'])

        Returns:
            str: Formatted markdown string with:
                - Content warnings (if any)
                - Result count
                - Numbered list of results with title, source, summary (max 200 chars),
                  optional URL, and optional timestamp

        Example:
            >>> results = [
            ...     {
            ...         'title': 'Coping with Anxiety',
            ...         'source': 'NIMH',
            ...         'summary': 'Evidence-based strategies for managing anxiety...',
            ...         'url': 'https://nimh.nih.gov/anxiety',
            ...         'relevance_score': 0.87
            ...     }
            ... ]
            >>> formatted = self._format_results(results, ['crisis'])
            >>> print(formatted)
            ⚠️ Content Warning: crisis

            Found 1 relevant result(s):

            1. **Coping with Anxiety**
               Source: NIMH
               Evidence-based strategies for managing anxiety...
               Link: https://nimh.nih.gov/anxiety
        """
        if not results:
            return "No specific resources found for your query. Please try:\n- Using different search terms\n- Being more specific about what you're looking for\n- Asking in a different way"

        formatted_results = ""

        # Add content warnings if present
        if content_warnings:
            formatted_results = f"⚠️ Content Warning: {', '.join(content_warnings)}\n\n"

        formatted_results += f"Found {len(results)} relevant result(s):\n\n"

        for idx, result in enumerate(results, 1):
            formatted_results += f"{idx}. **{result['title']}**\n"
            formatted_results += f"   Source: {result['source']}\n"
            formatted_results += f"   {result['summary'][:200]}{'...' if len(result['summary']) > 200 else ''}\n"

            if result.get('url'):
                formatted_results += f"   Link: {result['url']}\n"
            if result.get('timestamp'):
                formatted_results += f"   Date: {result['timestamp']}\n"

            formatted_results += "\n"

        return formatted_results

    def _format_history(self) -> str:
        """Format search history for context"""
        if not self.search_history:
            return "No previous searches"
            
        recent_searches = self.search_history[-3:]  # Get last 3 searches
        formatted = "Recent Searches:\n"
        
        for search in recent_searches:
            formatted += f"Query: {search['query']}\n"
            formatted += f"Topics: {', '.join(search['params']['topics'])}\n"
            
        return formatted
        
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()

