from langchain_community.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, List, Optional
import json

class SearchAgent:
    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=api_key,
            temperature=0.3,
            max_tokens=1000
        )
        
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
            
            # Perform search (implement your search logic here)
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
                    
        except Exception:
            pass
            
        return params

    def _execute_search(self, params: Dict) -> str:
        """Execute the search using the parsed parameters"""
        # Implement your actual search logic here
        # This is a placeholder that returns formatted results
        results = []
        
        # Add relevant resources (replace with actual search implementation)
        if 'anxiety' in params['topics']:
            results.append({
                'title': 'Understanding Anxiety Disorders',
                'source': 'National Institute of Mental Health',
                'url': 'https://www.nimh.nih.gov/health/topics/anxiety-disorders',
                'summary': 'Comprehensive overview of anxiety disorders, symptoms, and treatments.'
            })
            
        if 'depression' in params['topics']:
            results.append({
                'title': 'Depression: What You Need to Know',
                'source': 'American Psychological Association',
                'url': 'https://www.apa.org/topics/depression',
                'summary': 'Expert information about depression, its causes, and treatment options.'
            })
            
        # Format results
        if not results:
            return "No specific resources found. Please try refining your search."
            
        formatted_results = "Relevant Resources:\n\n"
        for r in results:
            formatted_results += f"- {r['title']}\n"
            formatted_results += f"  Source: {r['source']}\n"
            formatted_results += f"  Summary: {r['summary']}\n\n"
            
        if params['content_warnings']:
            formatted_results = "Content Warning: " + ", ".join(params['content_warnings']) + "\n\n" + formatted_results
            
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
        from datetime import datetime
        return datetime.now().isoformat()

