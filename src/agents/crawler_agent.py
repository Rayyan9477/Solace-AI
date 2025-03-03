from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import requests
import asyncio
from urllib.parse import urlparse
import logging
from concurrent.futures import ThreadPoolExecutor
from config.settings import AppConfig
from utils.helpers import TextHelper, DocumentHelper
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

class MentalHealthSpider(CrawlSpider):
    """Spider for crawling mental health resources"""
    name = 'mental_health_spider'
    
    def __init__(self, start_urls=None, allowed_domains=None, *args, **kwargs):
        self.start_urls = start_urls or []
        self.allowed_domains = allowed_domains or []
        self.text_helper = TextHelper()
        super().__init__(*args, **kwargs)
        
        # Define crawling rules
        self.rules = (
            Rule(
                LinkExtractor(
                    allow=AppConfig.CRAWLER_CONFIG.get('url_patterns', []),
                    deny=AppConfig.CRAWLER_CONFIG.get('blocked_patterns', [])
                ),
                callback='parse_item',
                follow=True
            ),
        )
        
    async def parse_item(self, response):
        """Parse webpage content"""
        try:
            # Extract main content
            content = ' '.join(response.css(AppConfig.CRAWLER_CONFIG['content_selectors']).getall())
            
            # Clean and process text
            processed_content = await self.text_helper.preprocess_text(content)
            
            # Extract metadata
            metadata = {
                'url': response.url,
                'title': response.css('title::text').get(''),
                'timestamp': self.text_helper.get_timestamp()
            }
            
            return {
                'content': processed_content,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to parse page {response.url}: {str(e)}")
            return None

@tool("content_crawler")
async def crawl_content(query: str, urls: List[str], max_pages: int = 10) -> Dict[str, Any]:
    """
    Crawls mental health resources from specified URLs
    
    Args:
        query: Search query to guide crawling
        urls: List of URLs to crawl
        max_pages: Maximum number of pages to crawl
        
    Returns:
        Dictionary containing crawled content and metadata
    """
    try:
        process = CrawlerProcess(AppConfig.CRAWLER_CONFIG.get('settings', {}))
        doc_helper = DocumentHelper()
        
        # Configure spider
        spider = MentalHealthSpider(
            start_urls=urls,
            allowed_domains=[url.split('/')[2] for url in urls]
        )
        
        # Run crawler
        results = []
        process.crawl(
            spider,
            max_pages=max_pages
        )
        process.start()
        
        # Process results
        documents = []
        for result in results:
            if not result:
                continue
                
            # Create document
            doc = await doc_helper.create_document(
                text=result['content'],
                metadata=result['metadata']
            )
            documents.append(doc)
            
        return {
            'documents': documents,
            'query': query,
            'total_pages': len(documents),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Crawling failed: {str(e)}")
        return {
            'documents': [],
            'query': query,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@tool("content_validator")
async def validate_content(documents: List[Dict[str, Any]], min_length: int = 100) -> Dict[str, Any]:
    """
    Validates and filters crawled mental health content
    
    Args:
        documents: List of documents to validate
        min_length: Minimum content length to consider valid
        
    Returns:
        Dictionary containing validated documents and metadata
    """
    try:
        text_helper = TextHelper()
        valid_docs = []
        
        for doc in documents:
            # Check content length
            if len(doc['text']) < min_length:
                continue
                
            # Extract keywords
            keywords = await text_helper.extract_keywords(doc['text'])
            
            # Check relevance
            relevant_terms = AppConfig.CRAWLER_CONFIG.get('relevant_terms', [])
            if not any(term in keywords for term in relevant_terms):
                continue
                
            # Update metadata
            doc['metadata'].update({
                'keywords': keywords,
                'validated': True,
                'validation_timestamp': datetime.now().isoformat()
            })
            
            valid_docs.append(doc)
            
        return {
            'documents': valid_docs,
            'total_valid': len(valid_docs),
            'total_processed': len(documents),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Content validation failed: {str(e)}")
        return {
            'documents': [],
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

class CrawlerAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        # Let BaseAgent handle memory initialization
        super().__init__(
            api_key=AppConfig.ANTHROPIC_API_KEY,
            name="information_gatherer",
            description="Expert system for gathering and validating mental health information",
            tools=[crawl_content, validate_content],
            knowledge=AgentKnowledge()
        )
        
        self.search_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in mental health information curation.
Your role is to:
1. Analyze and synthesize information from multiple sources
2. Ensure accuracy and reliability of information
3. Present information in a helpful, accessible way
4. Filter out potentially harmful content
5. Prioritize evidence-based resources

Focus on:
- Credible sources
- Current best practices
- User-appropriate content
- Trigger awareness
- Professional guidelines"""),
            HumanMessage(content="""Query: {query}
Crawled Content: {content}
Validation Results: {validation}
Previous Searches: {history}

Provide a structured information summary:
Key Points: [main takeaways]
Sources: [credible sources used]
Recommendations: [practical suggestions]
Additional Resources: [helpful links]
Content Warnings: [if applicable]""")
        ])

    async def _generate_response(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Get crawled content
            crawled_data = tool_results.get('content_crawler', {})
            validation_data = tool_results.get('content_validator', {})
            
            # Generate information summary
            llm_response = await self.llm.agenerate_messages([
                self.search_prompt.format_messages(
                    query=input_data.get('query', ''),
                    content=crawled_data.get('documents', []),
                    validation=validation_data,
                    history=self._format_history(context.get('memory', {}))
                )[0]
            ])
            
            # Parse response
            summary = self._parse_result(llm_response.generations[0][0].text)
            
            # Add metadata
            summary['timestamp'] = datetime.now().isoformat()
            summary['confidence'] = self._calculate_confidence(
                summary,
                crawled_data.get('documents', [])
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return self._fallback_analysis()

    def _parse_result(self, text: str) -> Dict[str, Any]:
        """Parse the structured information summary"""
        result = {
            'key_points': [],
            'sources': [],
            'recommendations': [],
            'additional_resources': [],
            'content_warnings': []
        }
        
        try:
            lines = text.strip().split('\n')
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = [x.strip() for x in line.split(':', 1)]
                
                if 'Key Points' in key:
                    result['key_points'] = [p.strip() for p in value.split(',')]
                elif 'Sources' in key:
                    result['sources'] = [s.strip() for s in value.split(',')]
                elif 'Recommendations' in key:
                    result['recommendations'] = [r.strip() for r in value.split(',')]
                elif 'Additional Resources' in key:
                    result['additional_resources'] = [r.strip() for r in value.split(',')]
                elif 'Content Warnings' in key:
                    result['content_warnings'] = [w.strip() for w in value.split(',')]
                    
        except Exception:
            pass
            
        return result

    def _calculate_confidence(
        self,
        summary: Dict[str, Any],
        documents: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in information summary"""
        confidence = 1.0
        
        # Lower confidence if few results
        if len(documents) < 3:
            confidence *= 0.8
            
        # Lower confidence if few trusted sources
        trusted_sources = sum(1 for doc in documents 
                            if self._is_trusted_source(doc['metadata'].get('url', '')))
        if trusted_sources < 2:
            confidence *= 0.7
            
        # Lower confidence if summary is incomplete
        if not summary['key_points'] or not summary['sources']:
            confidence *= 0.8
            
        return confidence

    def _format_history(self, memory: Dict[str, Any]) -> str:
        """Format search history"""
        if not memory:
            return "No previous search history available"
            
        recent_searches = memory.get('recent_searches', [])[:3]
        if not recent_searches:
            return "No recent searches"
            
        formatted = "Recent Searches:\n"
        for search in recent_searches:
            formatted += f"- Query: {search.get('query')}\n"
            formatted += f"  Sources: {', '.join(search.get('sources', []))}\n"
            
        return formatted

    def _is_trusted_source(self, url: str) -> bool:
        """Check if source is trusted"""
        try:
            domain = urlparse(url).netloc
            trusted_domains = {
                'nimh.nih.gov',
                'who.int',
                'mayoclinic.org',
                'psychiatry.org',
                'healthline.com',
                'psychologytoday.com'
            }
            return domain in trusted_domains
        except:
            return False

    def _fallback_analysis(self) -> Dict[str, Any]:
        """Conservative fallback analysis"""
        return {
            'key_points': ['Unable to gather comprehensive information'],
            'sources': [],
            'recommendations': [
                'Visit official mental health websites',
                'Consult with mental health professionals',
                'Check reputable organizations for resources'
            ],
            'additional_resources': [
                'National Institute of Mental Health (nimh.nih.gov)',
                'World Health Organization (who.int/mental_health)'
            ],
            'content_warnings': ['Limited information available'],
            'confidence': 0.3,  # Very low confidence for fallback
            'timestamp': datetime.now().isoformat()
        }