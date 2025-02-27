from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from agno.tools import Tool
from agno.memory import ConversationMemory
from agno.knowledge import VectorKnowledge
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

class MentalHealthSpider(CrawlSpider):
    name = 'mental_health'
    
    # Trusted domains for mental health information
    allowed_domains = [
        'nimh.nih.gov',
        'psychiatry.org',
        'who.int',
        'mayoclinic.org',
        'healthline.com',
        'psychologytoday.com'
    ]
    
    # Start URLs for crawling
    start_urls = [
        'https://www.nimh.nih.gov/health/',
        'https://www.psychiatry.org/patients-families',
        'https://www.who.int/mental_health/',
        'https://www.mayoclinic.org/diseases-conditions/mental-illness/symptoms-causes/',
        'https://www.healthline.com/health/mental-health',
        'https://www.psychologytoday.com/us/basics/'
    ]
    
    # Rules for following links
    rules = (
        Rule(
            LinkExtractor(
                allow=('health', 'mental', 'disorder', 'condition', 'treatment'),
                deny=('login', 'signup', 'account', 'cart')
            ),
            callback='parse_item',
            follow=True
        ),
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = []
        
    def parse_item(self, response):
        """Parse mental health content"""
        # Extract main content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in ['script', 'style', 'nav', 'footer', 'header']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Extract relevant content
        title = soup.find('h1').get_text() if soup.find('h1') else ''
        content = ' '.join([p.get_text() for p in soup.find_all('p')])
        
        if title and content:
            self.results.append({
                'url': response.url,
                'title': title,
                'content': content[:1000],  # Limit content length
                'source': urlparse(response.url).netloc
            })

class ContentCrawlerTool(Tool):
    def __init__(self):
        super().__init__(
            name="content_crawler",
            description="Crawls trusted mental health websites for relevant information"
        )
        self.process = CrawlerProcess({
            'USER_AGENT': 'MentalHealthBot/1.0 (+https://example.com/bot)',
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS': 16,
            'DOWNLOAD_DELAY': 1,
            'COOKIES_ENABLED': False
        })
        
    async def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get('query', '')
        
        # Create spider instance
        spider = MentalHealthSpider()
        
        # Run spider in a separate thread
        with ThreadPoolExecutor() as executor:
            await asyncio.get_event_loop().run_in_executor(
                executor,
                self.process.crawl,
                spider
            )
        
        # Filter and rank results
        relevant_results = self._filter_results(spider.results, query)
        
        return {
            'results': relevant_results[:5],  # Return top 5 results
            'total_found': len(relevant_results),
            'sources': list(set(r['source'] for r in relevant_results))
        }
        
    def _filter_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Filter and rank results based on relevance"""
        query_terms = set(query.lower().split())
        
        # Score and filter results
        scored_results = []
        for result in results:
            score = self._calculate_relevance(result, query_terms)
            if score > 0.3:  # Minimum relevance threshold
                result['relevance_score'] = score
                scored_results.append(result)
                
        # Sort by relevance
        return sorted(scored_results, key=lambda x: x['relevance_score'], reverse=True)
        
    def _calculate_relevance(self, result: Dict, query_terms: set) -> float:
        """Calculate result relevance score"""
        text = f"{result['title']} {result['content']}".lower()
        
        # Calculate term frequency
        term_matches = sum(1 for term in query_terms if term in text)
        
        # Calculate relevance score
        score = term_matches / len(query_terms)
        
        # Boost score for trusted sources
        if 'nimh.nih.gov' in result['source']:
            score *= 1.3
        elif 'who.int' in result['source']:
            score *= 1.2
            
        return min(score, 1.0)

class ContentValidatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="content_validator",
            description="Validates and sanitizes crawled content"
        )
        self.unsafe_patterns = [
            'suicide', 'self-harm', 'kill', 'death',
            'abuse', 'violence', 'weapon'
        ]
        
    async def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get('content', [])
        
        validated_content = []
        warnings = []
        
        for item in content:
            validation = self._validate_content(item)
            if validation['safe']:
                validated_content.append(item)
            else:
                warnings.extend(validation['warnings'])
                
        return {
            'validated_content': validated_content,
            'warnings': warnings,
            'removed_count': len(content) - len(validated_content)
        }
        
    def _validate_content(self, content_item: Dict) -> Dict[str, Any]:
        """Validate content for safety and appropriateness"""
        text = f"{content_item['title']} {content_item['content']}".lower()
        warnings = []
        
        # Check for unsafe patterns
        found_patterns = [p for p in self.unsafe_patterns if p in text]
        
        # Validate source
        if not self._is_trusted_source(content_item['source']):
            warnings.append(f"Untrusted source: {content_item['source']}")
            
        return {
            'safe': not found_patterns and not warnings,
            'warnings': warnings + [f"Found unsafe pattern: {p}" for p in found_patterns]
        }
        
    def _is_trusted_source(self, domain: str) -> bool:
        """Check if the source is trusted"""
        trusted_domains = {
            'nimh.nih.gov',
            'who.int',
            'mayoclinic.org',
            'psychiatry.org',
            'healthline.com',
            'psychologytoday.com'
        }
        return domain in trusted_domains

class CrawlerAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="information_gatherer",
            description="Expert system for gathering and validating mental health information",
            tools=[
                ContentCrawlerTool(),
                ContentValidatorTool()
            ],
            memory=ConversationMemory(),
            knowledge=VectorKnowledge()
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
                    content=crawled_data.get('results', []),
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
                crawled_data.get('results', [])
            )
            
            return summary
            
        except Exception as e:
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
        results: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in information summary"""
        confidence = 1.0
        
        # Lower confidence if few results
        if len(results) < 3:
            confidence *= 0.8
            
        # Lower confidence if few trusted sources
        trusted_sources = sum(1 for r in results if self._is_trusted_source(r['source']))
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

    def _is_trusted_source(self, domain: str) -> bool:
        """Check if source is trusted"""
        trusted_domains = {
            'nimh.nih.gov',
            'who.int',
            'mayoclinic.org',
            'psychiatry.org',
            'healthline.com',
            'psychologytoday.com'
        }
        return domain in trusted_domains

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