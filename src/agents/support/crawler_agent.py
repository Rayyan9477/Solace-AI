from typing import Dict, Any, Optional, List
from ..base.base_agent import BaseAgent
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
from urllib.parse import urlparse
import logging
from src.config.settings import AppConfig
from src.utils.helpers import TextHelper, DocumentHelper
from langchain.schema.language_model import BaseLanguageModel

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
    def __init__(self, model: BaseLanguageModel):
        # Let BaseAgent handle memory initialization
        super().__init__(
            model=model,
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

    def safe_crawl(self, query: str) -> str:
        """
        Safely crawl and retrieve REAL mental health resources from trusted websites.

        This method performs actual web crawling, content extraction, validation,
        and summarization of mental health information from reputable sources.

        Args:
            query: Search query for mental health resources

        Returns:
            Formatted string containing crawled, validated, and safe resources
        """
        try:
            # Define safe domains for mental health resources
            safe_domains = [
                'nimh.nih.gov',
                'who.int',
                'mayoclinic.org',
                'psychiatry.org',
                'mentalhealth.gov',
                'samhsa.gov'
            ]

            # Build actual URLs for crawling
            urls_to_crawl = self._build_crawl_urls(query, safe_domains)

            logger.info(f"Starting real crawl for query: {query} across {len(urls_to_crawl)} URLs")

            # Crawl and extract content from each URL
            crawled_results = []
            for url in urls_to_crawl[:3]:  # Limit to top 3 to avoid timeouts
                try:
                    crawled_data = self._crawl_single_url(url)
                    if crawled_data:
                        crawled_results.append(crawled_data)
                except Exception as e:
                    logger.warning(f"Failed to crawl {url}: {str(e)}")
                    continue

            if not crawled_results:
                logger.warning(f"No content successfully crawled for query: {query}")
                return self._get_fallback_resources(query)

            # Validate and filter crawled content
            validated_results = []
            for result in crawled_results:
                validation = self._validate_crawled_content(result['content'])
                if validation['is_safe'] and validation['quality_score'] > 0.5:
                    result['validation'] = validation
                    validated_results.append(result)

            if not validated_results:
                logger.warning("No crawled content passed validation")
                return self._get_fallback_resources(query)

            # Format and return results
            return self._format_crawled_results(validated_results, query)

        except Exception as e:
            logger.error(f"Safe crawling failed: {str(e)}")
            return self._get_fallback_resources(query)

    def _build_crawl_urls(self, query: str, domains: List[str]) -> List[str]:
        """Build specific URLs to crawl based on query and domains."""
        urls = []

        # Encode query for URL safety
        import urllib.parse
        encoded_query = urllib.parse.quote(query)

        for domain in domains:
            # Build domain-specific search URLs
            if 'nimh.nih.gov' in domain:
                urls.append(f'https://{domain}/health/topics/{encoded_query.replace("+", "-")}')
            elif 'who.int' in domain:
                urls.append(f'https://{domain}/news-room/fact-sheets/detail/{encoded_query}')
            elif 'mayoclinic.org' in domain:
                urls.append(f'https://www.{domain}/diseases-conditions/{encoded_query}/symptoms-causes')
            else:
                urls.append(f'https://www.{domain}/search?q={encoded_query}')

        return urls

    def _crawl_single_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Crawl a single URL and extract content using requests/BeautifulSoup."""
        try:
            # Use requests to fetch the page
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'MentalHealthBot/1.0 (Educational Purpose)'
            })

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: Status {response.status_code}")
                return None

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else "Untitled"

            # Extract main content (prioritize article/main tags)
            main_content = soup.find('main') or soup.find('article') or soup.find('body')

            if not main_content:
                return None

            # Extract text, removing scripts and styles
            for tag in main_content(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()

            content_text = main_content.get_text(separator='\n', strip=True)

            # Clean and truncate content
            content_text = '\n'.join([
                line.strip() for line in content_text.split('\n')
                if line.strip() and len(line.strip()) > 20
            ])[:2000]  # Limit to 2000 chars

            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'domain': url.split('/')[2],
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return None

    def _validate_crawled_content(self, content: str) -> Dict[str, Any]:
        """Validate crawled content for safety and quality."""
        validation_result = {
            'is_safe': True,
            'quality_score': 0.7,
            'issues': []
        }

        # Check content length
        if len(content) < 100:
            validation_result['quality_score'] -= 0.3
            validation_result['issues'].append('Content too short')

        # Check for harmful content
        harmful_patterns = ['suicide methods', 'self-harm instructions', 'dangerous practices']
        content_lower = content.lower()

        for pattern in harmful_patterns:
            if pattern in content_lower:
                validation_result['is_safe'] = False
                validation_result['issues'].append(f'Harmful content: {pattern}')
                break

        # Check for quality indicators
        quality_indicators = ['research', 'evidence', 'study', 'treatment', 'professional']
        quality_score_boost = sum(1 for indicator in quality_indicators if indicator in content_lower)
        validation_result['quality_score'] += min(0.3, quality_score_boost * 0.1)

        return validation_result

    def _format_crawled_results(self, results: List[Dict], query: str) -> str:
        """Format crawled and validated results for presentation."""
        output = f"**Real-time crawled results for: {query}**\n\n"
        output += f"Found {len(results)} validated source(s):\n\n"

        for idx, result in enumerate(results, 1):
            output += f"{idx}. **{result['title']}**\n"
            output += f"   Source: {result['domain']}\n"
            output += f"   Quality Score: {result['validation']['quality_score']:.2f}/1.0\n"

            # Show content preview
            preview = result['content'][:300]
            output += f"   Content: {preview}...\n"
            output += f"   Full URL: {result['url']}\n\n"

        output += "\n**Note:** Information crawled from trusted mental health sources. "
        output += "Always consult with healthcare professionals for personalized advice.\n"

        return output

    def _get_fallback_resources(self, query: str) -> str:
        """Provide fallback resources when crawling fails."""
        return f"""Unable to crawl live content for "{query}" at this time.

**Trusted Mental Health Resources:**

1. **National Institute of Mental Health (NIMH)**
   - https://www.nimh.nih.gov/
   - Comprehensive mental health information and research

2. **World Health Organization (WHO)**
   - https://www.who.int/health-topics/mental-health
   - Global mental health guidelines and resources

3. **Substance Abuse and Mental Health Services Administration (SAMHSA)**
   - https://www.samhsa.gov/
   - US national helpline: 1-800-662-4357

4. **Crisis Resources:**
   - National Suicide Prevention Lifeline: 988
   - Crisis Text Line: Text HOME to 741741

**Note:** Please consult with a mental health professional for personalized guidance."""

    def _is_safe_content(self, content: str) -> bool:
        """Check if content is safe and appropriate"""
        # Define unsafe terms
        unsafe_terms = {
            'suicide', 'self-harm', 'harmful', 'dangerous',
            'illegal', 'unethical', 'controversial'
        }
        
        # Check for unsafe terms
        content_lower = content.lower()
        if any(term in content_lower for term in unsafe_terms):
            return False
            
        # Check content length
        if len(content) < 100:
            return False
            
        # Check for professional tone
        professional_indicators = {
            'research', 'study', 'clinical', 'professional',
            'evidence-based', 'treatment', 'therapy'
        }
        
        return any(indicator in content_lower for indicator in professional_indicators)

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

        except Exception as e:
            logger.warning(f"Error parsing crawler result: {str(e)}")
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
        except Exception as e:
            logger.warning(f"Error checking trusted source for URL {url}: {str(e)}")
            return False

    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """
        Enhanced fallback analysis with topic-specific information.

        When web crawling fails, provides relevant mental health information
        based on the query topic using a knowledge base of common topics.

        Args:
            query: The search query/topic

        Returns:
            Dict with topic-relevant key points, recommendations, and resources
        """
        query_lower = query.lower()

        # Topic-specific knowledge base for common mental health topics
        topic_knowledge = self._get_topic_specific_knowledge(query_lower)

        # Get topic-appropriate recommendations
        recommendations = self._get_topic_recommendations(query_lower)

        # Get relevant resources
        resources = self._get_topic_resources(query_lower)

        return {
            'key_points': topic_knowledge,
            'sources': ['Built-in knowledge base (fallback mode)'],
            'recommendations': recommendations,
            'additional_resources': resources,
            'content_warnings': self._get_content_warnings(query_lower),
            'confidence': 0.5,  # Moderate confidence for topic-specific fallback
            'analysis_method': 'fallback_topic_based',
            'timestamp': datetime.now().isoformat()
        }

    def _get_topic_specific_knowledge(self, query: str) -> List[str]:
        """Get topic-specific knowledge points based on query."""
        knowledge_base = {
            'depression': [
                'Depression is a common mental health condition affecting mood and daily functioning',
                'Symptoms include persistent sadness, loss of interest, and changes in sleep or appetite',
                'Evidence-based treatments include therapy (especially CBT) and medication',
                'Professional help is important; depression is treatable with proper support'
            ],
            'anxiety': [
                'Anxiety disorders involve excessive worry, fear, or nervousness',
                'Common types include GAD, panic disorder, and social anxiety',
                'Cognitive-behavioral therapy (CBT) is highly effective for anxiety',
                'Relaxation techniques, mindfulness, and lifestyle changes can help manage symptoms'
            ],
            'stress': [
                'Stress is the body\'s response to challenges or demands',
                'Chronic stress can impact physical and mental health',
                'Stress management techniques include exercise, meditation, and time management',
                'Professional support may be helpful for persistent or severe stress'
            ],
            'ptsd': [
                'PTSD develops after experiencing or witnessing traumatic events',
                'Symptoms include flashbacks, avoidance, hypervigilance, and mood changes',
                'Evidence-based treatments include trauma-focused therapy and EMDR',
                'Recovery is possible with appropriate professional treatment'
            ],
            'bipolar': [
                'Bipolar disorder involves mood episodes ranging from depression to mania',
                'Treatment typically includes mood stabilizers and psychotherapy',
                'Regular monitoring and medication adherence are crucial',
                'With treatment, people with bipolar disorder can lead fulfilling lives'
            ]
        }

        # Match query to topics
        for topic, points in knowledge_base.items():
            if topic in query:
                return points

        # Default general mental health information
        return [
            'Mental health is an important aspect of overall wellbeing',
            'Professional support is available through therapists, counselors, and psychiatrists',
            'Early intervention and treatment improve outcomes',
            'Recovery and management are possible with appropriate support'
        ]

    def _get_topic_recommendations(self, query: str) -> List[str]:
        """Get topic-appropriate recommendations."""
        if any(word in query for word in ['crisis', 'suicide', 'self-harm', 'emergency']):
            return [
                'Seek immediate help if in crisis - call 988 or 911',
                'Contact emergency services or crisis hotline immediately',
                'Do not wait - professional help is available 24/7'
            ]
        elif any(word in query for word in ['depression', 'anxiety', 'ptsd', 'bipolar']):
            return [
                'Consult with a mental health professional for proper assessment',
                'Evidence-based treatments (therapy and/or medication) are effective',
                'Build a support network of trusted friends, family, or support groups',
                'Practice self-care and healthy coping strategies'
            ]
        else:
            return [
                'Visit reputable mental health websites for reliable information',
                'Consider consulting with mental health professionals',
                'Check with organizations like NAMI or NIMH for resources',
                'Take care of your mental health as you would physical health'
            ]

    def _get_topic_resources(self, query: str) -> List[str]:
        """Get relevant resources based on topic."""
        base_resources = [
            'National Institute of Mental Health (nimh.nih.gov)',
            'National Alliance on Mental Illness (nami.org)',
            'MentalHealth.gov'
        ]

        # Crisis resources for urgent queries
        if any(word in query for word in ['crisis', 'suicide', 'emergency', 'help']):
            return [
                'National Crisis Hotline: 988',
                'Crisis Text Line: Text HOME to 741741',
                'Emergency Services: 911',
                'National Suicide Prevention Lifeline: 1-800-273-8255'
            ] + base_resources

        # Specific condition resources
        if 'anxiety' in query:
            base_resources.insert(0, 'Anxiety and Depression Association of America (adaa.org)')
        elif 'depression' in query:
            base_resources.insert(0, 'Depression and Bipolar Support Alliance (dbsalliance.org)')
        elif 'ptsd' in query:
            base_resources.insert(0, 'National Center for PTSD (ptsd.va.gov)')

        return base_resources

    def _get_content_warnings(self, query: str) -> List[str]:
        """Get appropriate content warnings based on query."""
        warnings = []

        if any(word in query for word in ['suicide', 'self-harm', 'violence']):
            warnings.append('Content may discuss sensitive topics including self-harm')

        if any(word in query for word in ['trauma', 'abuse', 'assault', 'ptsd']):
            warnings.append('Content may discuss traumatic experiences')

        if not warnings:
            warnings.append('Information provided from knowledge base fallback')

        return warnings