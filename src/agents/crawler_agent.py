# Updated safe_crawl to use get_results from the SerpAPIWrapper

import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from config.settings import AppConfig
import re
import logging
from urllib.parse import urlparse

class CrawlerAgent:
    def __init__(self, config: dict):
        self.config = config
        self.search = SerpAPIWrapper()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MentalHealthBot/1.0 (+https://example.com/bot)',
            'Accept-Language': 'en-US,en;q=0.5'
        })
        
        self.tools = [
            Tool(
                name="WebSearch",
                func=self.safe_crawl,
                description="Searches web for mental health resources"
            )
        ]
        self.logger = logging.getLogger(__name__)

    def safe_crawl(self, query: str) -> str:
        """Safe web crawling with content validation"""
        try:
            results_dict = self.search.get_results(query)
            results = results_dict.get("organic_results", [])[:self.config['max_results']]
            return self._process_results(results)
        except Exception as e:
            self.logger.error(f"Crawler error: {str(e)}")
            return "I couldn't find additional resources right now. Please try again later."

    def _process_results(self, results) -> str:
        content = []
        for result in results:
            if 'link' in result:
                if self._is_valid_url(result['link']):
                    page_content = self._fetch_safe_content(result['link'])
                    content.append(f"**Source:** {result.get('title', result['link'])}\n{page_content}")
        return "\n\n".join(content[:3])

    def _is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc in {'example.com', 'localhost'}:
            return False
        return parsed.scheme in {'http', 'https'}

    def _fetch_safe_content(self, url: str, depth=0) -> str:
        if depth >= self.config['max_depth']:
            return ""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in ['script', 'style', 'nav', 'footer']:
                for element in soup.find_all(tag):
                    element.decompose()
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text[:500] + "..."
        except Exception as e:
            self.logger.warning(f"Failed to fetch {url}: {str(e)}")
            return ""

    def validate_content(self, text: str) -> bool:
        blacklist = {'violence', 'suicide', 'self-harm'}
        return not any(word in text.lower() for word in blacklist)