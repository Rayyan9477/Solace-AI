"""
Solace-AI Personality Service - LIWC Feature Extraction.
Advanced Linguistic Inquiry and Word Count feature extraction for personality mapping.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog
from ..schemas import PersonalityTrait, OceanScoresDTO, TraitScoreDTO

logger = structlog.get_logger(__name__)


class LIWCProcessorSettings(BaseSettings):
    """LIWC processor configuration."""
    min_word_count: int = Field(default=20, ge=5, le=100)
    max_text_length: int = Field(default=10000, ge=100, le=50000)
    normalize_scores: bool = Field(default=True)
    include_function_words: bool = Field(default=True)
    include_punctuation: bool = Field(default=True)
    confidence_base: float = Field(default=0.4, ge=0.0, le=1.0)
    confidence_word_factor: float = Field(default=0.001, ge=0.0, le=0.01)
    max_confidence: float = Field(default=0.8, ge=0.5, le=1.0)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_LIWC_", env_file=".env", extra="ignore")


@dataclass
class LIWCFeatureVector:
    """Complete LIWC feature vector."""
    feature_id: UUID = field(default_factory=uuid4)
    word_count: int = 0
    words_per_sentence: float = 0.0
    six_letter_words: float = 0.0
    pronouns: float = 0.0
    i_words: float = 0.0
    we_words: float = 0.0
    you_words: float = 0.0
    they_words: float = 0.0
    social: float = 0.0
    family: float = 0.0
    friends: float = 0.0
    positive_emotion: float = 0.0
    negative_emotion: float = 0.0
    anxiety: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    cognitive: float = 0.0
    insight: float = 0.0
    causation: float = 0.0
    discrepancy: float = 0.0
    tentative: float = 0.0
    certainty: float = 0.0
    achievement: float = 0.0
    power: float = 0.0
    reward: float = 0.0
    risk: float = 0.0
    work: float = 0.0
    leisure: float = 0.0
    home: float = 0.0
    money: float = 0.0
    religion: float = 0.0
    death: float = 0.0
    # Expanded LIWC-22 categories
    function: float = 0.0
    pronoun: float = 0.0
    ppron: float = 0.0
    shehe: float = 0.0
    article: float = 0.0
    prep: float = 0.0
    auxverb: float = 0.0
    adverb: float = 0.0
    conj: float = 0.0
    negate: float = 0.0
    verb: float = 0.0
    adj: float = 0.0
    compare: float = 0.0
    interrog: float = 0.0
    number: float = 0.0
    quant: float = 0.0
    affect: float = 0.0
    affiliation: float = 0.0
    differ: float = 0.0
    see: float = 0.0
    hear: float = 0.0
    feel: float = 0.0
    bio: float = 0.0
    body: float = 0.0
    health: float = 0.0
    sexual: float = 0.0
    ingest: float = 0.0
    female: float = 0.0
    male: float = 0.0
    focuspast: float = 0.0
    focuspresent: float = 0.0
    focusfuture: float = 0.0
    motion: float = 0.0
    space: float = 0.0
    time_rel: float = 0.0
    swear: float = 0.0
    netspeak: float = 0.0
    assent: float = 0.0
    nonflu: float = 0.0
    filler: float = 0.0
    question_marks: float = 0.0
    exclamation_marks: float = 0.0
    quotes: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of features."""
        return {k: float(getattr(self, k)) for k in [
            'word_count', 'words_per_sentence', 'six_letter_words', 'pronouns', 'i_words',
            'we_words', 'you_words', 'they_words', 'social', 'family', 'friends',
            'positive_emotion', 'negative_emotion', 'anxiety', 'anger', 'sadness',
            'cognitive', 'insight', 'causation', 'discrepancy', 'tentative', 'certainty',
            'achievement', 'power', 'reward', 'risk', 'work', 'leisure', 'home', 'money',
            'religion', 'death',
            'function', 'pronoun', 'ppron', 'shehe', 'article', 'prep', 'auxverb',
            'adverb', 'conj', 'negate', 'verb', 'adj', 'compare', 'interrog',
            'number', 'quant', 'affect', 'affiliation', 'differ',
            'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest',
            'female', 'male', 'focuspast', 'focuspresent', 'focusfuture',
            'motion', 'space', 'time_rel',
            'swear', 'netspeak', 'assent', 'nonflu', 'filler',
            'question_marks', 'exclamation_marks', 'quotes']}


class LIWCDictionary:
    """LIWC word dictionary for feature extraction."""
    _CATEGORIES = {
        'i_words': frozenset({'i', 'me', 'my', 'mine', 'myself'}),
        'we_words': frozenset({'we', 'us', 'our', 'ours', 'ourselves'}),
        'you_words': frozenset({'you', 'your', 'yours', 'yourself', 'yourselves'}),
        'they_words': frozenset({'they', 'them', 'their', 'theirs', 'themselves'}),
        'social': frozenset({'talk', 'share', 'meet', 'call', 'visit', 'listen', 'communicate', 'discuss', 'tell', 'ask'}),
        'family': frozenset({'family', 'mom', 'dad', 'mother', 'father', 'parent', 'brother', 'sister', 'son', 'daughter'}),
        'friends': frozenset({'friend', 'buddy', 'pal', 'companion', 'mate', 'neighbor', 'colleague', 'peer'}),
        'positive_emotion': frozenset({'happy', 'love', 'good', 'great', 'wonderful', 'joy', 'exciting', 'beautiful', 'hope', 'kind', 'grateful', 'blessed', 'amazing', 'fantastic', 'excellent', 'awesome', 'delighted', 'pleased', 'thankful', 'appreciate', 'enjoy', 'caring', 'warm', 'peaceful', 'calm', 'content', 'proud'}),
        'negative_emotion': frozenset({'sad', 'angry', 'hate', 'bad', 'terrible', 'awful', 'hurt', 'pain', 'fear', 'worry', 'anxious', 'upset', 'frustrated', 'disappointed', 'depressed', 'lonely', 'miserable', 'guilty', 'ashamed'}),
        'anxiety': frozenset({'worried', 'anxious', 'nervous', 'afraid', 'scared', 'fear', 'panic', 'stress', 'tense'}),
        'anger': frozenset({'angry', 'mad', 'furious', 'rage', 'hate', 'hostile', 'annoyed', 'irritated', 'frustrated'}),
        'sadness': frozenset({'sad', 'unhappy', 'depressed', 'lonely', 'grief', 'sorrow', 'miserable', 'hopeless', 'cry'}),
        'cognitive': frozenset({'think', 'know', 'believe', 'understand', 'consider', 'realize', 'reason', 'decide', 'analyze', 'evaluate', 'assess', 'determine', 'conclude', 'assume', 'suppose', 'expect', 'remember', 'forget'}),
        'insight': frozenset({'realize', 'understand', 'discover', 'learn', 'insight', 'aware', 'recognize', 'comprehend'}),
        'causation': frozenset({'because', 'cause', 'effect', 'hence', 'therefore', 'thus', 'consequently', 'result'}),
        'discrepancy': frozenset({'should', 'would', 'could', 'ought', 'need', 'want', 'wish', 'hope', 'expect'}),
        'tentative': frozenset({'maybe', 'perhaps', 'might', 'possibly', 'probably', 'guess', 'seem', 'appear', 'unsure'}),
        'certainty': frozenset({'always', 'never', 'definitely', 'certainly', 'absolutely', 'clearly', 'sure', 'obvious'}),
        'achievement': frozenset({'achieve', 'success', 'goal', 'accomplish', 'complete', 'win', 'effort', 'improve', 'progress', 'excel', 'master', 'overcome', 'succeed', 'productive', 'efficient', 'effective', 'competent'}),
        'power': frozenset({'power', 'control', 'lead', 'authority', 'dominate', 'command', 'influence', 'strength'}),
        'reward': frozenset({'reward', 'prize', 'benefit', 'gain', 'earn', 'deserve', 'bonus', 'incentive'}),
        'risk': frozenset({'risk', 'danger', 'threat', 'hazard', 'unsafe', 'vulnerable', 'uncertain', 'gamble'}),
        'work': frozenset({'work', 'job', 'career', 'office', 'boss', 'employee', 'business', 'professional', 'project'}),
        'leisure': frozenset({'relax', 'vacation', 'hobby', 'fun', 'play', 'game', 'entertainment', 'movie', 'music'}),
        'home': frozenset({'home', 'house', 'apartment', 'room', 'kitchen', 'bedroom', 'yard', 'neighbor', 'live'}),
        'money': frozenset({'money', 'cash', 'pay', 'cost', 'price', 'buy', 'sell', 'income', 'expense', 'budget', 'debt'}),
        'religion': frozenset({'god', 'church', 'pray', 'faith', 'spiritual', 'soul', 'heaven', 'bless', 'worship'}),
        'death': frozenset({'death', 'die', 'dead', 'kill', 'funeral', 'grave', 'loss', 'grief', 'mourn'}),
        # --- Expanded LIWC-22 categories ---
        # Function words
        'function': frozenset({'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall', 'may', 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet'}),
        'pronoun': frozenset({'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves', 'it', 'its', 'itself'}),
        'ppron': frozenset({'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves'}),
        'shehe': frozenset({'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself'}),
        'article': frozenset({'a', 'an', 'the'}),
        'prep': frozenset({'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'about', 'against', 'among', 'around', 'toward', 'upon', 'within', 'without', 'along', 'across', 'behind', 'beyond'}),
        'auxverb': frozenset({'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could', 'must'}),
        'adverb': frozenset({'very', 'really', 'quite', 'just', 'also', 'often', 'always', 'never', 'sometimes', 'usually', 'already', 'still', 'even', 'almost', 'enough', 'too', 'well', 'here', 'there', 'now', 'then', 'soon', 'quickly', 'slowly', 'easily', 'actually', 'probably', 'simply', 'hardly', 'nearly'}),
        'conj': frozenset({'and', 'but', 'or', 'nor', 'so', 'yet', 'because', 'although', 'though', 'while', 'if', 'unless', 'until', 'since', 'whether', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'meanwhile'}),
        'negate': frozenset({'no', 'not', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none', 'cannot', 'without'}),
        'verb': frozenset({'go', 'get', 'make', 'take', 'come', 'see', 'know', 'think', 'say', 'give', 'find', 'tell', 'ask', 'use', 'try', 'leave', 'call', 'keep', 'let', 'begin', 'seem', 'help', 'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'bring', 'happen', 'write', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain'}),
        'adj': frozenset({'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'important', 'young', 'early', 'hard', 'major', 'better', 'best', 'free', 'strong', 'real', 'sure', 'true', 'whole', 'clear', 'full', 'easy', 'able', 'likely', 'simple', 'difficult', 'bad', 'happy', 'sad', 'nice', 'beautiful', 'possible', 'available', 'special', 'certain'}),
        'compare': frozenset({'more', 'less', 'most', 'least', 'better', 'best', 'worse', 'worst', 'greater', 'greatest', 'larger', 'largest', 'smaller', 'smallest', 'higher', 'highest', 'lower', 'lowest', 'than', 'as', 'like', 'similar', 'different', 'equal', 'same'}),
        'interrog': frozenset({'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose'}),
        'number': frozenset({'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'million', 'billion', 'first', 'second', 'third', 'half', 'quarter', 'dozen', 'once', 'twice'}),
        'quant': frozenset({'all', 'every', 'each', 'some', 'any', 'many', 'much', 'few', 'several', 'most', 'enough', 'plenty', 'both', 'neither', 'either', 'none', 'various', 'numerous'}),
        'affect': frozenset({'happy', 'love', 'good', 'great', 'joy', 'hope', 'kind', 'care', 'sad', 'angry', 'hate', 'bad', 'hurt', 'pain', 'fear', 'worry', 'upset', 'grateful', 'proud', 'excited', 'anxious', 'lonely', 'guilty'}),
        # Drives
        'affiliation': frozenset({'ally', 'bond', 'club', 'cohort', 'collaborate', 'companion', 'fellowship', 'group', 'join', 'member', 'partner', 'team', 'together', 'unite', 'belong', 'community', 'cooperate', 'mutual', 'share', 'collective'}),
        # Cognition expanded
        'differ': frozenset({'but', 'however', 'although', 'though', 'rather', 'instead', 'whereas', 'otherwise', 'except', 'unlike', 'contrast', 'distinguish', 'differ', 'difference', 'alternatively'}),
        # Perception
        'see': frozenset({'see', 'saw', 'seen', 'look', 'watch', 'view', 'observe', 'notice', 'stare', 'glance', 'gaze', 'visible', 'appear', 'sight', 'witness'}),
        'hear': frozenset({'hear', 'heard', 'listen', 'sound', 'loud', 'quiet', 'noise', 'voice', 'ring', 'shout', 'whisper', 'silent', 'echo', 'music', 'song'}),
        'feel': frozenset({'feel', 'felt', 'touch', 'warm', 'cold', 'hot', 'soft', 'hard', 'rough', 'smooth', 'sharp', 'pain', 'pressure', 'comfort', 'sensation'}),
        # Biological
        'bio': frozenset({'eat', 'drink', 'sleep', 'wake', 'breath', 'breathe', 'blood', 'body', 'health', 'sick', 'pain', 'heart', 'brain', 'stomach', 'muscle', 'bone', 'skin', 'alive', 'born', 'pregnant'}),
        'body': frozenset({'hand', 'head', 'face', 'eye', 'arm', 'leg', 'foot', 'heart', 'brain', 'stomach', 'back', 'shoulder', 'chest', 'skin', 'bone', 'muscle', 'finger', 'hair', 'mouth', 'nose'}),
        'health': frozenset({'health', 'healthy', 'sick', 'ill', 'disease', 'doctor', 'hospital', 'medicine', 'symptom', 'therapy', 'treatment', 'cure', 'diagnose', 'clinic', 'nurse', 'medical', 'patient', 'heal', 'recovery', 'wellness'}),
        'sexual': frozenset({'sex', 'sexual', 'intimate', 'romance', 'romantic', 'kiss', 'attraction', 'desire', 'passion', 'lover', 'sensual'}),
        'ingest': frozenset({'eat', 'drink', 'food', 'meal', 'cook', 'taste', 'hungry', 'thirsty', 'appetite', 'swallow', 'chew', 'sip', 'bite', 'digest', 'breakfast', 'lunch', 'dinner', 'snack', 'coffee', 'tea'}),
        # Social expanded
        'female': frozenset({'she', 'her', 'hers', 'herself', 'woman', 'women', 'girl', 'mother', 'daughter', 'sister', 'wife', 'aunt', 'grandmother', 'lady', 'female'}),
        'male': frozenset({'he', 'him', 'his', 'himself', 'man', 'men', 'boy', 'father', 'son', 'brother', 'husband', 'uncle', 'grandfather', 'gentleman', 'male'}),
        # Time orientation
        'focuspast': frozenset({'was', 'were', 'had', 'did', 'ago', 'yesterday', 'last', 'used', 'before', 'once', 'previously', 'former', 'earlier', 'past', 'remembered'}),
        'focuspresent': frozenset({'is', 'are', 'am', 'now', 'today', 'currently', 'present', 'being', 'right', 'here', 'this', 'these', 'ongoing', 'existing', 'happening'}),
        'focusfuture': frozenset({'will', 'going', 'gonna', 'tomorrow', 'soon', 'later', 'next', 'future', 'plan', 'intend', 'expect', 'hope', 'ahead', 'upcoming', 'eventually'}),
        # Relativity
        'motion': frozenset({'go', 'walk', 'run', 'move', 'come', 'leave', 'arrive', 'travel', 'drive', 'fly', 'follow', 'approach', 'return', 'step', 'climb', 'fall', 'jump', 'rush', 'hurry', 'wander'}),
        'space': frozenset({'here', 'there', 'where', 'up', 'down', 'in', 'out', 'above', 'below', 'near', 'far', 'close', 'around', 'between', 'inside', 'outside', 'behind', 'front', 'left', 'right', 'top', 'bottom', 'side', 'place', 'area'}),
        'time_rel': frozenset({'time', 'when', 'then', 'now', 'before', 'after', 'during', 'while', 'until', 'since', 'always', 'never', 'often', 'sometimes', 'soon', 'late', 'early', 'long', 'moment', 'minute', 'hour', 'day', 'week', 'month', 'year'}),
        # Informal language
        'swear': frozenset({'damn', 'hell', 'crap', 'suck', 'stupid', 'dumb', 'jerk', 'idiot', 'fool', 'bloody'}),
        'netspeak': frozenset({'lol', 'omg', 'brb', 'btw', 'idk', 'imo', 'tbh', 'smh', 'fwiw', 'iirc', 'afaik', 'nvm', 'thx', 'pls', 'plz', 'bc', 'ur', 'u', 'r', 'gonna', 'wanna', 'gotta', 'kinda', 'sorta'}),
        'assent': frozenset({'yes', 'yeah', 'yep', 'yup', 'ok', 'okay', 'sure', 'right', 'true', 'agreed', 'absolutely', 'definitely', 'indeed', 'exactly', 'correct'}),
        'nonflu': frozenset({'uh', 'um', 'er', 'ah', 'hmm', 'hm', 'umm', 'uhh', 'ehm', 'erm'}),
        'filler': frozenset({'like', 'basically', 'literally', 'actually', 'honestly', 'seriously', 'obviously', 'totally', 'simply', 'essentially', 'really', 'just', 'well', 'so', 'anyway', 'whatever', 'somehow'}),
    }

    def get_categories(self) -> dict[str, frozenset[str]]:
        return self._CATEGORIES


class FeatureExtractor:
    """Extracts LIWC features from text."""
    _SENTENCE_PATTERN = re.compile(r'[.!?]+')
    _WORD_PATTERN = re.compile(r'\b[a-z]+\b')

    def __init__(self, dictionary: LIWCDictionary | None = None) -> None:
        self._dictionary = dictionary or LIWCDictionary()
        self._categories = self._dictionary.get_categories()

    def extract(self, text: str) -> LIWCFeatureVector:
        """Extract LIWC features from text."""
        words = self._WORD_PATTERN.findall(text.lower())
        wc = len(words) or 1
        sentences = self._SENTENCE_PATTERN.split(text)
        sc = max(1, len([s for s in sentences if s.strip()]))
        cc = {n: sum(1 for w in words if w in ws) for n, ws in self._categories.items()}
        pronouns = cc['i_words'] + cc['we_words'] + cc['you_words'] + cc['they_words']
        six_letter = sum(1 for w in words if len(w) >= 6)
        return LIWCFeatureVector(
            word_count=len(words), words_per_sentence=len(words) / sc, six_letter_words=six_letter / wc * 100,
            pronouns=pronouns / wc * 100, i_words=cc['i_words'] / wc * 100, we_words=cc['we_words'] / wc * 100,
            you_words=cc['you_words'] / wc * 100, they_words=cc['they_words'] / wc * 100, social=cc['social'] / wc * 100,
            family=cc['family'] / wc * 100, friends=cc['friends'] / wc * 100, positive_emotion=cc['positive_emotion'] / wc * 100,
            negative_emotion=cc['negative_emotion'] / wc * 100, anxiety=cc['anxiety'] / wc * 100, anger=cc['anger'] / wc * 100,
            sadness=cc['sadness'] / wc * 100, cognitive=cc['cognitive'] / wc * 100, insight=cc['insight'] / wc * 100,
            causation=cc['causation'] / wc * 100, discrepancy=cc['discrepancy'] / wc * 100, tentative=cc['tentative'] / wc * 100,
            certainty=cc['certainty'] / wc * 100, achievement=cc['achievement'] / wc * 100, power=cc['power'] / wc * 100,
            reward=cc['reward'] / wc * 100, risk=cc['risk'] / wc * 100, work=cc['work'] / wc * 100, leisure=cc['leisure'] / wc * 100,
            home=cc['home'] / wc * 100, money=cc['money'] / wc * 100, religion=cc['religion'] / wc * 100, death=cc['death'] / wc * 100,
            function=cc['function'] / wc * 100, pronoun=cc['pronoun'] / wc * 100, ppron=cc['ppron'] / wc * 100,
            shehe=cc['shehe'] / wc * 100, article=cc['article'] / wc * 100, prep=cc['prep'] / wc * 100,
            auxverb=cc['auxverb'] / wc * 100, adverb=cc['adverb'] / wc * 100, conj=cc['conj'] / wc * 100,
            negate=cc['negate'] / wc * 100, verb=cc['verb'] / wc * 100, adj=cc['adj'] / wc * 100,
            compare=cc['compare'] / wc * 100, interrog=cc['interrog'] / wc * 100, number=cc['number'] / wc * 100,
            quant=cc['quant'] / wc * 100, affect=cc['affect'] / wc * 100, affiliation=cc['affiliation'] / wc * 100,
            differ=cc['differ'] / wc * 100, see=cc['see'] / wc * 100, hear=cc['hear'] / wc * 100,
            feel=cc['feel'] / wc * 100, bio=cc['bio'] / wc * 100, body=cc['body'] / wc * 100,
            health=cc['health'] / wc * 100, sexual=cc['sexual'] / wc * 100, ingest=cc['ingest'] / wc * 100,
            female=cc['female'] / wc * 100, male=cc['male'] / wc * 100, focuspast=cc['focuspast'] / wc * 100,
            focuspresent=cc['focuspresent'] / wc * 100, focusfuture=cc['focusfuture'] / wc * 100,
            motion=cc['motion'] / wc * 100, space=cc['space'] / wc * 100, time_rel=cc['time_rel'] / wc * 100,
            swear=cc['swear'] / wc * 100, netspeak=cc['netspeak'] / wc * 100, assent=cc['assent'] / wc * 100,
            nonflu=cc['nonflu'] / wc * 100, filler=cc['filler'] / wc * 100,
            question_marks=text.count('?') / wc * 100, exclamation_marks=text.count('!') / wc * 100,
            quotes=text.count('"') / wc * 100)


class PersonalityMapper:
    """Maps LIWC features to OCEAN personality scores."""

    def map_to_ocean(self, f: LIWCFeatureVector) -> dict[PersonalityTrait, float]:
        """Map LIWC features to OCEAN scores."""
        clamp = lambda v: max(0.0, min(1.0, v))
        return {
            PersonalityTrait.OPENNESS: clamp(0.5 + f.insight * 0.15 + f.cognitive * 0.08 + f.six_letter_words * 0.005 + f.question_marks * 0.1 - f.certainty * 0.05),
            PersonalityTrait.CONSCIENTIOUSNESS: clamp(0.5 + f.achievement * 0.15 + f.work * 0.1 + f.certainty * 0.08 - f.tentative * 0.08 - f.discrepancy * 0.05),
            PersonalityTrait.EXTRAVERSION: clamp(0.5 + f.social * 0.12 + f.we_words * 0.1 + f.friends * 0.15 + f.positive_emotion * 0.08 + f.exclamation_marks * 0.1 - f.i_words * 0.03),
            PersonalityTrait.AGREEABLENESS: clamp(0.5 + f.positive_emotion * 0.1 + f.social * 0.08 + f.family * 0.1 + f.friends * 0.1 - f.negative_emotion * 0.08 - f.anger * 0.12 - f.power * 0.05),
            PersonalityTrait.NEUROTICISM: clamp(0.5 + f.negative_emotion * 0.12 + f.anxiety * 0.15 + f.sadness * 0.12 + f.anger * 0.08 + f.tentative * 0.06 - f.positive_emotion * 0.08 - f.certainty * 0.05),
        }


class LIWCProcessor:
    """Main LIWC processor for personality analysis."""

    def __init__(self, settings: LIWCProcessorSettings | None = None) -> None:
        self._settings = settings or LIWCProcessorSettings()
        self._extractor = FeatureExtractor()
        self._mapper = PersonalityMapper()
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("liwc_processor_initialized")

    async def process(self, text: str) -> OceanScoresDTO:
        """Process text and return OCEAN scores."""
        if not text or len(text.strip()) < self._settings.min_word_count:
            logger.warning("text_too_short_for_liwc", length=len(text))
            return self._neutral_scores(confidence=0.2)
        features = self._extractor.extract(text[:self._settings.max_text_length])
        if features.word_count < self._settings.min_word_count:
            return self._neutral_scores(confidence=0.3)
        scores = self._mapper.map_to_ocean(features)
        conf = min(self._settings.max_confidence, self._settings.confidence_base + features.word_count * self._settings.confidence_word_factor)
        margin = 0.15 * (1 - conf)
        trait_scores = []
        evidence_funcs = {
            PersonalityTrait.OPENNESS: lambda f: (['high_insight'] if f.insight > 1.0 else []) + (['cognitive_processing'] if f.cognitive > 2.0 else []) + (['complex_vocabulary'] if f.six_letter_words > 15 else []),
            PersonalityTrait.CONSCIENTIOUSNESS: lambda f: (['achievement_focus'] if f.achievement > 1.0 else []) + (['work_oriented'] if f.work > 1.5 else []) + (['decisive_language'] if f.certainty > 1.0 else []),
            PersonalityTrait.EXTRAVERSION: lambda f: (['social_language'] if f.social > 2.0 else []) + (['collective_focus'] if f.we_words > 1.5 else []) + (['positive_affect'] if f.positive_emotion > 2.0 else []),
            PersonalityTrait.AGREEABLENESS: lambda f: (['warm_language'] if f.positive_emotion > 2.0 else []) + (['relationship_focus'] if f.family > 0.5 or f.friends > 0.5 else []) + (['low_hostility'] if f.anger < 0.5 and f.negative_emotion < 1.0 else []),
            PersonalityTrait.NEUROTICISM: lambda f: (['anxiety_language'] if f.anxiety > 0.5 else []) + (['sadness_expressed'] if f.sadness > 0.5 else []) + (['negative_affect'] if f.negative_emotion > 2.0 else []),
        }
        for trait, value in scores.items():
            evidence = evidence_funcs[trait](features)
            trait_scores.append(TraitScoreDTO(trait=trait, value=value, confidence_lower=max(0.0, value - margin), confidence_upper=min(1.0, value + margin), sample_count=1, evidence_markers=evidence[:3]))
        return OceanScoresDTO(openness=scores[PersonalityTrait.OPENNESS], conscientiousness=scores[PersonalityTrait.CONSCIENTIOUSNESS], extraversion=scores[PersonalityTrait.EXTRAVERSION], agreeableness=scores[PersonalityTrait.AGREEABLENESS], neuroticism=scores[PersonalityTrait.NEUROTICISM], overall_confidence=conf, trait_scores=trait_scores)

    async def extract_features(self, text: str) -> LIWCFeatureVector:
        return self._extractor.extract(text[:self._settings.max_text_length])

    def _neutral_scores(self, confidence: float = 0.3) -> OceanScoresDTO:
        return OceanScoresDTO(openness=0.5, conscientiousness=0.5, extraversion=0.5, agreeableness=0.5, neuroticism=0.5, overall_confidence=confidence, trait_scores=[])

    async def shutdown(self) -> None:
        self._initialized = False
        logger.info("liwc_processor_shutdown")
