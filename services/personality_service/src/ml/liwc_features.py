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
            'religion', 'death', 'question_marks', 'exclamation_marks', 'quotes']}


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
