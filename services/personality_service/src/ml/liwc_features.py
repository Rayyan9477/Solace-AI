"""
Solace-AI Personality Service - LIWC Feature Extraction.
Advanced Linguistic Inquiry and Word Count feature extraction for personality mapping.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import PersonalityTrait, AssessmentSource, OceanScoresDTO, TraitScoreDTO

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
class LIWCCategory:
    """LIWC category with word lists."""
    name: str
    words: frozenset[str]
    weight: float = 1.0


@dataclass
class LIWCFeatureVector:
    """Complete LIWC feature vector."""
    feature_id: UUID = field(default_factory=uuid4)
    word_count: int = 0
    words_per_sentence: float = 0.0
    six_letter_words: float = 0.0
    # Function words
    pronouns: float = 0.0
    i_words: float = 0.0
    we_words: float = 0.0
    you_words: float = 0.0
    they_words: float = 0.0
    # Social processes
    social: float = 0.0
    family: float = 0.0
    friends: float = 0.0
    # Affective processes
    positive_emotion: float = 0.0
    negative_emotion: float = 0.0
    anxiety: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    # Cognitive processes
    cognitive: float = 0.0
    insight: float = 0.0
    causation: float = 0.0
    discrepancy: float = 0.0
    tentative: float = 0.0
    certainty: float = 0.0
    # Drives
    achievement: float = 0.0
    power: float = 0.0
    reward: float = 0.0
    risk: float = 0.0
    # Personal concerns
    work: float = 0.0
    leisure: float = 0.0
    home: float = 0.0
    money: float = 0.0
    religion: float = 0.0
    death: float = 0.0
    # Punctuation
    question_marks: float = 0.0
    exclamation_marks: float = 0.0
    quotes: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of features."""
        return {
            'word_count': float(self.word_count),
            'words_per_sentence': self.words_per_sentence,
            'six_letter_words': self.six_letter_words,
            'pronouns': self.pronouns,
            'i_words': self.i_words,
            'we_words': self.we_words,
            'you_words': self.you_words,
            'they_words': self.they_words,
            'social': self.social,
            'family': self.family,
            'friends': self.friends,
            'positive_emotion': self.positive_emotion,
            'negative_emotion': self.negative_emotion,
            'anxiety': self.anxiety,
            'anger': self.anger,
            'sadness': self.sadness,
            'cognitive': self.cognitive,
            'insight': self.insight,
            'causation': self.causation,
            'discrepancy': self.discrepancy,
            'tentative': self.tentative,
            'certainty': self.certainty,
            'achievement': self.achievement,
            'power': self.power,
            'reward': self.reward,
            'risk': self.risk,
            'work': self.work,
            'leisure': self.leisure,
            'home': self.home,
            'money': self.money,
            'religion': self.religion,
            'death': self.death,
            'question_marks': self.question_marks,
            'exclamation_marks': self.exclamation_marks,
            'quotes': self.quotes,
        }


class LIWCDictionary:
    """LIWC word dictionary for feature extraction."""
    _I_WORDS = frozenset({'i', 'me', 'my', 'mine', 'myself'})
    _WE_WORDS = frozenset({'we', 'us', 'our', 'ours', 'ourselves'})
    _YOU_WORDS = frozenset({'you', 'your', 'yours', 'yourself', 'yourselves'})
    _THEY_WORDS = frozenset({'they', 'them', 'their', 'theirs', 'themselves'})
    _SOCIAL = frozenset({'talk', 'share', 'meet', 'call', 'visit', 'listen', 'communicate', 'discuss', 'tell', 'ask'})
    _FAMILY = frozenset({'family', 'mom', 'dad', 'mother', 'father', 'parent', 'brother', 'sister', 'son', 'daughter'})
    _FRIENDS = frozenset({'friend', 'buddy', 'pal', 'companion', 'mate', 'neighbor', 'colleague', 'peer'})
    _POSITIVE = frozenset({
        'happy', 'love', 'good', 'great', 'wonderful', 'joy', 'exciting', 'beautiful', 'hope', 'kind',
        'grateful', 'blessed', 'amazing', 'fantastic', 'excellent', 'awesome', 'delighted', 'pleased',
        'thankful', 'appreciate', 'enjoy', 'caring', 'warm', 'peaceful', 'calm', 'content', 'proud',
    })
    _NEGATIVE = frozenset({
        'sad', 'angry', 'hate', 'bad', 'terrible', 'awful', 'hurt', 'pain', 'fear', 'worry', 'anxious',
        'upset', 'frustrated', 'disappointed', 'depressed', 'lonely', 'miserable', 'guilty', 'ashamed',
    })
    _ANXIETY = frozenset({'worried', 'anxious', 'nervous', 'afraid', 'scared', 'fear', 'panic', 'stress', 'tense'})
    _ANGER = frozenset({'angry', 'mad', 'furious', 'rage', 'hate', 'hostile', 'annoyed', 'irritated', 'frustrated'})
    _SADNESS = frozenset({'sad', 'unhappy', 'depressed', 'lonely', 'grief', 'sorrow', 'miserable', 'hopeless', 'cry'})
    _COGNITIVE = frozenset({
        'think', 'know', 'believe', 'understand', 'consider', 'realize', 'reason', 'decide', 'analyze',
        'evaluate', 'assess', 'determine', 'conclude', 'assume', 'suppose', 'expect', 'remember', 'forget',
    })
    _INSIGHT = frozenset({'realize', 'understand', 'discover', 'learn', 'insight', 'aware', 'recognize', 'comprehend'})
    _CAUSATION = frozenset({'because', 'cause', 'effect', 'hence', 'therefore', 'thus', 'consequently', 'result'})
    _DISCREPANCY = frozenset({'should', 'would', 'could', 'ought', 'need', 'want', 'wish', 'hope', 'expect'})
    _TENTATIVE = frozenset({'maybe', 'perhaps', 'might', 'possibly', 'probably', 'guess', 'seem', 'appear', 'unsure'})
    _CERTAINTY = frozenset({'always', 'never', 'definitely', 'certainly', 'absolutely', 'clearly', 'sure', 'obvious'})
    _ACHIEVEMENT = frozenset({
        'achieve', 'success', 'goal', 'accomplish', 'complete', 'win', 'effort', 'improve', 'progress',
        'excel', 'master', 'overcome', 'succeed', 'productive', 'efficient', 'effective', 'competent',
    })
    _POWER = frozenset({'power', 'control', 'lead', 'authority', 'dominate', 'command', 'influence', 'strength'})
    _REWARD = frozenset({'reward', 'prize', 'benefit', 'gain', 'earn', 'deserve', 'bonus', 'incentive'})
    _RISK = frozenset({'risk', 'danger', 'threat', 'hazard', 'unsafe', 'vulnerable', 'uncertain', 'gamble'})
    _WORK = frozenset({'work', 'job', 'career', 'office', 'boss', 'employee', 'business', 'professional', 'project'})
    _LEISURE = frozenset({'relax', 'vacation', 'hobby', 'fun', 'play', 'game', 'entertainment', 'movie', 'music'})
    _HOME = frozenset({'home', 'house', 'apartment', 'room', 'kitchen', 'bedroom', 'yard', 'neighbor', 'live'})
    _MONEY = frozenset({'money', 'cash', 'pay', 'cost', 'price', 'buy', 'sell', 'income', 'expense', 'budget', 'debt'})
    _RELIGION = frozenset({'god', 'church', 'pray', 'faith', 'spiritual', 'soul', 'heaven', 'bless', 'worship'})
    _DEATH = frozenset({'death', 'die', 'dead', 'kill', 'funeral', 'grave', 'loss', 'grief', 'mourn'})

    def get_categories(self) -> dict[str, frozenset[str]]:
        """Get all LIWC categories."""
        return {
            'i_words': self._I_WORDS,
            'we_words': self._WE_WORDS,
            'you_words': self._YOU_WORDS,
            'they_words': self._THEY_WORDS,
            'social': self._SOCIAL,
            'family': self._FAMILY,
            'friends': self._FRIENDS,
            'positive_emotion': self._POSITIVE,
            'negative_emotion': self._NEGATIVE,
            'anxiety': self._ANXIETY,
            'anger': self._ANGER,
            'sadness': self._SADNESS,
            'cognitive': self._COGNITIVE,
            'insight': self._INSIGHT,
            'causation': self._CAUSATION,
            'discrepancy': self._DISCREPANCY,
            'tentative': self._TENTATIVE,
            'certainty': self._CERTAINTY,
            'achievement': self._ACHIEVEMENT,
            'power': self._POWER,
            'reward': self._REWARD,
            'risk': self._RISK,
            'work': self._WORK,
            'leisure': self._LEISURE,
            'home': self._HOME,
            'money': self._MONEY,
            'religion': self._RELIGION,
            'death': self._DEATH,
        }


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
        word_count = len(words) or 1
        sentences = self._SENTENCE_PATTERN.split(text)
        sentence_count = max(1, len([s for s in sentences if s.strip()]))
        category_counts = {name: self._count_matches(words, word_set) for name, word_set in self._categories.items()}
        pronouns = (
            category_counts['i_words'] + category_counts['we_words'] +
            category_counts['you_words'] + category_counts['they_words']
        )
        six_letter_count = sum(1 for w in words if len(w) >= 6)
        return LIWCFeatureVector(
            word_count=len(words),
            words_per_sentence=len(words) / sentence_count,
            six_letter_words=six_letter_count / word_count * 100,
            pronouns=pronouns / word_count * 100,
            i_words=category_counts['i_words'] / word_count * 100,
            we_words=category_counts['we_words'] / word_count * 100,
            you_words=category_counts['you_words'] / word_count * 100,
            they_words=category_counts['they_words'] / word_count * 100,
            social=category_counts['social'] / word_count * 100,
            family=category_counts['family'] / word_count * 100,
            friends=category_counts['friends'] / word_count * 100,
            positive_emotion=category_counts['positive_emotion'] / word_count * 100,
            negative_emotion=category_counts['negative_emotion'] / word_count * 100,
            anxiety=category_counts['anxiety'] / word_count * 100,
            anger=category_counts['anger'] / word_count * 100,
            sadness=category_counts['sadness'] / word_count * 100,
            cognitive=category_counts['cognitive'] / word_count * 100,
            insight=category_counts['insight'] / word_count * 100,
            causation=category_counts['causation'] / word_count * 100,
            discrepancy=category_counts['discrepancy'] / word_count * 100,
            tentative=category_counts['tentative'] / word_count * 100,
            certainty=category_counts['certainty'] / word_count * 100,
            achievement=category_counts['achievement'] / word_count * 100,
            power=category_counts['power'] / word_count * 100,
            reward=category_counts['reward'] / word_count * 100,
            risk=category_counts['risk'] / word_count * 100,
            work=category_counts['work'] / word_count * 100,
            leisure=category_counts['leisure'] / word_count * 100,
            home=category_counts['home'] / word_count * 100,
            money=category_counts['money'] / word_count * 100,
            religion=category_counts['religion'] / word_count * 100,
            death=category_counts['death'] / word_count * 100,
            question_marks=text.count('?') / word_count * 100,
            exclamation_marks=text.count('!') / word_count * 100,
            quotes=(text.count('"') + text.count("'")) / word_count * 100,
        )

    def _count_matches(self, words: list[str], word_set: frozenset[str]) -> int:
        """Count word matches."""
        return sum(1 for w in words if w in word_set)


class PersonalityMapper:
    """Maps LIWC features to OCEAN personality scores."""

    def map_to_ocean(self, features: LIWCFeatureVector) -> dict[PersonalityTrait, float]:
        """Map LIWC features to OCEAN scores."""
        openness = self._compute_openness(features)
        conscientiousness = self._compute_conscientiousness(features)
        extraversion = self._compute_extraversion(features)
        agreeableness = self._compute_agreeableness(features)
        neuroticism = self._compute_neuroticism(features)
        return {
            PersonalityTrait.OPENNESS: openness,
            PersonalityTrait.CONSCIENTIOUSNESS: conscientiousness,
            PersonalityTrait.EXTRAVERSION: extraversion,
            PersonalityTrait.AGREEABLENESS: agreeableness,
            PersonalityTrait.NEUROTICISM: neuroticism,
        }

    def _compute_openness(self, f: LIWCFeatureVector) -> float:
        """Compute openness from features."""
        base = 0.5
        base += f.insight * 0.15
        base += f.cognitive * 0.08
        base += f.six_letter_words * 0.005
        base += f.question_marks * 0.1
        base -= f.certainty * 0.05
        return self._clamp(base)

    def _compute_conscientiousness(self, f: LIWCFeatureVector) -> float:
        """Compute conscientiousness from features."""
        base = 0.5
        base += f.achievement * 0.15
        base += f.work * 0.1
        base += f.certainty * 0.08
        base -= f.tentative * 0.08
        base -= f.discrepancy * 0.05
        return self._clamp(base)

    def _compute_extraversion(self, f: LIWCFeatureVector) -> float:
        """Compute extraversion from features."""
        base = 0.5
        base += f.social * 0.12
        base += f.we_words * 0.1
        base += f.friends * 0.15
        base += f.positive_emotion * 0.08
        base += f.exclamation_marks * 0.1
        base -= f.i_words * 0.03
        return self._clamp(base)

    def _compute_agreeableness(self, f: LIWCFeatureVector) -> float:
        """Compute agreeableness from features."""
        base = 0.5
        base += f.positive_emotion * 0.1
        base += f.social * 0.08
        base += f.family * 0.1
        base += f.friends * 0.1
        base -= f.negative_emotion * 0.08
        base -= f.anger * 0.12
        base -= f.power * 0.05
        return self._clamp(base)

    def _compute_neuroticism(self, f: LIWCFeatureVector) -> float:
        """Compute neuroticism from features."""
        base = 0.5
        base += f.negative_emotion * 0.12
        base += f.anxiety * 0.15
        base += f.sadness * 0.12
        base += f.anger * 0.08
        base += f.tentative * 0.06
        base -= f.positive_emotion * 0.08
        base -= f.certainty * 0.05
        return self._clamp(base)

    def _clamp(self, value: float) -> float:
        """Clamp to [0, 1]."""
        return max(0.0, min(1.0, value))


class LIWCProcessor:
    """Main LIWC processor for personality analysis."""

    def __init__(self, settings: LIWCProcessorSettings | None = None) -> None:
        self._settings = settings or LIWCProcessorSettings()
        self._extractor = FeatureExtractor()
        self._mapper = PersonalityMapper()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the LIWC processor."""
        self._initialized = True
        logger.info("liwc_processor_initialized")

    async def process(self, text: str) -> OceanScoresDTO:
        """Process text and return OCEAN scores."""
        if not text or len(text.strip()) < self._settings.min_word_count:
            logger.warning("text_too_short_for_liwc", length=len(text))
            return self._neutral_scores(confidence=0.2)
        truncated_text = text[:self._settings.max_text_length]
        features = self._extractor.extract(truncated_text)
        if features.word_count < self._settings.min_word_count:
            return self._neutral_scores(confidence=0.3)
        scores = self._mapper.map_to_ocean(features)
        confidence = self._compute_confidence(features.word_count)
        trait_scores = self._build_trait_scores(scores, features, confidence)
        return OceanScoresDTO(
            openness=scores[PersonalityTrait.OPENNESS],
            conscientiousness=scores[PersonalityTrait.CONSCIENTIOUSNESS],
            extraversion=scores[PersonalityTrait.EXTRAVERSION],
            agreeableness=scores[PersonalityTrait.AGREEABLENESS],
            neuroticism=scores[PersonalityTrait.NEUROTICISM],
            overall_confidence=confidence,
            trait_scores=trait_scores,
        )

    async def extract_features(self, text: str) -> LIWCFeatureVector:
        """Extract LIWC features without personality mapping."""
        truncated_text = text[:self._settings.max_text_length]
        return self._extractor.extract(truncated_text)

    def _compute_confidence(self, word_count: int) -> float:
        """Compute confidence based on word count."""
        base = self._settings.confidence_base
        word_boost = word_count * self._settings.confidence_word_factor
        return min(self._settings.max_confidence, base + word_boost)

    def _build_trait_scores(
        self, scores: dict[PersonalityTrait, float], features: LIWCFeatureVector, confidence: float
    ) -> list[TraitScoreDTO]:
        """Build detailed trait scores."""
        result = []
        margin = 0.15 * (1 - confidence)
        evidence_map = {
            PersonalityTrait.OPENNESS: self._get_openness_evidence(features),
            PersonalityTrait.CONSCIENTIOUSNESS: self._get_conscientiousness_evidence(features),
            PersonalityTrait.EXTRAVERSION: self._get_extraversion_evidence(features),
            PersonalityTrait.AGREEABLENESS: self._get_agreeableness_evidence(features),
            PersonalityTrait.NEUROTICISM: self._get_neuroticism_evidence(features),
        }
        for trait, value in scores.items():
            lower = max(0.0, value - margin)
            upper = min(1.0, value + margin)
            result.append(TraitScoreDTO(
                trait=trait,
                value=value,
                confidence_lower=lower,
                confidence_upper=upper,
                sample_count=1,
                evidence_markers=evidence_map.get(trait, [])[:3],
            ))
        return result

    def _get_openness_evidence(self, f: LIWCFeatureVector) -> list[str]:
        """Get evidence markers for openness."""
        evidence = []
        if f.insight > 1.0:
            evidence.append("high_insight")
        if f.cognitive > 2.0:
            evidence.append("cognitive_processing")
        if f.six_letter_words > 15:
            evidence.append("complex_vocabulary")
        return evidence

    def _get_conscientiousness_evidence(self, f: LIWCFeatureVector) -> list[str]:
        """Get evidence markers for conscientiousness."""
        evidence = []
        if f.achievement > 1.0:
            evidence.append("achievement_focus")
        if f.work > 1.5:
            evidence.append("work_oriented")
        if f.certainty > 1.0:
            evidence.append("decisive_language")
        return evidence

    def _get_extraversion_evidence(self, f: LIWCFeatureVector) -> list[str]:
        """Get evidence markers for extraversion."""
        evidence = []
        if f.social > 2.0:
            evidence.append("social_language")
        if f.we_words > 1.5:
            evidence.append("collective_focus")
        if f.positive_emotion > 2.0:
            evidence.append("positive_affect")
        return evidence

    def _get_agreeableness_evidence(self, f: LIWCFeatureVector) -> list[str]:
        """Get evidence markers for agreeableness."""
        evidence = []
        if f.positive_emotion > 2.0:
            evidence.append("warm_language")
        if f.family > 0.5 or f.friends > 0.5:
            evidence.append("relationship_focus")
        if f.anger < 0.5 and f.negative_emotion < 1.0:
            evidence.append("low_hostility")
        return evidence

    def _get_neuroticism_evidence(self, f: LIWCFeatureVector) -> list[str]:
        """Get evidence markers for neuroticism."""
        evidence = []
        if f.anxiety > 0.5:
            evidence.append("anxiety_language")
        if f.sadness > 0.5:
            evidence.append("sadness_expressed")
        if f.negative_emotion > 2.0:
            evidence.append("negative_affect")
        return evidence

    def _neutral_scores(self, confidence: float = 0.3) -> OceanScoresDTO:
        """Return neutral scores."""
        return OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            overall_confidence=confidence,
            trait_scores=[],
        )

    async def shutdown(self) -> None:
        """Shutdown the processor."""
        self._initialized = False
        logger.info("liwc_processor_shutdown")
