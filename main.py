import praw
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline


class EnhancedUSCISAnalyzer:
    def __init__(self, reddit_credentials):
        # Initialize Reddit API
        self.reddit = praw.Reddit(**reddit_credentials)

        # Initialize NLP models
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.emotion_analyzer = pipeline(
            "text-classification", model="j-hartmann/emotion-english-distilroberta-base"
        )

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(
                "Please install spaCy English model: python -m spacy download 'en_core_web_sm"
            )
            self.nlp = None

        # Initialize data structures
        self.posts_data = []
        self.comments_data = []
        self.sentiment_data = []
        self.network_data = []
        self.topics_data = []

        # Enhanced visa categories with priority levels
        self.visa_categories = {
            "Family-Based-Immediate": {
                "codes": ["IR-1", "CR-1", "IR-2", "CR-2", "IR-5"],
                "priority": "high",
                "processing_expectation": "fast",
            },
            "Employment-Based-Priority": {
                "codes": ["EB-1", "EB-2", "EB-3"],
                "priority": "high",
                "processing_expectation": "medium",
            },
            "Employment-Based-Investment": {
                "codes": ["EB-5"],
                "priority": "high",
                "processing_expectation": "slow",
            },
            "Nonimmigrant-Work": {
                "codes": ["H-1B", "L-1", "O-1", "P-1", "E-2"],
                "priority": "high",
                "processing_expectation": "medium",
            },
            "Nonimmigrant-Student": {
                "codes": ["F-1", "M-1", "J-1"],
                "priority": "medium",
                "processing_expectation": "fast",
            },
            "Nonimmigrant-Visitor": {
                "codes": ["B-1", "B-2"],
                "priority": "low",
                "processing_expectation": "fast",
            },
            "Humanitarian": {
                "codes": ["asylum", "refugee", "U-1", "T-1"],
                "priority": "critical",
                "processing_expectation": "variable",
            },
        }

        self.processing_milestones = {
            "major_positive": {
                "keywords": [
                    "approved",
                    "case was approved",
                    "naturalization ceremony",
                    "oath_ceremony",
                    "green card received",
                ],
                "weight": 1.0,
            },
            "positive": {
                "keywords": [
                    "interview waived",
                    "card is being produced",
                    "ready for pickup",
                    "case was updated",
                    "biometrics were taken",
                ],
                "weight": 0.7,
            },
            "neutral_progress": {
                "keywords": [
                    "case was received",
                    "fee was received",
                    "fingerprint review completed",
                    "case is being actively reviewed",
                    "transferred to another office",
                ],
                "weight": 0.3,
            },
            "concerning": {
                "keywords": [
                    "request for evidence",
                    "RFE",
                    "additional evidence needed",
                    "interview scheduled",
                    "medical examination",
                ],
                "weight": -0.2,
            },
            "negative": {
                "keywords": [
                    "notice of intent to deny",
                    "NOID",
                    "case was rejected",
                    "interview cancelled",
                    "administrative processing",
                ],
                "weight": -0.7,
            },
            "major_negative": {
                "keywords": [
                    "denied",
                    "case was denied",
                    "appeal filed",
                    "motion to reopen",
                ],
                "weight": -1.0,
            },
        }
        # Service centers with regional characteristics
        self.service_centers = {
            "California Service Center": {"region": "West", "code": "CSC"},
            "Nebraska Service Center": {"region": "Midwest", "code": "NSC"},
            "Texas Service Center": {"region": "South", "code": "TSC"},
            "Vermont Service Center": {"region": "Northeast", "code": "VSC"},
            "Potomac Service Center": {"region": "East", "code": "PSC"},
            "National Benefits Center": {"region": "National", "code": "NBC"},
            "National Visa Center": {"region": "National", "code": "NVC"},
        }

        # Create directories for data storage
        self.setup_directories()
