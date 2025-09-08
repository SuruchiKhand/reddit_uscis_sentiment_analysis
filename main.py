import praw
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json


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

    def setup_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            "data/raw",
            "data/processed",
            "data/exports",
            "visualizations",
            "reports",
            "models",
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def enhanced_collect_data(
        self,
        start_date=datetime(2025, 1, 20),
        end_date=None,
        limit=2000,
        include_hot=True,
    ):
        if end_date is None:
            end_date = datetime.now()
        print(
            f"Enhanced data collection from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        subreddit = self.reddit.subreddit("USCIS")
        posts_collected = 0

        # Collect from multiple sorting methods
        sorting_methods = ["new", "hot", "top"] if include_hot else ["new", "top"]
        for method in sorting_methods:
            print(f'Collecting posts using "{method}" sorting...')
            if method == "new":
                submissions = subreddit.new(limit=limit)
            elif method == "hot":
                submissions = subreddit.hot(limit=limit // 2)
            else:
                submissions = subreddit.top(time_filter="month", limit=limit // 2)

            for submission in submissions:
                submission_date = datetime.fromtimestamp(submission.created_utc)

                if start_date <= submission_date <= end_date:
                    post_data = {
                        "id": submission.id,
                        "title": submission.title,
                        "text": submission.text,
                        "author": (
                            str(submission.author) if submission.author else "[deleted]"
                        ),
                        "score": submission.score,
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                        "created_utc": submission.created_utc,
                        "created_date": submission_date,
                        "url": submission.url,
                        "flair": submission.link_flair_text,
                        "is_self": submission.is_self,
                        "domain": submission.domain,
                        "gilded": submission.gilded,
                        "stickied": submission.stickied,
                        "sorting_method": method,
                        "awards": submission.total_awards_received,
                        "crosspost_parent": (
                            submission.crosspost_parent
                            if hasattr(submission, "crosspost_parent")
                            else None
                        ),
                    }
                    self.posts_data.append(post_data)

                    # Enhanced comment collection with threading
                    try:
                        submission.comments.replace_more(limit=5)
                        comment_forest = submission.comments.list()

                        for comment in comment_forest[:20]:
                            comment_data = {
                                "id": comment.id,
                                "post_id": submission.id,
                                "author": (
                                    str(comment.author)
                                    if comment.author
                                    else "[deleted]"
                                ),
                                "text": comment.body,
                                "score": comment.score,
                                "created_utc": comment.created_utc,
                                "created_date": datetime.fromtimestamp(
                                    comment.created_utc
                                ),
                                "parent_id": comment.parent_id,
                                "is_root": comment.is_root,
                            }
                            self.comments_data.append(comment_data)

                            # Build network relationships
                            if comment.parent_id.startswith("t1_"):
                                self.network_data.append(
                                    {
                                        "source": (
                                            str(comment.author)
                                            if comment.author
                                            else "[deleted]"
                                        ),
                                        "target": "parent_comment",
                                        "type": "reply",
                                        "timestamp": comment.created_utc,
                                        "post_id": submission.id,
                                    }
                                )
                            else:
                                self.network_data.append(
                                    {
                                        "source": (
                                            str(comment.author)
                                            if comment.author
                                            else "[deleted]"
                                        ),
                                        "target": (
                                            str(submission.author)
                                            if submission.author
                                            else "[deleted]"
                                        ),
                                        "type": "comment",
                                        "timestamp": comment.created_utc,
                                        "post_id": submission.id,
                                    }
                                )
                    except Exception as e:
                        print(
                            f"Error collecting comments for post {submission.id}: {e}"
                        )

                    posts_collected += 1
                    if posts_collected % 100 == 0:
                        print(f"Collected {posts_collected} posts so far...")

            print(
                f"Collection complete: {len(self.posts_data)} posts, {len(self.comments_data)} comments"
            )
            self.save_raw_data()

    def save_raw_data(self):
        """Save raw collected data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as pickle for Python objects
        with open(f"data/raw/posts_{timestamp}.pkl", "wb") as f:
            pickle.dump(self.posts_data, f)

        with open(f"data/raw/comments_{timestamp}.pkl", "wb") as f:
            pickle.dump(self.comments_data, f)

        # Save as JSON for interoperability
        with open(f"data/raw/network_data_{timestamp}.json", "w") as f:
            json.dump(self.network_data, f, indent=4)
