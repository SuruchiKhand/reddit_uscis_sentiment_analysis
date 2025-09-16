import praw
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import networkx as nx
import re
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict


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
            json.dump(self.network_data, f, default=str)

    def enhanced_sentiment_analysis(self, text, processing_milestones=None):

        # Base sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        blob = TextBlob(text)

        try:
            emotions = self.emotion_analyzer(text[:512])
            emotion_scores = {
                emotion["label"]: emotion["score"] for emotion in emotions
            }
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        except:
            emotion_scores = {"neutral": 1.0}
            dominant_emotion = ("neutral", 1.0)

        # Processing milestone adjustment
        milestone_adjustment = 0
        if processing_milestones:
            for category, milestones in processing_milestones.items():
                if milestones:
                    milestone_adjustment += self.processing_milestones.get(
                        category, {}
                    ).get("weight", 0)

        # Adjusted compound scene
        adjusted_compound = vader_scores["compound"] + (milestone_adjustment * 0.3)
        adjusted_compound = max(-1, min(1, adjusted_compound))

        return {
            "vader_compound": vader_scores["compound"],
            "vader_positive": vader_scores["pos"],
            "vader_neutral": vader_scores["neu"],
            "vader_negative": vader_scores["neg"],
            "textblob_polarity": blob.sentiment.polarity,
            "textblob_subjectivity": blob.sentiment.subjectivity,
            "emotion_scores": emotion_scores,
            "dominant_emotion": emotion_scores,
            "emotion_confidence": dominant_emotion[0],
            "milestone_adjustment": adjusted_compound,
            "sentiment_category": self.categorize_esentiment(adjusted_compound),
        }

    def categorize_sentiment(self, compound_score):
        if compound_score >= 0.5:
            return "very_positive"
        elif compound_score >= 0.1:
            return "positive"
        elif compound_score >= -0.1:
            return "neutral"
        elif compound_score >= -0.5:
            return "negative"
        else:
            return "very negative"

    def enhanced_classification(self, text):
        text_lower = text.lower()

        # Visa category classification with confidence scores
        category_matches = {}
        for category, info in self.visa_categories.items():
            score = 0
            matches = []

            # Check for exact code matches (high weight)
            for code in info["codes"]:
                if code.lower() in text_lower:
                    score += 3
                    matches.append(f"code:{code}")

            # Check for form patterns (medium weight)
            form_patterns = ["i-130", "i-485", "ds-260", "i-129", "i-539"]
            for pattern in form_patterns:
                if pattern in text_lower:
                    score += 2
                    matches.append(f"form:{pattern}")

            # Check for contextual keywords (low weight)
            category_keywords = {
                "Family-Based-Immediate": [
                    "spouse",
                    "husband",
                    "wife",
                    "parent",
                    "child",
                    "immediate relative",
                ],
                "Family-Based-Preference": [
                    "sibling",
                    "brother",
                    "sister",
                    "adult child",
                    "unmarried",
                ],
                "Employment-Based-Priority": [
                    "job",
                    "employer",
                    "work",
                    "employment",
                    "labor certification",
                ],
                "Employment-Based-Investment": [
                    "investor",
                    "investment",
                    "business",
                    "job creation",
                ],
                "Nonimmigrant-Work": [
                    "temporary work",
                    "specialty occupation",
                    "transfer",
                    "extraordinary ability",
                ],
                "Nonimmigrant-Student": [
                    "student",
                    "study",
                    "school",
                    "university",
                    "academic",
                ],
                "Nonimmigrant-Visitor": [
                    "tourist",
                    "visitor",
                    "business trip",
                    "vacation",
                ],
                "Humanitarian": [
                    "asylum",
                    "refugee",
                    "persecution",
                    "violence",
                    "protection",
                ],
            }
            if category in category_keywords:
                for keyword in category_keywords[category]:
                    if keyword in text_lower:
                        score += 1
                        matches.append(f"keyword:{keyword}")

            if score > 0:
                category_matches[category] = {"score": score, "matches": matches}

        # Processing milestone detection with enhanced patterns
        milestones_patterns = {}
        for category, info in self.processing_milestones.items():
            found_milestones = []
            for keyword in info["keywords"]:
                if keyword.lower() in text_lower:
                    found_milestones.append(keyword)
            if found_milestones:
                milestones_patterns[category] = found_milestones
        return category_matches, milestones_patterns

    def perform_topic_modeling(self, n_topics=15, min_df=2, max_df=0.8):
        print("Performing topic modeling...")

        # Combine all text data
        texts = []
        metadata = []

        for item in self.sentiment_data:
            if len(item["full_text"].strip()) > 50:  # Filter very short texts
                texts.append(item["full_text"])
                metadata.append(
                    {
                        "id": item["id"],
                        "type": item["content_type"],
                        "date": item["created_date"],
                        "sentiment": item["adjusted_compound"],
                    }
                )
        if len(texts) < n_topics:
            print(
                f"Warning: Only {len(texts)} documents available for {n_topics} topics"
            )
            n_topics = min(len(texts) // 2, n_topics)

        # Create TF-IDF matrix
        stop_words = list(stopwords.words("english")) + [
            "uscis",
            "case",
            "application",
            "form",
            "time",
            "day",
            "month",
            "year",
            "people",
            "person",
            "anyone",
            "everyone",
            "someone",
            "anything",
        ]

        vectorizer = TfidVectorizer(
            max_features=1000,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            ngram_range=(1, 2),
            lowercase=True,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            # Perform LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100,
                learning_method="batch",
            )

            lda.fit(tfidf_matrix)

            # Extract topics with interpretable names
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_word_indices = topic.argsort()[-15:][::-1]
                top_words = [feature_names[i] for i in top_word_indices]
                top_scores = [topic[i] for i in top_word_indices]

                # Generate topic name based on top words
                topic_name = self.generate_topic_name(top_words[:50])

                topics.append(
                    {
                        "topic_id": topic_idx,
                        "name": topic_name,
                        "words": top_words,
                        "scores": top_scores,
                        "coherence": np.mean(top_scores[:5]),
                    }
                )

                # Document-topic assignments
                doc_topic_matrix = lda.transform(tfidf_matrix)

                # Assign topics to documents
                for i, doc_topics in enumerate(doc_topic_matrix):
                    dominant_topic = np.argmax(doc_topics)
                    topic_confidence = doc_topics[dominant_topic]

                    metadata[i].update(
                        {
                            "dominant_topic": dominant_topic,
                            "topic_confidence": topic_confidence,
                            "topic_name": topics[dominant_topic]["name"],
                        }
                    )

                self.topics_data = {
                    "topics": topics,
                    "document_metadata": metadata,
                    "model_params": {
                        "n_topics": n_topics,
                        "min_df": min_df,
                        "max_df": max_df,
                    },
                }

                # Save topic model
                with open("models/topic_model.pkl", "wb") as f:
                    pickle.dump(
                        {"lda": lda, "vectorizer": vectorizer, "topics": topics}, f
                    )

                print(f"Topic modeling complete: {n_topics} topics extracted")
                return self.topics_data
        except Exception as e:
            print(f"Error in topic modeling: {e}")
            return None

    def generate_topic_name(self, top_words):

        # Immigration-specific topic naming logic
        if any(
            word in top_words for word in ["approved", "approval", "card", "produced"]
        ):
            return "Approvals & Card Production"
        elif any(word in top_words for word in ["interview", "scheduled", "completed"]):
            return "Interview Process"
        elif any(
            word in top_words for word in ["rfe", "evidence", "additional", "request"]
        ):
            return "RFE & Additional Evidence"
        elif any(
            word in top_words for word in ["denied", "denial", "rejected", "appeal"]
        ):
            return "Denials & Appeals"
        elif any(
            word in top_words for word in ["waiting", "delay", "processing", "time"]
        ):
            return "Processing Delays & Wait Times"
        elif any(
            word in top_words for word in ["biometrics", "fingerprint", "appointment"]
        ):
            return "Biometrics & Appointments"
        elif any(
            word in top_words for word in ["marriage", "spouse", "family", "relative"]
        ):
            return "Family-Based Immigration"
        elif any(
            word in top_words for word in ["job", "employer", "work", "employment"]
        ):
            return "Employment-Based Immigration"
        elif any(word in top_words for word in ["student", "f1", "opt", "study"]):
            return "Student Visas & OPT"
        elif any(word in top_words for word in ["asylum", "refugee", "persecution"]):
            return "Asylum & Refugee Cases"
        elif any(word in top_words for word in ["timeline", "experience", "sharing"]):
            return "Timeline Sharing & Experiences"
        elif any(
            word in top_words for word in ["attorney", "lawyer", "legal", "advice"]
        ):
            return "Legal Advice & Representation"
        elif any(word in top_words for word in ["expedite", "urgent", "emergency"]):
            return "Expedite Requests"
        else:
            return f"Topic: {' & '.join(top_words[:2])}"

    def analyze_user_network(self):
        print("Analyzing user interaction network...")

        # Build network graph
        G = nx.Graph()

        # Add edges from network data
        for interaction in self.network_data:
            if (
                interaction["source"] != "[deleted]"
                and interaction["target"] != "[deleted]"
            ):
                G.add_edge(
                    interaction["source"],
                    interaction["target"],
                    type=interaction["type"],
                    timestamp=interaction["timestamp"],
                )

        if len(G.nodes()) == 0:
            print("No network data available")
            return None

        # Calculate network metrics
        try:
            # Centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)

            # Community detection
            communities = nx.community.greedy_modularity_communities(G)

            # Identify key influencers
            influencers = []
            for node in G.nodes():
                influencer_score = (
                    degree_centrality.get(node, 0) * 0.4
                    + betweenness_centrality.get(node, 0) * 0.3
                    + closeness_centrality.get(node, 0) * 0.3
                )
                influencers.append(
                    {
                        "user": node,
                        "degree_centrality": degree_centrality.get(node, 0),
                        "betweenness_centrality": betweenness_centrality.get(node, 0),
                        "closenness_centrality": closeness_centrality.get(node, 0),
                        "influencer_score": influencer_score,
                    }
                )

            # Sort by influence score
            influencers.sort(key=lambda x: x["influencer_score"], reverse=True)

            network_analysis = {
                "total_users": len(G.nodes()),
                "total_interactions": len(G.edges()),
                "network_density": nx.density(G),
                "number_of_communities": len(communities),
                "top_influencers": influencers[:10],
                "average_clustering": nx.average_clustering(G),
                "diameter": nx.diameter(G) if nx.is_connected(G) else None,
            }
            print(
                f"Network analysis complete: {len(G.nodes())} users, {len(G.edges())} interactions"
            )
            return network_analysis

        except Exception as e:
            print(f"Erro in network analysis: {e}")
            return None

    def advanced_processing(self):
        print("Start advanced data processing...")

        all_data = []

        # Process posts
        for post in self.posts_data:
            full_text = f"{post['title']} {post['text']}"

            # Enhanced classification
            visa_matches, milestone_patterns = self.enhanced_classification(full_text)

            # Enhanced sentiment analysis
            sentiment = self.enhanced_sentiment_analysis(full_text, milestone_patterns)

            # Extract additional metadata
            metadata = self.extract_advanced_metadata(full_text, post)

            processed_item = {
                **post,
                **sentiment,
                "visa_category_matches": visa_matches,
                "processing_milestones": milestone_patterns,
                "metadata": metadata,
                "content_type": "post",
                "full_text": full_text,
                "text_length": len(full_text),
                "engagement_store": self.calculate_engagement_score(post),
            }

            all_data.append(processed_item)

        # Process comments
        for comment in self.comments_data:
            visa_matches, milestone_patterns = self.enhanced_classification(
                comment["text"]
            )
            sentiment = self.enhanced_sentiment_analysis(
                comment["text"], milestone_patterns
            )
            metadata = self.extract_advanced_metadata(comment["text"], comment)

            processed_item = {
                **comment,
                **sentiment,
                "visa_category_matches": visa_matches,
                "processing_milestones": milestone_patterns,
                "metadata": metadata,
                "content_type": "comment",
                "full_text": comment["text"],
                "text_length": len(comment["text"]),
                "engagement_score": comment["score"],
            }

        self.sentiment_data = all_data

        # Perform advanced analytics
        self.perform_topic_modeling()
        self.network_analysis = self.analyze_user_network()

        print(
            f"Advanced processing complete: {len(self.sentiment_data)} items analyzed"
        )

        # Save processed data
        self.save_processed_data()

    def extract_advanced_metadata(self, text, item):
        metadata = {
            "question_indicators": self.detect_question_patterns(text),
            "urgency_level": self.detect_urgency_level(text),
            "experience_sharing": self.detect_experience_sharing(text),
            "advice_seeking": self.detect_advice_seeking(text),
            "timeline_mention": self.extract_location_mentions(text),
            "location_mentions": self.extract_form_mentions(text),
            "service_center_mentions": self.extract_service_center_mentions(text),
        }
        return metadata

    def detect_question_patterns(self, text):
        question_patterns = [
            r"\?",
            r"how long",
            r"when will",
            r"is it normal",
            r"anyone else",
            r"should i",
            r"can i",
            r"will i",
            r"am i",
            r"has anyone",
        ]

        question_count = sum(
            1 for pattern in question_patterns if re.search(pattern, text, text.lower())
        )

        return {
            "is_question": question_count > 0,
            "question_intensity": question_count,
            "contains_question_mark": "?" in text,
        }

    def detect_urgency_level(self, text):
        urgent_keywords = [
            "urgent",
            "emergency",
            "asap",
            "immediately",
            "critical",
            "deadline",
            "expires",
            "time sensitive",
        ]
        moderate_keywords = ["soon", "quickly", "fast", "expedite"]

        urgent_count = sum(1 for word in urgent_keywords if word in text.lower())
        moderate_count = sum(
            1 for keyword in moderate_keywords if keyword in text.lower()
        )

        if urgent_count > 0:
            return "high"
        elif moderate_count > 0:
            return "medium"
        else:
            return "low"

    def detect_experience_sharing(self, text):
        experience_patterns = [
            r"my timeline",
            r"my experience",
            r"i received",
            r"i got",
            r"i was",
            r"just got",
            r"finally",
            r"update:",
            r"approved!",
            r"timeline:",
        ]

    def detect_advice_seeking(self, text):
        advice_patterns = [
            r"should i",
            r"what should",
            r"any advice",
            r"help please",
            r"what to do",
            r"recommendations",
            r"suggestions",
        ]
        return any(re.search(pattern, text.lower()) for pattern in advice_patterns)

    def extract_timeline_mentions(self, text):
        timeline_patterns = [
            r"(\d+)\s*(days?|weeks?|months?|years?)",
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}",
            r"\d{1,2}/\d{1,2}/\d{4}",
        ]
        timelines = []
        for pattern in timeline_patterns:
            matches = re.findall(pattern, text.lower())
            timelines.extend(matches)

        return timelines

    def extract_location_mentions(self, text):
        locations = [
            "atlanta",
            "boston",
            "chicago",
            "dallas",
            "detroit",
            "el paso",
            "houston",
            "las vegas",
            "los angeles",
            "miami",
            "new york",
            "newark",
            "orlando",
            "philadelphia",
            "phoenix",
            "san antonio",
            "san francisco",
            "seattle",
            "tampa",
            "washington dc",
        ]

        found_locations = []
        for location in locations:
            if location in text.lower():
                found_locations.append(location)
        return found_locations

    def extract_form_mentions(self, text):
        form_pattern = r"\b([IN]-\d{3}[A-Z]?|DS-\d{3})\b"
        forms = re.findall(form_pattern, text.upper())
        return list(set(forms))

    def extract_service_center_mentions(self, text):
        centers = []
        for center, info in self.service_centers.items():
            if center.lower() in text.lower() or info["code"].lower() in text.lower():
                centers.append(center)
        return centers

    def calculate_engagement_score(self, item):
        if "num_comments" in item:
            return (
                item["score"] * 0.3
                + item["num_comments"] * 0.5
                + item["upvote_ratio"] * 20
                + item.get("awards", 0) * 5
            )
        else:
            return item["score"]

    def save_processed_data(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as pandas DataFrame
        df = pd.DataFrame(self.sentiment_data)
        df.to_csv(f"data/processed/processed_data_{timestamp}.csv", index=False)
        df.to_parquet(f"data/processed/processed_data_{timestamp}.parquet")

        # Save topic modeling results
        if self.topics_data:
            with open(f"data/processed/topics_data_{timestamp}.json", "w") as f:
                json.dump(self.topics_data, f, indent=2, default=str)
        print(f"Processed data saved with timestamp {timestamp}")

    def create_advanced_visaualizations(self):
        print("Creating advanced visaulizations")

        # Create main dashboard
        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=(
                "Sentiment Trends Over Time",
                "Visa Category Distribution",
                "Emotion Analysis",
                "Processing Milestone Impact",
                "Service Center Performance",
                "Topic Distribution",
                "Engagement vs Sentiment",
                "Network Influence",
                "Urgency Levels",
                "Timeline Mentions",
                "Geographic Distribution",
                "Content Type Analysis",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "histogram"}, {"type": "bar"}, {"type": "bar"}],
            ],
        )

        df = pd.DataFrame(self.sentiment_data)
        df["created_date"] = pd.to_datetime(df["created_date"])
        df["week"] = df["created_date"].dt.to_period("W")

        # 1. Sentiment trends over time with volume
        weekly_sentiment = (
            df.groupby("week")
            .agg({"adjusted_compound": "mean", "id": "count"})
            .reset_index()
        )
        fig.add_trace(
            go.Bar(
                x=weekly_sentiment["week"].astype(str),
                y=weekly_sentiment["id"],
                name="Volume",
                opacity=0.3,
                yaxis="y2",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # 2. Top visa categories by volume
        visa_categories = []
        for item in self.sentiment_data:
            for category in item.get("visa_category_matches", {}):
                visa_categories.append(category)

        if visa_categories:
            category_counts = Counter(visa_categories).most_common(8)
            fig.add_trace(
                go.Bar(
                    x=[cat[0].replace("-", " ") for cat in category_counts],
                    y=[cat[1] for cat in category_counts],
                    marker_color="lightblue",
                    name="Categories",
                ),
                row=1,
                col=2,
            )

        # 3. Emotion distribution
        emotions = df["dominant_emotion"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=emotions.index,
                values=emotions.values,
                name="Emotions",
                marker_colors=px.colors.qualitative.Set3,
            ),
            row=1,
            col=3,
        )

        # 4. Processing milestone sentiment impact
        milestone_sentiment = {}
        for item in self.sentiment_data:
            for milestone_type, milestones in item.get(
                "processing_milestones", {}
            ).items():
                if milestones:
                    if milestone_type not in milestone_sentiment:
                        milestone_sentiment[milestone_type] = []
                    milestone_sentiment[milestone_type].append(
                        item["adjusted_compound"]
                    )

        milestone_avgs = {k: np.mean(v) for k, v in milestone_sentiment.items() if v}
        if milestone_avgs:
            colors = [
                "green" if v > 0.1 else "red" if v < -0.1 else "gray"
                for v in milestone_avgs.values()
            ]
            fig.add_trace(
                go.Bar(
                    x=list(milestone_avgs.keys()),
                    y=list(milestone_avgs.values()),
                    marker_color=colors,
                    name="Milestones",
                ),
                row=2,
                col=1,
            )

        # 5. Service center performance (if network analysis available)
        if hasattr(self, "network_analysis") and self.network_analysis:
            service_performance = {}
            for item in self.sentiment_data:
                for center in item["metadata"].get("service_center_mentions", []):
                    if center not in service_performance:
                        service_performance[center] = []
                    service_performance[center].append(item["adjusted_compound"])

            center_avgs = {
                k: np.mean(v) for k, v in service_performance.items() if len(v) > 5
            }
            if center_avgs:
                fig.add_trace(
                    go.Bar(
                        x=[
                            k.replace(" Service Center", "") for k in center_avgs.keys()
                        ],
                        y=list(center_avgs.values()),
                        marker_color="lightgreen",
                        name="Service Centers",
                    ),
                    row=2,
                    col=2,
                )

        # 6. Topic distribution (if topics available)
        if self.topics_data and "document_metadata" in self.topics_data:
            topic_counts = Counter(
                [doc["topics_name"] for doc in self.topics_data["document_metadata"]]
            )
            top_topics = topic_counts.most_common(8)
            fig.add_trace(
                go.Pie(
                    labels=[t[0] for t in top_topics],
                    values=[t[1] for t in top_topics],
                    name="Topics",
                    marker_colors=px.colors.qualitative.Pastel,
                ),
                row=2,
                col=3,
            )

        # 7. Engagement vs Sentiment correlation
        posts_df = df[df["content_type"] == "post"].copy()
        if not posts_df.empty and "engagement_score" in posts_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=posts_df["ajusted_compound"],
                    y=posts_df["engagement_score"],
                    mode="markers",
                    marker=dict(
                        color=posts_df["adjusted_compound"],
                        colorscale="RdYlBu",
                        size=8,
                        opacity=0.7,
                    ),
                    name="Posts",
                ),
                row=3,
                col=1,
            )

        # 8. Top network influencers (if available)
        if (
            hasattr(self, "network_analysis")
            and self.network_analysis
            and "top_influencers" in self.network_analysis
        ):
            top_influencers = self.network_analysis["top_influencers"]
            fig.add_trace(
                go.Bar(
                    x=[
                        (
                            inf["user"][:10] + "..."
                            if len(inf["user"]) > 10
                            else inf["user"]
                        )
                        for inf in top_influencers
                    ],
                    y=[inf["influencer_score"] for inf in top_influencers],
                    marker_color="orange",
                    name="Top Influencers",
                ),
                row=3,
                col=2,
            )

        # 9. Urgency level distribution
        urgency_counts = Counter(
            [item["metadata"]["urgency_level"] for item in self.sentiment_data]
        )
        fig.add_trace(
            go.Bar(
                x=list(urgency_counts.keys()),
                y=list(urgency_counts.values()),
                marker_color=["red", "orange", "green"],
                name="Urgency",
            ),
            row=3,
            col=3,
        )

        # 10. Timeline mentions distribution
        timeline_counts = []
        for item in self.sentiment_data:
            timeline_counts.extend(item["metadata"].get("timeline_mentions", []))

        if timeline_counts:
            fig.add_trace(
                go.Histogram(
                    x=[str(t) for t in timeline_counts[:50]],
                    name="Timelines",
                ),
                row=4,
                col=1,
            )

        # 11. Geographic distribution
        location_counts = []
        for item in self.sentiment_data:
            location_counts.extend(item["metadata"].get("location_mentions", []))

        if location_counts:
            top_locations = Counter(location_counts).most_common(8)
            fig.add_trace(
                go.Bar(
                    x=[loc.title() for loc in top_locations],
                    y=[loc[1] for loc in top_locations],
                    marker_color="lightcoral",
                    name="Locations",
                ),
                row=4,
                col=2,
            )

        # 12. Content type analysis
        content_analysis = (
            df.groupby("content_type")
            .agg({"adjusted_compound": "mean", "id": "count"})
            .reset_index()
        )
        fig.add_trace(
            go.Bar(
                x=content_analysis["content_type"],
                y=content_analysis["id"],
                name="Content Types",
                marker_color="lightsteelblue",
            ),
            row=4,
            col=3,
        )

        # Update layout
        fig.update_layout(
            height=1600,
            title_text="Advanced USCIS Reddit Sentiment Analysis Dashboard",
            showlegend=False,
            font=dict(size=10),
        )

        # Save and show
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.write_html(f"visualizations/advanced_dashboard_{timestamp}.html")
        fig.show()

        return fig

    def generate_comprehensive_report(self):
        print("Generating comprehensive analysis report...")

        df = pd.DataFrame(self.sentiment_data)

        # Calculate advanced metrics
        report = {
            "executive_summary": {
                "analysis_period": {
                    "start": df["created_date"].min(),
                    "end": df["created_date"].max(),
                    "total_days": (
                        df["created_date"].max() - df["created_date"].min()
                    ).days,
                },
                "data_volume": {
                    "total_posts": len(df[df["content_type"] == "post"]),
                    "total_comments": len(df[df["content_type"] == "comment"]),
                    "total_interactions": len(df),
                    "daily_average": len(df)
                    / max(
                        1, (df["created_date"].max() - df["created_date"].min()).days
                    ),
                },
                "overall_sentiment": {
                    "average_sentiment": df["adjusted_compound"].mean(),
                    "sentiment_std": df["adjusted_compound"].std(),
                    "positive_ratio": len(df[df["adjusted_compound"] > 0.1]) / len(df),
                    "negative_ratio": len(df[df["adjusted_compound"] < -0.1]) / len(df),
                    "dominant_emotion": (
                        df["dominant_emotion"].mode().iloc[0]
                        if not df.empty
                        else "neutral"
                    ),
                },
            },
            "visa_category_insights": self.analyze_visa_categories_advanced(),
            "processing_milestone_analysis": self.analyze_processing_milestones(),
            "temporal_analysis": self.analyze_temporal_patterns(df),
            "community_dynamics": self.analyze_community_dynamics(df),
            "content_analysis": self.analyze_content_patterns(df),
            "geographic_insights": self.analyze_geographic_patterns(df),
            "topic_modeling_results": self.topics_data if self.topics_data else None,
            "network_analysis_results": (
                self.network_analysts if hasattr(self, "network_analysis") else None
            ),
            "key_findings": self.summarize_key_findings(),
            "recommendations": self.generate_recommendations(df),
        }

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        with open(f"reports/comprehensive_report_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save as formatted text report
        self.save_formatted_report(report, timestamp)

        print(f"Comprehensive report generated and saved with timestamp {timestamp}")
        return report

    def analyze_visa_categories_advanced(self):
        category_analysis = {}

        for item in self.sentiment_data:
            for category, match_info in item.get("visa_category_matches", {}).items():
                if category not in category_analysis:
                    category_analysis[category] = {
                        "mentions": 0,
                        "sentiments": [],
                        "emotions": [],
                        "urgency_levels": [],
                        "engagement_scores": [],
                    }
                category_analysis[category]["mentions"] += 1
                category_analysis[category]["sentiments"].append(
                    item["adjusted_compound"]
                )
                category_analysis[category]["emotions"].append(item["dominant_emotion"])
                category_analysis[category]["urgency_levels"].append(
                    item["metadata"]["urgency_level"]
                )

                if "engagement_score" in item:
                    category_analysis[category]["engagement_scores"].append(
                        item["engagement_score"]
                    )

        # Calculate summary statistics
        for category, data in category_analysis.items():
            if data["sentiments"]:
                data["avg_sentiment"] = np.mean(data["sentiments"])
                data["sentiment_std"] = np.std(data["sentiments"])
                data["dominant_emotion"] = Counter(data["emotions"]).most_common(1)[0][
                    0
                ]
                data["high_urgency_ratio"] = data["urgency_levels"].count("high") / len(
                    data["urgency_levels"]
                )

            if data["engagement_scores"]:
                data["avg_engagement"] = np.mean(data["engagement_scores"])
            return dict(
                sorted(
                    category_analysis.items(),
                    key=lambda x: x[1]["mentions"],
                    reverse=True,
                )
            )

    def analyze_processing_milestones(self):
        milestone_analysis = {}

        for item in self.sentiment_data:
            for milestone_type, milestones in item.get(
                "processing_milestones", {}
            ).items():
                if milestones:
                    milestone_analysis[milestone_type]["mentions"] += 1
                    milestone_analysis[milestone_type]["sentiments"].append(
                        item["adjusted_compound"]
                    )
                    milestone_analysis[milestone_type]["emotions"].append(
                        item["dominant_emotion"]
                    )

        # Calculate statistics
        for milestone, data in milestone_analysis.items():
            if data["sentiments"]:
                data["avg_sentiment"] = np.mean(data["sentiments"])
                data["sentiment_range"] = [
                    min(data["sentiments"]),
                    max(data["sentiments"]),
                ]
                data["dominant_emotion"] = Counter(data["emotions"]).most_common(1)[0][
                    0
                ]
                data["expected_weight"] = self.processing_milestones[milestone_type][
                    "weight"
                ]

        return milestone_analysis

    def analyze_temporal_patterns(self, df):

        df["hour"] = df["created_date"].dt.hour
        df["day_of_week"] = df["created_date"].dt.day_name()
        df["week"] = df["created_date"].dt.to_period("W")

        temporal_analysis = {
            "hourly_patterns": {
                "activity_by_hour": df.groupby("hour").size().to_dict(),
                "sentiment_by_hour": df.groupby("hour")["adjusted_compound"]
                .mean()
                .to_dict(),
            },
            "weekly_patterns": {
                "activity_by_day": df.groupby("day_of_week").size().to_dict(),
                "sentiment_by_day": df.groupby("day_of_week")["adjusted_compound"]
                .mean()
                .to_dict(),
            },
            "trend_analysis": {
                "weekly_sentiment": df.groupby("week")["adjusted_compound"]
                .mean()
                .to_dict(),
                "weekly_volume": df.groupby("week").size().to_dict(),
                "sentiment_trend": (
                    "improving"
                    if df["adjusted_compound"].corr(df.index) > 0
                    else "declining"
                ),
            },
        }
        return temporal_analysis

    def analyze_community_dynamics(self, df):

        posts_df = df[df["content_type"] == "post"]
        comments_df = df[df["content_type"] == "comment"]

        community_analysis = {
            "engagement_patterns": {
                "avg_comments_per_post": (
                    comments_df.groupby("posts_id").size().mean()
                    if not comments_df.empty
                    else 0
                ),
                "high_engagement_threshold": (
                    posts_df["engagement score"].quantile(0.9)
                    if "engagement score" in posts_df.columns
                    else 0
                ),
                "question_response_ratio": self.calculate_question_response_rate(df),
            },
            "support_patterns": {
                "advice_seeking_ratio": sum(
                    1
                    for item in self.sentiment_data
                    if item["metadata"]["advice_seeking"]
                )
                / len(self.sentiment_data),
                "experience_sharing_ratio": sum(
                    1
                    for item in self.sentiment_data
                    if item["metadata"]["experience_sharing"]
                )
                / len(self.sentiment_data),
            },
            "content_quality": {
                "avg_text_length": df["text_length"].mean(),
                "detailed_post_ratio": len(df[df["text_length"] > 500]) / len(df),
            },
        }
        return community_analysis

    def calculate_question_response_rate(self, df):
        question_posts = [
            item
            for item in self.sentiment_data
            if item["content_type"] == "post"
            and item["metadata"]["question_indicators"]["is_question"]
        ]

        if not question_posts:
            return 0

        responded_questions = sum(
            1 for post in question_posts if post.get("num_comments", 0) > 0
        )
        return responded_questions / len(question_posts)

    def analyze_content_patterns(self, df):

        content_analysis = {
            "question_patterns": {
                "total_questions": sum(
                    1
                    for item in self.sentiment_data
                    if item["metadata"]["question_indicators"]["is_question"]
                ),
                "question_types": self.categorize_question_types(),
                "avg_question_sentiment": np.mean(
                    [
                        item["adjusted_compound"]
                        for item in self.sentiment_data
                        if item["metadata"]["question_indicators"]["is_question"]
                    ]
                ),
            },
            "urgency_analysis": {
                "urgency_distribution": Counter(
                    [item["metadata"]["urgency_level"] for item in self.sentiment_data]
                ).most_common(),
                "urgency_sentiment_correlation": self.analyze_urgency_sentiment_correlation(),
            },
            "form_mentions": {
                "most_mentioned_forms": self.get_most_mentioned_forms(),
                "common_wait_times": self.extract_common_wait_times(),
            },
        }
        return content_analysis

    def categorize_question_types(self):
        question_categories = {
            "timeline": ["how long", "when will", "timeline", "processing time"],
            "status": ["case status", "what does this mean", "is this normal"],
            "process": ["what happens next", "next steps", "should i"],
            "documents": ["what documents", "do i need", "required"],
            "expedite": ["expedite", "urgent", "emergency"],
            "denial": ["denied", "rfe", "noid", "appeal"],
        }

        question_type_counts = {category: 0 for category in question_categories}
        for item in self.sentiment_data:
            if item["metadata"]["question_indicators"]["is_question"]:
                text_lower = item["full_text"].lower()
                for category, keywords in question_categories.items():
                    if any(keyword in text_lower for keyword in keywords):
                        question_type_counts[category] += 1
                        break

        return question_type_counts

    def analyze_urgency_sentiment_correlation(self):
        urgency_sentiment = {"high": [], "medium": [], "low": []}
        for item in self.sentiment_data:
            urgency = item["metadata"]["urgency_level"]
            urgency_sentiment[urgency].append(item["adjusted_compound"])

        return {
            level: np.mean(sentiments) if sentiments else 0
            for level, sentiments in urgency_sentiment.items()
        }

    def get_most_mentioned_forms(self):
        form_mentions = []
        for item in self.sentiment_data:
            form_mentions.extend(item["metadata"].get("location_mentions", []))

        return Counter(form_mentions).most_common(10)

    def analyze_form_sentiment(self):
        form_sentiment = defaultdict(list)
        for item in self.sentiment_data:
            for form in item["metadata"].get("form_mentions", []):
                form_sentiment[form].append(item["adjusted_compound"])
        return {
            form: np.mean(sentiments)
            for form, sentiments in form_sentiment.items()
            if len(sentiments) >= 5
        }  # Only forms with at least 5 mentions

    def extract_common_wait_times(self):
        wait_patterns = {
            "days": r"(\d+)\s*days?",
            "weeks": r"(\d+)\s*weeks?",
            "months": r"(\d+)\s*months?",
            "years": r"(\d+)\s*years?",
        }
        wait_times = {"days": [], "weeks": [], "months": [], "years": []}
        for item in self.sentiment_data:
            text = item["full_text"].lower()
            for unit, pattern in wait_patterns.items():
                matches = re.findall(pattern, text)
                wait_times[unit].extend(
                    [int(m) for m in matches if int(m) < 100]
                )  # Filter out unrealistic values

        return {
            unit: {
                "mentions": len(times),
                "average": np.mean(times) if times else 0,
                "median": np.median(times) if times else 0,
            }
            for unit, times in wait_times.items()
        }
