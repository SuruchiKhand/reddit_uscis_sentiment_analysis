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
