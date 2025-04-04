"""
AI Engine Module
Provides AI functionality for task management, including prioritization and categorization.
"""

import datetime
import logging
import math
import re
from typing import Dict, List, Any, Optional, Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class AIEngine:
    """
    Provides AI functionality for analyzing tasks, including:
    - Task prioritization based on due dates and content
    - Task categorization suggestions
    - Similarity analysis between tasks
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.urgent_keywords = {
            'urgent', 'important', 'critical', 'asap', 'deadline', 'due',
            'priority', 'immediate', 'emergency', 'crucial', 'vital'
        }
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Dictionary to track processed tasks to avoid unnecessary reprocessing
        self.processed_tasks: Dict[int, Dict[str, Any]] = {}
    
    def prioritize_tasks(self, tasks: List[Any]) -> List[int]:
        """
        Prioritize tasks based on multiple factors.
        
        Args:
            tasks: List of Task objects to prioritize
            
        Returns:
            List of priority scores (1-5, 5 being highest priority)
        """
        if not tasks:
            return []
        
        # Calculate base scores for each task
        task_scores = []
        
        for task in tasks:
            # Calculate due date score (0-1)
            due_date_score = self._calculate_due_date_score(task)
            
            # Calculate content importance score (0-1)
            content_score = self._calculate_content_importance(task)
            
            # Combine scores (weighted average)
            combined_score = 0.6 * due_date_score + 0.4 * content_score
            task_scores.append(combined_score)
        
        # Convert to priority levels (1-5)
        priorities = self._normalize_scores(task_scores)
        
        return priorities
    
    def suggest_category(self, task: Any) -> str:
        """
        Suggests a category for a task based on its content.
        
        Args:
            task: A Task object to categorize
            
        Returns:
            Suggested category name
        """
        # Extract text from task
        task_text = f"{task.title} {task.description}"
        
        # Define category patterns
        category_patterns = {
            'Work': [r'\bwork\b', r'\bjob\b', r'\bproject\b', r'\bclient\b', r'\boffice\b', 
                     r'\bmeeting\b', r'\breport\b', r'\bboss\b', r'\bpresentation\b'],
            'Personal': [r'\bhome\b', r'\bfamily\b', r'\bfriend\b', r'\bpersonal\b', 
                        r'\bleisure\b', r'\bhobby\b', r'\bvacation\b'],
            'Health': [r'\bhealth\b', r'\bdoctor\b', r'\bmedical\b', r'\bworkout\b', 
                      r'\bfitness\b', r'\bdiet\b', r'\bexercise\b', r'\bgym\b'],
            'Finance': [r'\bmoney\b', r'\bfinance\b', r'\bpayment\b', r'\bbill\b', 
                       r'\btax\b', r'\bbank\b', r'\bbudget\b', r'\binvoice\b'],
            'Shopping': [r'\bbuy\b', r'\bpurchase\b', r'\bshopping\b', r'\border\b', 
                        r'\bstore\b', r'\bamazon\b', r'\bonline\b'],
            'Study': [r'\bstudy\b', r'\bschool\b', r'\bcourse\b', r'\beducation\b', 
                     r'\bclass\b', r'\bexam\b', r'\bhomework\b', r'\bassignment\b'],
            'Travel': [r'\btravel\b', r'\btrip\b', r'\bflight\b', r'\bhotel\b', 
                      r'\bvacation\b', r'\broute\b', r'\btransport\b']
        }
        
        # Check each category pattern
        matches = {}
        for category, patterns in category_patterns.items():
            count = 0
            for pattern in patterns:
                if re.search(pattern, task_text.lower()):
                    count += 1
            matches[category] = count
        
        # Get the category with most matches
        if matches and max(matches.values()) > 0:
            return max(matches.items(), key=lambda x: x[1])[0]
        
        # If no clear category, return "General"
        return "General"
    
    def find_similar_tasks(self, task: Any, all_tasks: List[Any], threshold: float = 0.3) -> List[Any]:
        """
        Find tasks similar to the given task.
        
        Args:
            task: The task to find similarities for
            all_tasks: List of all tasks to search in
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar tasks
        """
        if not all_tasks or len(all_tasks) < 2:
            return []
        
        # Prepare corpus of tasks
        task_texts = [f"{t.title} {t.description}" for t in all_tasks]
        
        # Generate TF-IDF vectors
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(task_texts)
        except:
            # In case of vectorization failure, return empty list
            return []
        
        # Find the index of our target task
        task_idx = next((i for i, t in enumerate(all_tasks) if t.task_id == task.task_id), -1)
        if task_idx == -1:
            return []
        
        # Calculate similarities
        task_vector = tfidf_matrix[task_idx:task_idx+1]
        cosine_similarities = cosine_similarity(task_vector, tfidf_matrix).flatten()
        
        # Get similar tasks (ignoring the task itself)
        similar_indices = [i for i, score in enumerate(cosine_similarities) 
                          if score > threshold and i != task_idx]
        
        similar_tasks = [all_tasks[i] for i in similar_indices]
        
        return similar_tasks
    
    def _calculate_due_date_score(self, task: Any) -> float:
        """
        Calculate a score based on the task's due date.
        Closer due dates get higher scores.
        
        Args:
            task: Task object with due_date attribute
            
        Returns:
            Score between 0 and 1
        """
        # If no due date, give a middle priority
        if not task.due_date:
            return 0.5
            
        # Calculate days until due
        today = datetime.datetime.now().date()
        due_date = task.due_date.date()
        days_until_due = (due_date - today).days
        
        if days_until_due < 0:
            # Overdue tasks get highest priority
            return 1.0
        elif days_until_due == 0:
            # Due today
            return 0.9
        elif days_until_due <= 1:
            # Due tomorrow
            return 0.8
        elif days_until_due <= 3:
            # Due within 3 days
            return 0.7
        elif days_until_due <= 7:
            # Due within a week
            return 0.6
        elif days_until_due <= 14:
            # Due within two weeks
            return 0.5
        elif days_until_due <= 30:
            # Due within a month
            return 0.4
        else:
            # Due in more than a month
            return 0.3
    
    def _calculate_content_importance(self, task: Any) -> float:
        """
        Analyzes task content to determine importance.
        Looks for urgent keywords and phrases.
        
        Args:
            task: Task object with title and description
            
        Returns:
            Score between 0 and 1
        """
        # Get task content
        task_text = f"{task.title} {task.description}".lower()
        
        # Count urgent keywords
        words = word_tokenize(task_text)
        filtered_words = [w for w in words if w.lower() not in self.stop_words]
        
        # Count urgent keywords
        urgent_count = sum(1 for word in filtered_words if word.lower() in self.urgent_keywords)
        
        # Check for deadline mentions
        has_deadline = 1 if re.search(r'deadline|due\s+by|by\s+\w+day', task_text) else 0
        
        # Check for explicit priority mentions
        priority_level = 0
        if re.search(r'high\s+priority|highest\s+priority', task_text):
            priority_level = 1.0
        elif re.search(r'medium\s+priority', task_text):
            priority_level = 0.5
        elif re.search(r'low\s+priority', task_text):
            priority_level = 0.2
            
        # Manual priority overrides content scoring if explicitly set
        if task.manual_priority is not None:
            return task.manual_priority / 5.0
            
        # Calculate final score
        score = min(1.0, (0.15 * urgent_count + 0.3 * has_deadline + 0.5 * priority_level))
        
        # Ensure minimum score is 0.1
        return max(0.1, score)
    
    def _normalize_scores(self, scores: List[float]) -> List[int]:
        """
        Convert continuous scores to discrete priority levels (1-5).
        
        Args:
            scores: List of scores between 0 and 1
            
        Returns:
            List of integer priority levels (1-5)
        """
        if not scores:
            return []
            
        # Simple mapping of score ranges to priority levels
        priority_levels = []
        for score in scores:
            if score >= 0.8:
                priority_levels.append(5)  # Highest priority
            elif score >= 0.6:
                priority_levels.append(4)
            elif score >= 0.4:
                priority_levels.append(3)
            elif score >= 0.2:
                priority_levels.append(2)
            else:
                priority_levels.append(1)  # Lowest priority
                
        return priority_levels
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key terms from text using TF-IDF.
        
        Args:
            text: Text to analyze
            top_n: Number of top keywords to return
            
        Returns:
            List of extracted keywords
        """
        if not text or len(text.strip()) == 0:
            return []
            
        # Tokenize and clean text
        words = word_tokenize(text.lower())
        filtered_words = [w for w in words if w.lower() not in self.stop_words 
                        and w.isalpha() and len(w) > 2]
        
        # If not enough words after filtering, return the filtered words
        if len(filtered_words) <= top_n:
            return filtered_words
            
        # Use TF-IDF to identify important terms
        try:
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform([' '.join(filtered_words)])
            
            # Get feature names
            feature_names = tfidf.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray().flatten()
            
            # Create a list of (term, score) tuples and sort by score
            term_scores = [(term, score) for term, score in zip(feature_names, scores)]
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top terms
            return [term for term, score in term_scores[:top_n]]
        except:
            # In case of errors, return most frequent words
            from collections import Counter
            word_counts = Counter(filtered_words)
            return [word for word, count in word_counts.most_common(top_n)]
