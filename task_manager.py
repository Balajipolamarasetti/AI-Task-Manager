"""
Task Manager Module
Core functionality for managing tasks.
"""

import datetime
import logging
import os
from typing import Dict, List, Optional, Any, Union

from ai_engine import AIEngine
from storage_handler import StorageHandler
from utils import format_date, validate_date

logger = logging.getLogger(__name__)

class Task:
    """
    Represents a single task in the system.
    """
    def __init__(
        self,
        title: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        due_date: Optional[str] = None,
        manual_priority: Optional[int] = None,
        status: str = "pending",
        task_id: Optional[int] = None,
        created_at: Optional[datetime.datetime] = None
    ):
        self.task_id = task_id
        self.title = title
        self.description = description or ""
        self.category = category or "General"
        self.due_date = validate_date(due_date)
        self.manual_priority = manual_priority
        self.ai_priority = None  # To be set by AI engine
        self.status = status
        self.created_at = created_at or datetime.datetime.now()
        self.completed_at = None
    
    def mark_as_completed(self) -> None:
        """Mark the task as completed and record completion time."""
        self.status = "completed"
        self.completed_at = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for storage."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "manual_priority": self.manual_priority,
            "ai_priority": self.ai_priority,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create a Task object from a dictionary."""
        task = cls(
            title=data["title"],
            description=data.get("description"),
            category=data.get("category"),
            due_date=None,  # We'll set it directly to avoid re-validation
            manual_priority=data.get("manual_priority"),
            status=data.get("status", "pending"),
            task_id=data.get("task_id"),
        )
        
        # Set dates directly to avoid validation issues
        if data.get("due_date"):
            try:
                task.due_date = datetime.datetime.fromisoformat(data["due_date"])
            except (ValueError, TypeError):
                task.due_date = None
        
        if data.get("created_at"):
            try:
                task.created_at = datetime.datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                task.created_at = datetime.datetime.now()
        
        if data.get("completed_at"):
            try:
                task.completed_at = datetime.datetime.fromisoformat(data["completed_at"])
            except (ValueError, TypeError):
                task.completed_at = None
        
        task.ai_priority = data.get("ai_priority")
        
        return task
    
    def __str__(self) -> str:
        """String representation of the task."""
        status_str = "[✓]" if self.status == "completed" else "[ ]"
        priority_str = f"P{self.ai_priority}" if self.ai_priority is not None else f"P{self.manual_priority}" if self.manual_priority is not None else "P?"
        due_str = f"Due: {format_date(self.due_date)}" if self.due_date else "No due date"
        return f"{self.task_id}. {status_str} {priority_str} - {self.title} ({self.category}) - {due_str}"


class TaskManager:
    """
    Manages all operations related to tasks, including creation, updates, and retrieval.
    Integrates with AI components for task prioritization.
    """
    def __init__(self):
        self.storage = StorageHandler()
        self.ai_engine = AIEngine()
        self.tasks = self.storage.load_tasks()
        
        # Generate IDs for tasks if not already assigned
        if self.tasks:
            max_id = max(task.task_id for task in self.tasks if task.task_id is not None) if any(task.task_id is not None for task in self.tasks) else 0
        else:
            max_id = 0
            
        for task in self.tasks:
            if task.task_id is None:
                max_id += 1
                task.task_id = max_id
        
        # Run initial AI prioritization
        self._update_ai_priorities()
    
    def add_task(
        self,
        title: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        due_date: Optional[str] = None,
        manual_priority: Optional[int] = None
    ) -> Task:
        """
        Add a new task to the system.
        
        Args:
            title: The title of the task
            description: Optional description
            category: Optional category
            due_date: Optional due date in YYYY-MM-DD format
            manual_priority: Optional manual priority (1-5)
            
        Returns:
            The created Task object
        """
        if not title:
            raise ValueError("Task title cannot be empty")
        
        # Generate a new ID
        if self.tasks:
            new_id = max(task.task_id for task in self.tasks) + 1
        else:
            new_id = 1
        
        task = Task(
            task_id=new_id,
            title=title,
            description=description,
            category=category,
            due_date=due_date,
            manual_priority=manual_priority,
            status="pending"
        )
        
        self.tasks.append(task)
        
        # Update AI priorities
        self._update_ai_priorities()
        
        # Save to storage
        self.storage.save_tasks(self.tasks)
        
        return task
    
    def list_tasks(
        self,
        category: Optional[str] = None,
        due_date: Optional[str] = None,
        status: Optional[str] = None,
        sort_by: str = "priority"
    ) -> List[Task]:
        """
        List tasks with optional filtering and sorting.
        
        Args:
            category: Filter by category
            due_date: Filter by due date
            status: Filter by status (pending/completed)
            sort_by: Sort by field (priority, due, created)
            
