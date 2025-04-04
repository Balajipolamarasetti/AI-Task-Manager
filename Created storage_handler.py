"""
Storage Handler Module
Manages data persistence for the task management system.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class StorageHandler:
    """
    Handles data persistence operations, including saving and loading tasks.
    """
    
    def __init__(self, storage_file: str = "tasks.dat"):
        """
        Initialize the storage handler.
        
        Args:
            storage_file: Path to the file where tasks will be stored
        """
        self.storage_file = storage_file
        self.data_dir = os.path.expanduser("~/.aitaskmanager")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
                logger.info(f"Created data directory at {self.data_dir}")
            except Exception as e:
                logger.error(f"Failed to create data directory: {e}")
                # Fall back to current directory
                self.data_dir = "."
        
        # Full path to storage file
        self.storage_path = os.path.join(self.data_dir, self.storage_file)
    
    def save_tasks(self, tasks: List[Any]) -> bool:
        """
        Save tasks to persistent storage.
        
        Args:
            tasks: List of Task objects to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert tasks to dictionaries for storage
            task_dicts = [task.to_dict() for task in tasks]
            
            # Save to file using pickle
            with open(self.storage_path, 'wb') as f:
                pickle.dump(task_dicts, f)
            
            logger.debug(f"Saved {len(tasks)} tasks to {self.storage_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving tasks: {e}")
            return False
    
    def load_tasks(self) -> List[Any]:
        """
        Load tasks from persistent storage.
        
        Returns:
            List of Task objects
        """
        if not os.path.exists(self.storage_path):
            logger.info(f"No task data found at {self.storage_path}")
            return []
        
        try:
            # Load data from file
            with open(self.storage_path, 'rb') as f:
                task_dicts = pickle.load(f)
            
            # Import Task class here to avoid circular import
            from task_manager import Task
            
            # Convert dictionaries back to Task objects
            tasks = [Task.from_dict(task_dict) for task_dict in task_dicts]
            
            logger.debug(f"Loaded {len(tasks)} tasks from {self.storage_path}")
            return tasks
        except Exception as e:
            logger.error(f"Error loading tasks: {e}")
            return []
    
    def export_tasks(self, tasks: List[Any], filename: str) -> bool:
        """
        Export tasks to a specified file.
        
        Args:
            tasks: List of Task objects to export
            filename: Name of the export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert tasks to dictionaries
            task_dicts = [task.to_dict() for task in tasks]
            
            # Ensure the filename has the correct extension
            if not filename.endswith('.dat'):
                filename += '.dat'
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(task_dicts, f)
            
            logger.info(f"Exported {len(tasks)} tasks to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting tasks: {e}")
            return False
    
    def import_tasks(self, filename: str) -> List[Any]:
        """
        Import tasks from a specified file.
        
        Args:
            filename: Name of the file to import from
            
        Returns:
            List of imported Task objects
        """
        if not os.path.exists(filename):
            logger.error(f"Import file not found: {filename}")
            return []
        
        try:
            # Load data from file
            with open(filename, 'rb') as f:
                task_dicts = pickle.load(f)
            
            # Import Task class here to avoid circular import
            from task_manager import Task
            
            # Convert dictionaries to Task objects
            tasks = [Task.from_dict(task_dict) for task_dict in task_dicts]
            
            logger.info(f"Imported {len(tasks)} tasks from {filename}")
            return tasks
        except Exception as e:
            logger.error(f"Error importing tasks: {e}")
            return []
