"""
Configuration management for the C60 AutoML framework.

This module provides a centralized configuration system for the C60 AutoML framework.
It handles loading, validating, and accessing configuration settings.
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path


class Config:
    """
    Centralized configuration management for the C60 AutoML framework.
    
    This class provides a singleton pattern for accessing configuration parameters
    throughout the application. It supports loading from dictionaries, YAML files,
    and environment variables.
    """
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        """Ensure only one instance of Config exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Initialize with default configuration
            cls._instance._config = cls._get_default_config()
        return cls._instance
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Return the default configuration."""
        return {
            'data': {
                'train_path': 'data/train.csv',
                'test_path': 'data/test.csv',
                'validation_split': 0.2,
            },
            'model': {
                'type': 'classifier',  # 'classifier' or 'regressor'
                'optimize_metric': 'accuracy',
                'random_state': 42,
            },
            'training': {
                'max_epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_patience': 10,
            },
            'evolution': {
                'population_size': 50,
                'n_generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
            },
            'paths': {
                'output_dir': 'output',
                'model_dir': 'models',
                'log_dir': 'logs',
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'c60.log',
            }
        }
    
    def update(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.
        
        Args:
            config: Dictionary containing configuration updates.
        """
        self._update_dict(self._config, config)
    
    def _update_dict(self, original: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary with values from another dictionary.
        
        Args:
            original: The dictionary to update.
            updates: Dictionary containing updates to apply.
        """
        for key, value in updates.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._update_dict(original[key], value)
            else:
                original[key] = value
    
    def from_yaml(self, filepath: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            filepath: Path to the YAML configuration file.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            self.update(config or {})
    
    def from_env(self) -> None:
        """Update configuration from environment variables."""
        # This is a simplified example - in a real implementation, you would
        # map environment variables to configuration keys
        for key, value in os.environ.items():
            if key.startswith('C60_'):
                # Convert C60_MODEL_TYPE to model.type
                parts = key[4:].lower().split('__')
                current = self._config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = self._parse_env_value(value)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable string values into appropriate Python types."""
        # Try to convert to int, float, bool, or keep as string
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., 'model.type')
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default if not found
        """
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value using dictionary-style access."""
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the current configuration as a dictionary."""
        # Use json to create a deep copy
        import json
        return json.loads(json.dumps(self._config))
    
    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        import json
        return json.dumps(self._config, indent=2, sort_keys=True)


# Create a default instance for easy importing
config = Config()
