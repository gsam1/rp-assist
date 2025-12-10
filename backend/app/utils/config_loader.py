"""Configuration loader for TOML config files"""

import toml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and validate configuration from TOML file"""

    def __init__(self, config_path: str = "config.toml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from TOML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please copy config.example.toml to {self.config_path}"
            )

        with open(self.config_path, 'r') as f:
            self._config = toml.load(f)

        self._validate()

    def _validate(self) -> None:
        """Validate configuration has required sections"""
        required_sections = ['app', 'llm', 'vectorstore', 'embedding']

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section"""
        return self._config.get(section, {})

    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config


# Global config instance
_config: ConfigLoader = None


def get_config() -> ConfigLoader:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


def reload_config() -> None:
    """Reload the configuration"""
    global _config
    _config = None
    get_config()
