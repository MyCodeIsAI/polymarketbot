"""Configuration loader with YAML and environment variable support.

This module handles loading configuration from YAML files and environment
variables, with validation and error reporting.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from .models import AppConfig


# Global config instance (singleton pattern)
_config: Optional[AppConfig] = None


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file."""
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")


def _resolve_env_references(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve ${ENV_VAR} references in string values."""
    result = {}

    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _resolve_env_references(value)
        elif isinstance(value, list):
            result[key] = [
                _resolve_env_references(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            env_value = os.getenv(env_var)
            if env_value is None:
                raise ConfigError(f"Environment variable not set: {env_var}")
            result[key] = env_value
        else:
            result[key] = value

    return result


def load_config(
    config_dir: Optional[Path] = None,
    accounts_file: str = "accounts.yaml",
    settings_file: str = "settings.yaml",
    env_file: Optional[str] = ".env",
) -> AppConfig:
    """Load application configuration from YAML files and environment.

    Args:
        config_dir: Directory containing config files. Defaults to ./config
        accounts_file: Name of accounts config file
        settings_file: Name of settings config file
        env_file: Name of .env file (relative to project root), or None to skip

    Returns:
        Validated AppConfig instance

    Raises:
        ConfigError: If configuration is invalid or files not found
    """
    global _config

    # Determine config directory
    if config_dir is None:
        # Look for config dir relative to current working directory
        config_dir = Path.cwd() / "config"
        if not config_dir.exists():
            # Try relative to this file's package
            config_dir = Path(__file__).parent.parent.parent / "config"

    config_dir = Path(config_dir)

    # Load .env file if specified
    if env_file:
        env_path = config_dir.parent / env_file
        if env_path.exists():
            load_dotenv(env_path)

    # Load YAML files
    accounts_path = config_dir / accounts_file
    settings_path = config_dir / settings_file

    accounts_data = _load_yaml_file(accounts_path)
    settings_data = _load_yaml_file(settings_path) if settings_path.exists() else {}

    # Merge configurations (accounts takes precedence for overlapping keys)
    merged_data = _deep_merge(settings_data, accounts_data)

    # Resolve environment variable references
    resolved_data = _resolve_env_references(merged_data)

    # Validate and create config object
    try:
        _config = AppConfig.model_validate(resolved_data)
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}")

    return _config


def get_config() -> AppConfig:
    """Get the loaded configuration.

    Returns:
        The loaded AppConfig instance

    Raises:
        ConfigError: If configuration has not been loaded
    """
    if _config is None:
        raise ConfigError("Configuration not loaded. Call load_config() first.")
    return _config


def get_env_value(env_var_name: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable value.

    This is a convenience function for accessing secrets that are
    referenced in config files but stored in environment variables.

    Args:
        env_var_name: Name of the environment variable
        default: Default value if not set

    Returns:
        The environment variable value or default
    """
    return os.getenv(env_var_name, default)


def require_env_value(env_var_name: str) -> str:
    """Get a required environment variable value.

    Args:
        env_var_name: Name of the environment variable

    Returns:
        The environment variable value

    Raises:
        ConfigError: If the environment variable is not set
    """
    value = os.getenv(env_var_name)
    if value is None:
        raise ConfigError(f"Required environment variable not set: {env_var_name}")
    return value


def validate_required_secrets(config: AppConfig) -> list[str]:
    """Check that all required secrets are set in environment.

    Args:
        config: The loaded configuration

    Returns:
        List of missing environment variable names (empty if all set)
    """
    required_env_vars = [
        config.your_account.private_key_env,
        config.api_credentials.api_key_env,
        config.api_credentials.api_secret_env,
        config.api_credentials.api_passphrase_env,
    ]

    missing = []
    for var in required_env_vars:
        if os.getenv(var) is None:
            missing.append(var)

    return missing
