"""Utilities for handling configuration templates."""

from typing import Any


def save_config_to(
    config_template_path: str, config_path: str, replacements: dict[str, Any]
) -> None:
    """Save config to a file with replacements."""
    with open(config_template_path, mode="r", encoding="utf-8") as config_template_file:
        config_template = config_template_file.read()
    for key, value in replacements.items():
        config_template = config_template.replace(key, str(value))
    with open(config_path, mode="w", encoding="utf-8") as config_file:
        config_file.write(config_template)
