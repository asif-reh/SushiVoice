#!/usr/bin/env python3
"""Label formatting utilities for SushiVoice"""


def format_label_text(item_name: str, quantity: int = 1, max_width: int = 20) -> str:
    """
    Format label text for printing
    
    Args:
        item_name: Item name
        quantity: Number of labels
        max_width: Maximum characters per line
    
    Returns:
        Formatted label text
    """
    # Center item name
    centered_name = item_name.center(max_width)
    
    # Add separator
    separator = "-" * max_width
    
    # Build label
    label = f"{centered_name}\n{separator}\n"
    
    if quantity > 1:
        label += f"Quantity: {quantity}\n"
    
    return label


def wrap_text(text: str, max_width: int = 20) -> list:
    """
    Wrap text to fit printer width
    
    Args:
        text: Text to wrap
        max_width: Maximum characters per line
    
    Returns:
        List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_width:
            current_line += f"{word} "
        else:
            lines.append(current_line.strip())
            current_line = f"{word} "
    
    if current_line:
        lines.append(current_line.strip())
    
    return lines
