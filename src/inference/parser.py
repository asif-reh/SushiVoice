#!/usr/bin/env python3
"""
Command Parser for SushiVoice
Extracts quantity and item name from transcribed text using regex
"""

import re
from typing import Optional, Dict


class CommandParser:
    """Parse sushi labeling commands"""
    
    def __init__(self):
        """Initialize parser with word-to-number mappings"""
        self.word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }
        
        # Compile regex patterns
        self.patterns = [
            # Pattern 1: "print/label X times (of) (label of) ITEM"
            r'(?:print|label|make|create)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+(?:times?\s*(?:of)?\s*)?(?:label[s]?\s*(?:of|for)?\s*)?(.+?)$',
            
            # Pattern 2: "print X ITEM" (short form)
            r'(?:print|label|make|create)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+(.+?)$',
            
            # Pattern 3: "X times ITEM" (without action verb)
            r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+(?:times?\s*(?:of)?\s*)?(?:label[s]?\s*(?:of|for)?\s*)?(.+?)$',
        ]
    
    def word_to_number(self, word: str) -> int:
        """Convert word to number"""
        word = word.lower().strip()
        if word.isdigit():
            return int(word)
        return self.word_to_num.get(word, 1)
    
    def clean_item_name(self, item: str) -> str:
        """Clean and format item name"""
        # Remove wake words if present
        wake_words = ['hey yosushi', 'yosushi', 'hello yosushi', 'hi yosushi']
        item_lower = item.lower()
        for wake in wake_words:
            if wake in item_lower:
                item_lower = item_lower.replace(wake, '')
        
        # Remove extra whitespace
        item = ' '.join(item_lower.split())
        
        # Title case
        item = item.strip().title()
        
        return item
    
    def parse(self, transcript: str, confidence: float = 1.0) -> Optional[Dict]:
        """
        Parse command from transcript
        
        Args:
            transcript: Transcribed text
            confidence: ASR confidence score (0-1)
        
        Returns:
            Dictionary with 'quantity', 'item', 'confidence', or None if parsing fails
        """
        if not transcript:
            return None
        
        transcript = transcript.lower().strip()
        
        # Try each pattern
        for pattern in self.patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                quantity_str = match.group(1)
                item_raw = match.group(2)
                
                # Convert quantity to number
                quantity = self.word_to_number(quantity_str)
                
                # Clean item name
                item = self.clean_item_name(item_raw)
                
                # Validate
                if item and len(item) > 0 and 1 <= quantity <= 20:
                    return {
                        'quantity': quantity,
                        'item': item,
                        'confidence': confidence,
                        'raw_transcript': transcript
                    }
        
        # No pattern matched
        return None
    
    def parse_with_fallback(self, transcript: str, confidence: float = 1.0) -> Optional[Dict]:
        """
        Parse with fallback heuristics
        
        If standard parsing fails, try to extract:
        - Last word/phrase as item name
        - First number found as quantity
        """
        result = self.parse(transcript, confidence)
        if result:
            return result
        
        # Fallback: look for any number and assume rest is item
        transcript_clean = transcript.lower().strip()
        
        # Find first number
        number_match = re.search(r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)', transcript_clean)
        
        if number_match:
            quantity = self.word_to_number(number_match.group(1))
            
            # Extract item (everything after the number)
            item_start = number_match.end()
            item_raw = transcript_clean[item_start:].strip()
            
            # Remove common words
            item_raw = re.sub(r'(times?|of|label[s]?|for|the)', '', item_raw, flags=re.IGNORECASE)
            item = self.clean_item_name(item_raw)
            
            if item and len(item) > 2:
                return {
                    'quantity': quantity,
                    'item': item,
                    'confidence': confidence * 0.8,  # Lower confidence for fallback
                    'raw_transcript': transcript,
                    'fallback': True
                }
        
        return None


if __name__ == '__main__':
    # Test parser
    print("Testing Command Parser...\n")
    
    parser = CommandParser()
    
    test_cases = [
        "hey yosushi print 5 times of label of chicken teriyaki",
        "yosushi label 3 times california roll",
        "print 10 labels of salmon nigiri",
        "make two tuna sashimi",
        "label fifteen spicy tuna roll",
        "print seven labels for dragon roll",
        "5 times of chicken teriyaki",  # No action verb
        "three salmon",  # Minimal
        "hey yosushi 8 tempura shrimp",  # Wake word present
        "random text with no pattern",  # Should fail
    ]
    
    for i, test in enumerate(test_cases, 1):
        result = parser.parse_with_fallback(test, confidence=0.95)
        print(f"{i}. Input: '{test}'")
        if result:
            print(f"   ✅ Parsed: {result['quantity']}x {result['item']} (confidence: {result['confidence']:.2f})")
            if result.get('fallback'):
                print(f"   ⚠️  Used fallback parsing")
        else:
            print(f"   ❌ Failed to parse")
        print()
    
    print("✅ Parser test complete!")
