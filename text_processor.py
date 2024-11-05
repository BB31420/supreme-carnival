# text_processor.py
import re
import unicodedata
from typing import Optional

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing special characters, normalizing Unicode, 
        and handling PDF artifacts.
        """
        if not text:
            return ""
            
        # Handle None or non-string input
        if not isinstance(text, str):
            return ""
            
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != "C" or char in '\n\t')
        
        # Remove PDF artifacts and page numbers
        text = re.sub(r'(\x00|\x01|\x02|\x03|\x04|\x05|\x06|\x07|\x08).*?(\x00|\x01|\x02|\x03|\x04|\x05|\x06|\x07|\x08)', '', text)
        text = re.sub(r'\d+\s*\n\s*[A-Z][A-Za-z\s]*\n', '', text)  # Remove table of contents entries
        text = re.sub(r'\.{3,}|�{3,}', '', text)  # Remove ellipsis and repeated special characters
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[A-Za-z\s]+\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    @staticmethod
    def is_content_page(text: str, min_words: int = 50) -> bool:
        """
        Determine if a page contains meaningful content.
        """
        cleaned_text = TextProcessor.clean_text(text)
        word_count = len(cleaned_text.split())
        
        # Check for common non-content markers
        non_content_markers = [
            r'^\s*table\s+of\s+contents\s*$',
            r'^\s*index\s*$',
            r'^\s*bibliography\s*$',
            r'^\s*references\s*$',
            r'^\s*copyright\s*$'
        ]
        
        for marker in non_content_markers:
            if re.search(marker, cleaned_text, re.IGNORECASE):
                return False
                
        return word_count >= min_words

    @staticmethod
    def extract_main_content(text: str) -> Optional[str]:
        """
        Extract main content from text, removing headers, footers, and other artifacts.
        """
        if not text:
            return None
            
        # Clean the text first
        text = TextProcessor.clean_text(text)
        
        # Split into lines and process
        lines = text.split('\n')
        content_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Skip likely headers/footers
            if len(line.strip()) < 50 and any([
                re.match(r'^\s*\d+\s*$', line),  # Page numbers
                re.match(r'^\s*chapter\s+\d+\s*$', line, re.IGNORECASE),  # Chapter headers
                re.match(r'^\s*[A-Za-z\s]+\s*\|\s*\d+\s*$', line),  # Headers with page numbers
            ]):
                continue
                
            content_lines.append(line)
            
        return '\n'.join(content_lines)