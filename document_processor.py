# document_processor.py
from pathlib import Path
import fitz
from typing import List, Dict, Generator
from dataclasses import dataclass
from llama_index.core import Document
import re

@dataclass
class ProcessedContent:
    text: str
    metadata: Dict
    source_page: int
    source_file: str

class DocumentProcessor:
    """Simplified document processor without LLM dependencies."""
    
    def __init__(self):
        # Common patterns for non-content sections
        self.noise_patterns = [
            r'^\s*copyright\s+©',
            r'^\s*all\s+rights\s+reserved',
            r'^\s*table\s+of\s+contents',
            r'^\s*index\s*$',
            r'^\s*appendix\s*[a-z]?\s*$',
            r'^\s*references\s*$',
            r'^\s*bibliography\s*$',
            r'^\s*page\s+\d+\s*$',
            # 3D Printing specific patterns
            r'^\s*printer\s+settings\s*$',
            r'^\s*material\s+safety\s+data\s*$',
            r'^\s*printer\s+specifications\s*$'
        ]
        
        # Configuration
        self.min_words = 50
        self.chunk_size = 1500
        self.chunk_overlap = 200
        
        # 3D Printing specific metadata tags
        self.technical_patterns = {
            'slicer_settings': r'(?i)(layer height|infill|support structures|retraction)',
            'materials': r'(?i)(PLA|ABS|PETG|TPU|resin)',
            'hardware': r'(?i)(nozzle|extruder|build plate|hot end)',
            'post_processing': r'(?i)(sanding|painting|acetone|heat treatment)',
            'troubleshooting': r'(?i)(stringing|warping|adhesion|layer shifting)'
        }

    def process_directory(self, directory: str) -> Generator[ProcessedContent, None, None]:
        """Process all PDFs in directory."""
        pdf_files = Path(directory).glob("**/*.pdf")
        for pdf_path in pdf_files:
            try:
                print(f"Processing {pdf_path.name}...")
                yield from self.process_pdf(str(pdf_path))
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue

    def process_pdf(self, pdf_path: str) -> Generator[ProcessedContent, None, None]:
        """Process single PDF file."""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract and clean content page by page
            current_text = ""
            current_page = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Skip if page appears to be noise
                if self._is_noise_page(page_text):
                    continue
                    
                # Clean the page text
                cleaned_text = self._clean_text(page_text)
                if not cleaned_text:
                    continue
                
                # Accumulate text until we reach chunk size
                current_text += cleaned_text + " "
                
                if len(current_text) >= self.chunk_size:
                    # Process the chunk
                    yield from self._process_chunk(
                        current_text, 
                        current_page, 
                        page_num, 
                        pdf_path
                    )
                    
                    # Keep overlap for context
                    current_text = current_text[-self.chunk_overlap:]
                    current_page = page_num
            
            # Process any remaining text
            if len(current_text.strip()) > self.min_words:
                yield from self._process_chunk(
                    current_text, 
                    current_page, 
                    len(doc)-1, 
                    pdf_path
                )
                
            doc.close()
            
        except Exception as e:
            print(f"Error in process_pdf for {pdf_path}: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
            
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        
        # Skip if matches noise patterns
        if any(re.search(pattern, text_lower) for pattern in self.noise_patterns):
            return ""
            
        # Basic cleaning
        cleaned = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', cleaned)  # Remove control chars
        cleaned = re.sub(r'(?<=[.!?])\s+', '\n', cleaned)  # Split sentences
        cleaned = cleaned.strip()
        
        return cleaned

    def _is_noise_page(self, text: str) -> bool:
        """Check if page appears to be non-content."""
        text_lower = text.lower()
        
        # Check word count
        if len(text_lower.split()) < self.min_words:
            return True
            
        # Check noise patterns
        if any(re.search(pattern, text_lower) for pattern in self.noise_patterns):
            return True
            
        # Check if page is mostly numbers (like an index)
        words = text_lower.split()
        numbers = sum(1 for word in words if re.match(r'^\d+$', word))
        if numbers > len(words) * 0.5:
            return True
            
        return False

    def _extract_technical_metadata(self, text: str) -> Dict[str, List[str]]:
        """Extract 3D printing technical metadata from text."""
        metadata = {}
        
        for category, pattern in self.technical_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                metadata[category] = list(set(matches))
                
        return metadata
        
    def _process_chunk(
        self, 
        text: str, 
        start_page: int, 
        end_page: int, 
        source_file: str
    ) -> Generator[ProcessedContent, None, None]:
        """Process text chunk into ProcessedContent objects."""
        
        # Split into meaningful chunks
        chunks = self._split_into_chunks(text)
        
        for chunk in chunks:
            if len(chunk.split()) < self.min_words:
                continue
                
            # Extract technical metadata
            technical_metadata = self._extract_technical_metadata(chunk)
            
            metadata = {
                'source_file': source_file,
                'start_page': start_page,
                'end_page': end_page,
                'chunk_length': len(chunk),
                'word_count': len(chunk.split()),
                'technical_metadata': technical_metadata
            }
            
            yield ProcessedContent(
                text=chunk,
                metadata=metadata,
                source_page=start_page,
                source_file=source_file
            )

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into logical chunks."""
        # First try to split at paragraph boundaries
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def create_documents(self, processed_contents: List[ProcessedContent]) -> List[Document]:
        """Convert ProcessedContent to LlamaIndex Documents."""
        documents = []
        
        for content in processed_contents:
            doc = Document(
                text=content.text,
                metadata={
                    'source': content.source_file,
                    'page': content.source_page,
                    **content.metadata
                }
            )
            documents.append(doc)
            
        return documents