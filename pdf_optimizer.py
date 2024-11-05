# pdf_optimizer.py
import fitz
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import os
import hashlib
import json
from config import PDFConfig, EnrichmentConfig
from text_processor import TextProcessor

class PDFOptimizer:
    def __init__(self, embed_model, llm):
        self.embed_model = embed_model
        self.llm = llm
        self.text_processor = TextProcessor()
        
    def optimize(self, pdf_path: str) -> Tuple[Optional[str], Dict]:
        """
        Optimize PDF and generate semantic enrichment.
        Returns (optimized_pdf_path, semantic_metadata) tuple.
        """
        try:
            pdf_doc = fitz.open(pdf_path)
            file_hash = self._calculate_file_hash(pdf_path)
            
            # Process document structure
            content_blocks = self._process_document_structure(pdf_doc)
            
            # Generate semantic enrichment
            semantic_metadata = self._generate_semantic_metadata(content_blocks)
            
            # Create optimized PDF if needed
            if len(content_blocks) < len(pdf_doc):
                optimized_path = self._create_optimized_pdf(
                    pdf_doc, 
                    [block['page_num'] for block in content_blocks],
                    pdf_path
                )
            else:
                optimized_path = pdf_path
                
            pdf_doc.close()
            
            # Add file metadata
            semantic_metadata['file_metadata'] = {
                'file_hash': file_hash,
                'file_path': pdf_path,
                'original_pages': len(pdf_doc),
                'optimized_pages': len(content_blocks)
            }
            
            return optimized_path, semantic_metadata
            
        except Exception as e:
            print(f"Error optimizing PDF {pdf_path}: {str(e)}")
            return None, {}

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _process_document_structure(self, pdf_doc: fitz.Document) -> List[Dict]:
        """Process document structure and identify content blocks with enhanced text cleaning."""
        content_blocks = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            raw_text = page.get_text()
            
            # Skip if page doesn't contain meaningful content
            if not self.text_processor.is_content_page(raw_text, PDFConfig.MIN_CONTENT_WORDS):
                continue
                
            # Extract and clean main content
            cleaned_text = self.text_processor.extract_main_content(raw_text)
            if not cleaned_text:
                continue
                
            content_blocks.append({
                'page_num': page_num,
                'text': cleaned_text,
                'preview': cleaned_text[:PDFConfig.MAX_PREVIEW_LENGTH]
            })
        
        return content_blocks

    def _generate_semantic_metadata(self, content_blocks: List[Dict]) -> Dict:
        """Generate semantic enrichment for content blocks."""
        metadata = {
            'sections': [],
            'global_summary': '',
            'key_concepts': []
        }
        
        for block in content_blocks:
            section_metadata = {}
            
            if EnrichmentConfig.GENERATE_LABELS:
                section_metadata['labels'] = self._generate_labels(block['text'])
                
            if EnrichmentConfig.GENERATE_SUMMARIES:
                section_metadata['summary'] = self._generate_summary(block['text'])
                
            if EnrichmentConfig.GENERATE_CONCEPTS:
                section_metadata['concepts'] = self._extract_key_concepts(block['text'])
                
            if EnrichmentConfig.RELEVANCE_SCORING:
                section_metadata['relevance_scores'] = self._generate_relevance_scores(block['text'])
                
            metadata['sections'].append({
                'page_num': block['page_num'],
                'preview': block['preview'],
                **section_metadata
            })
            
        # Generate global metadata if in comprehensive mode
        if PDFConfig.ENRICHMENT_LEVEL == 'comprehensive':
            full_text = "\n".join(block['text'] for block in content_blocks)
            metadata['global_summary'] = self._generate_summary(full_text, is_global=True)
            metadata['key_concepts'] = self._extract_key_concepts(full_text, is_global=True)
            
        return metadata

    def _generate_labels(self, text: str) -> List[str]:
        """Generate semantic labels for text section."""
        prompt = (
            "Generate up to 5 specific semantic labels for this text section. "
            "Labels should be relevant for document retrieval. "
            "Respond with only the labels, separated by commas:\n\n"
            f"{text[:PDFConfig.MAX_PREVIEW_LENGTH]}"
        )
        try:
            response = self.llm.complete(prompt)
            labels = [label.strip() for label in response.text.split(',')]
            return labels[:EnrichmentConfig.MAX_LABELS_PER_SECTION]
        except Exception as e:
            print(f"Error generating labels: {str(e)}")
            return []

    def _generate_summary(self, text: str, is_global: bool = False) -> str:
        """Generate concise summary of text section."""
        max_length = EnrichmentConfig.SUMMARY_MAX_LENGTH * (2 if is_global else 1)
        prompt = (
            f"{'Globally summarize' if is_global else 'Summarize'} the following text "
            f"in {max_length} characters or less, focusing on key points:\n\n"
            f"{text[:PDFConfig.MAX_PREVIEW_LENGTH]}"
        )
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()[:max_length]
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return ""

    def _extract_key_concepts(self, text: str, is_global: bool = False) -> List[str]:
        """Extract key concepts from text."""
        prompt = (
            f"Extract the {'overall' if is_global else 'main'} key concepts from this text. "
            "Respond with only the concepts, separated by commas:\n\n"
            f"{text[:PDFConfig.MAX_PREVIEW_LENGTH]}"
        )
        try:
            response = self.llm.complete(prompt)
            concepts = [concept.strip() for concept in response.text.split(',')]
            return concepts[:EnrichmentConfig.MAX_LABELS_PER_SECTION]
        except Exception as e:
            print(f"Error extracting concepts: {str(e)}")
            return []

    def _generate_relevance_scores(self, text: str) -> Dict[str, any]:
        # Define constants at the start of the method
        REQUIRED_CATEGORIES = {"technical", "theoretical", "practical", "introductory"}
        
        prompt = """
        Analyze this text and provide relevance scores. Respond with only this exact JSON:
        {
            "scores": {
                "technical": 0.0,
                "theoretical": 0.0,
                "practical": 0.0,
                "introductory": 0.0
            },
            "reasoning": {
                "technical": "why this score",
                "theoretical": "why this score",
                "practical": "why this score",
                "introductory": "why this score"
            }
        }

        Replace the 0.0 values with scores between 0 and 1, and the "why this score" with brief explanations.
        Text to analyze:
        """ + text[:PDFConfig.MAX_PREVIEW_LENGTH]
        
        try:
            response = self.llm.complete(prompt)
            # Clean up common JSON formatting issues
            cleaned_response = (response.text.strip()
                            .replace('\n', '')
                            .replace('```json', '')
                            .replace('```', ''))
            
            result = json.loads(cleaned_response)
            
            # Validate structure
            if not all(k in result for k in ("scores", "reasoning")):
                raise ValueError("Missing required top-level keys in response")
                
            if not all(k in result["scores"] for k in REQUIRED_CATEGORIES):
                raise ValueError(f"Missing score categories. Expected: {REQUIRED_CATEGORIES}")
                
            # Validate and clean scores
            for category in REQUIRED_CATEGORIES:
                score = result["scores"][category]
                if not isinstance(score, (int, float)):
                    result["scores"][category] = float(score)  # Try to convert
                score = result["scores"][category]
                if not 0 <= score <= 1:
                    raise ValueError(f"Score for {category} must be between 0 and 1")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}\nResponse was: {response.text}")
            return {
                "scores": {cat: 0.0 for cat in REQUIRED_CATEGORIES},
                "reasoning": {cat: "Error parsing LLM response" for cat in REQUIRED_CATEGORIES}
            }
        except Exception as e:
            print(f"Error generating relevance scores: {str(e)}")
            return {
                "scores": {cat: 0.0 for cat in REQUIRED_CATEGORIES},
                "reasoning": {cat: f"Error: {str(e)}" for cat in REQUIRED_CATEGORIES}
            }

    @staticmethod
    def batch_optimize(pdf_dir: str, embed_model, llm) -> List[Tuple[str, Dict]]:
        """
        Optimize all PDFs in a directory.
        Returns list of (optimized_path, metadata) tuples.
        """
        optimizer = PDFOptimizer(embed_model, llm)
        results = []
        
        for pdf_file in Path(pdf_dir).glob("**/*.pdf"):
            print(f"Optimizing {pdf_file.name}...")
            result = optimizer.optimize(str(pdf_file))
            if result[0]:  # if optimization was successful
                results.append(result)
                print(f"Processed {pdf_file.name}")
                
        return results
    
    def _create_optimized_pdf(self, pdf_doc: fitz.Document, content_pages: List[int], original_path: str) -> str:
        # Create new PDF with only content pages
        optimized_doc = fitz.open()
    
        for page_num in content_pages:
            optimized_doc.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)
        
        # Generate optimized filename
        optimized_path = str(Path(original_path).parent / f"optimized_{Path(original_path).name}")
        
        # Save optimized PDF
        optimized_doc.save(optimized_path)
        optimized_doc.close()
        
        return optimized_path