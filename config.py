# config.py
import os

class DBConfig:
    HOST = os.getenv('DB_HOST', 'localhost')
    PASSWORD = os.getenv('DB_PASSWORD', 'password')
    PORT = os.getenv('DB_PORT', '5432')
    USER = os.getenv('DB_USER', 'user')
    DATABASE = os.getenv('DB_NAME', '3d_printing_db')
    TABLE_NAME = 'pdf_3d_printing_documents'
    #FORCE_REPROCESS = os.getenv('FORCE_REPROCESS', 'false').lower() == 'true'
    FORCE_REPROCESS = True
class ModelConfig:
    URL = os.getenv('MODEL_URL', "https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored_V2_Q4.gguf")
    RERANK_NAME = os.getenv('RERANK_MODEL_NAME', "cross-encoder/ms-marco-MiniLM-L-6-v2")
    EMBED_NAME = os.getenv('EMBED_MODEL_NAME', "BAAI/bge-small-en-v1.5")
    EMBED_DIM = int(os.getenv('EMBED_DIM', 384))
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.1))
    LLM_MAX_NEW_TOKENS = int(os.getenv('LLM_MAX_NEW_TOKENS', 512))
    LLM_CONTEXT_WINDOW = int(os.getenv('LLM_CONTEXT_WINDOW', 4906))
    LLM_N_GPU_LAYERS = int(os.getenv('LLM_N_GPU_LAYERS', -1))
    LLM_N_BATCH = int(os.getenv('LLM_N_BATCH', 256))
    LLM_N_CTX = int(os.getenv('LLM_N_CTX', 3904))

class PDFConfig:
    ENRICHMENT_LEVEL = os.getenv('PDF_ENRICHMENT_LEVEL', 'comprehensive')  # 'basic' or 'comprehensive'
    FRONT_MATTER_MARKERS = [
        "copyright", "all rights reserved", "table of contents",
        "preface", "foreword", "acknowledgments", "author"
    ]
    BACK_MATTER_MARKERS = [
        "appendix", "glossary", "bibliography", "index",
        "references", "about the author", "author"
    ]
    MIN_CONTENT_WORDS = int(os.getenv('MIN_CONTENT_WORDS', 50))
    MAX_PREVIEW_LENGTH = int(os.getenv('MAX_PREVIEW_LENGTH', 1000))

class EnrichmentConfig:
    GENERATE_SUMMARIES = os.getenv('GENERATE_SUMMARIES', 'true').lower() == 'true'
    GENERATE_LABELS = os.getenv('GENERATE_LABELS', 'true').lower() == 'true'
    GENERATE_CONCEPTS = os.getenv('GENERATE_CONCEPTS', 'true').lower() == 'true'
    RELEVANCE_SCORING = os.getenv('RELEVANCE_SCORING', 'true').lower() == 'true'
    MAX_LABELS_PER_SECTION = int(os.getenv('MAX_LABELS_PER_SECTION', 5))
    SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 200))
    
class RetrieverConfig:
    SIMILARITY_TOP_K = int(os.getenv('SIMILARITY_TOP_K', 12))
    RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', 4))

class DocumentProcessorConfig:
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 512))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 128))