# models.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import ModelConfig

class Models:
    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(model_name=ModelConfig.EMBED_NAME)
        self.llm = self._setup_llm()
        self.rerank_model, self.rerank_tokenizer = self._setup_rerank_model()

    def _setup_llm(self):
        return LlamaCPP(
            model_url=ModelConfig.URL,
            temperature=ModelConfig.LLM_TEMPERATURE,
            max_new_tokens=ModelConfig.LLM_MAX_NEW_TOKENS,
            context_window=ModelConfig.LLM_CONTEXT_WINDOW,
            generate_kwargs={},
            model_kwargs={
                "n_gpu_layers": ModelConfig.LLM_N_GPU_LAYERS,
                "use_mmap": False,
                "use_mlock": False,
                "n_batch": ModelConfig.LLM_N_BATCH,
                "n_ctx": ModelConfig.LLM_N_CTX,
                "offload_kqv": True,
            },
            verbose=True,
        )

    def _setup_rerank_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(ModelConfig.RERANK_NAME)
        tokenizer = AutoTokenizer.from_pretrained(ModelConfig.RERANK_NAME)
        return model, tokenizer