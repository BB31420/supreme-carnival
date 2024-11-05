# vector_store.py
from llama_index.vector_stores.postgres import PGVectorStore
from config import DBConfig, ModelConfig

class VectorStore:
    def __init__(self):
        self.store = self._create_vector_store()

    def _create_vector_store(self):
        return PGVectorStore.from_params(
            table_name=DBConfig.TABLE_NAME,
            embed_dim=ModelConfig.EMBED_DIM,
            host=DBConfig.HOST,
            password=DBConfig.PASSWORD,
            port=DBConfig.PORT,
            user=DBConfig.USER,
            database=DBConfig.DATABASE
        )

    def add_nodes(self, nodes):
        self.store.add(nodes)

    def query(self, vector_store_query):
        return self.store.query(vector_store_query)