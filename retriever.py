# retriever.py
from dataclasses import dataclass
from typing import List, Dict, Optional
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery

@dataclass
class ElaboratedQuery:
    original_query: str
    elaborated_text: str
    focus_areas: List[str]
    technical_depth: float  # 0-1 scale
    metadata_filters: Dict[str, any]

class EnhancedRetriever(BaseRetriever):
    """Enhanced retriever with query elaboration and contextual retrieval."""
    
    def __init__(
        self,
        vector_store,
        embed_model,
        llm,
        similarity_top_k: int = 12,
        elaboration_temperature: float = 0.2
    ):
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._llm = llm
        self._similarity_top_k = similarity_top_k
        self._elaboration_temperature = elaboration_temperature
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # First, elaborate the query
        elaborated_query = self._elaborate_query(query_bundle.query_str)
        
        # Get embeddings for both original and elaborated queries
        original_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        elaborated_embedding = self._embed_model.get_query_embedding(
            elaborated_query.elaborated_text
        )
        
        # Retrieve nodes using both embeddings
        original_nodes = self._vector_search(
            original_embedding,
            elaborated_query.metadata_filters
        )
        elaborated_nodes = self._vector_search(
            elaborated_embedding,
            elaborated_query.metadata_filters
        )
        
        # Combine and deduplicate results
        combined_nodes = self._combine_results(
            original_nodes,
            elaborated_nodes,
            elaborated_query
        )
        
        return combined_nodes

    def _elaborate_query(self, query: str) -> ElaboratedQuery:
        """Elaborate the query to improve retrieval."""
        prompt = f"""Given this user query about 3D printing, analyze it and help improve retrieval by providing:
        1. An elaborated version that includes important context and terms
        2. Key technical focus areas (comma-separated)
        3. Technical depth score (0-1)
        4. Suggested document sections to focus on

        User Query: {query}
        
        Respond with each element on a new line, no labels needed."""
        
        response = self._llm.complete(
            prompt,
            temperature=self._elaboration_temperature
        )
        
        # Parse response (flexible format)
        lines = [line.strip() for line in response.text.split('\n') if line.strip()]
        
        elaborated_text = lines[0] if len(lines) > 0 else query
        focus_areas = [
            area.strip() 
            for area in (lines[1].split(',') if len(lines) > 1 else [])
        ]
        technical_depth = float(lines[2]) if len(lines) > 2 else 0.5
        
        # Create metadata filters based on focus areas
        metadata_filters = {
            'technical_depth': {
                'gte': max(0.0, technical_depth - 0.2),
                'lte': min(1.0, technical_depth + 0.3)
            }
        }
        
        return ElaboratedQuery(
            original_query=query,
            elaborated_text=elaborated_text,
            focus_areas=focus_areas,
            technical_depth=technical_depth,
            metadata_filters=metadata_filters
        )

    def _vector_search(
        self,
        embedding: List[float],
        metadata_filters: Optional[Dict] = None
    ) -> List[NodeWithScore]:
        """Perform vector search with metadata filtering."""
        vector_store_query = VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=self._similarity_top_k,
            metadata_filters=metadata_filters
        )
        
        results = self._vector_store.query(vector_store_query)
        return results.nodes

    def _combine_results(
        self,
        original_nodes: List[NodeWithScore],
        elaborated_nodes: List[NodeWithScore],
        elaborated_query: ElaboratedQuery
    ) -> List[NodeWithScore]:
        """Combine and rerank results from both queries."""
        # Create a dictionary to deduplicate nodes
        seen_nodes = {}
        
        # Process original nodes first
        for node in original_nodes:
            node_id = node.node.node_id
            if node_id not in seen_nodes:
                seen_nodes[node_id] = node
        
        # Add elaborated nodes with adjusted scores
        for node in elaborated_nodes:
            node_id = node.node.node_id
            if node_id not in seen_nodes:
                seen_nodes[node_id] = node
            else:
                # Average the scores if we found the node in both searches
                existing_score = seen_nodes[node_id].score
                new_score = (existing_score + node.score) / 2
                seen_nodes[node_id].score = new_score
        
        # Convert back to list and sort by score
        combined = list(seen_nodes.values())
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined[:self._similarity_top_k]