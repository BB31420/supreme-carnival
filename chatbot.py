# chatbot.py
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from database import Database
from models import Models
from document_processor import DocumentProcessor
from vector_store import VectorStore
from config import RetrieverConfig, DBConfig, DocumentProcessorConfig

class Chatbot:
    def __init__(self, document_directory):
        self.document_directory = document_directory
        self.setup()

    def setup(self):
        """Initialize the chatbot components."""
        # Initialize database if needed
        Database.setup()
        Database.create_table()
        
        should_process = Database.initialize_if_empty() or DBConfig.FORCE_REPROCESS
        
        self.models = Models()
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        
        # Initialize node parser
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=DocumentProcessorConfig.CHUNK_SIZE,
            chunk_overlap=DocumentProcessorConfig.CHUNK_OVERLAP
        )

        if should_process:
            if DBConfig.FORCE_REPROCESS:
                Database.clear(force=True)
            self._process_documents()
            
        self.setup_query_engine()  # Changed from _setup_query_engine to setup_query_engine

    def _process_documents(self):
        """Process documents and add to vector store."""
        try:
            # Process documents
            processed_contents = list(
                self.document_processor.process_directory(self.document_directory)
            )
            
            # Convert to LlamaIndex documents
            documents = self.document_processor.create_documents(processed_contents)
            
            # Create nodes with embeddings
            nodes = []
            for doc in documents:
                doc_nodes = self.node_parser.get_nodes_from_documents([doc])
                for node in doc_nodes:
                    # Add embedding to node
                    node.embedding = self.models.embed_model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                    )
                    nodes.append(node)
            
            # Add nodes to vector store
            if nodes:
                self.vector_store.add_nodes(nodes)
                print(f"Successfully processed {len(nodes)} nodes from {len(documents)} documents")
            else:
                print("No nodes were created from the documents")
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            raise

    def setup_query_engine(self):
        """Set up the query engine."""
        Settings.llm = self.models.llm
        Settings.embed_model = self.models.embed_model
        
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store.store
        )
        
        self.index = VectorStoreIndex(
            [],
            storage_context=storage_context,
            show_progress=True
        )
        
        self.query_engine = self.index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True
        )

    def chat_loop(self):
        """Run the chat interaction loop."""
        print("3D Printing Chatbot Ready - Type 'quit' to exit")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
                
            try:
                # Get response
                response = self.query_engine.query(user_input)
                
                # Print main response
                print("\nChatbot:", response)
                
                # Print source information
                print("\nSources:")
                for source_node in response.source_nodes:
                    print(f"\nFrom: {source_node.node.metadata.get('source', 'Unknown')}")
                    print(f"Page: {source_node.node.metadata.get('page', 'Unknown')}")
                    if 'technical_metadata' in source_node.node.metadata:
                        print("Technical Context:", source_node.node.metadata['technical_metadata'])
                    
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                print("Please try rephrasing your question.")