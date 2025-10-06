import pandas as pd
import os
from sodapy import Socrata
from typing import Optional, List,Dict,Any
from parsel import Selector
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStorePaper():
    def __init__(self, html_articles: Optional[Dict[str, Any]] = None, persist_directory: str = "", recreate_index: bool = True):
        self.html_articles = html_articles
        self.persist_directory = persist_directory
        self.recreate_index = recreate_index
                # Initialize components
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[Any] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.html_splitter : Optional[HTMLSectionSplitter] = None

        self.initialize_store()
    
    def initialize_store(self):
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        # Check if index already exists
        self.index_exists = self._check_index_exists()

        if not self.recreate_index and self.index_exists:
            print(f"Using existing index at {self.persist_directory}")
            print(f"   Set recreate_index=True to force recreation")
        else:
            if self.index_exists and self.recreate_index:
                print(f"Recreating existing index at {self.persist_directory}")
            else:
                print(f"Creating new index at {self.persist_directory}")

        # Only initialize splitters if we have html_articles to process
        if self.html_articles is not None:
            headers_to_split_on = [
                    ("h1", "Title"),           # Article titles (PCD, MMWR)
                    ("h2", "Major Section"),   # Main sections in PCD (Abstract, Methods, Results, Discussion)
                    ("h3", "Section"),         # Main sections in EID/MMWR, subsections in PCD
                    ("h4", "Subsection"),      # Author info, minor sections
                    ("h5", "Detail")           # Detailed subsections
                        ]
            self.text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=500,
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])

            # Create HTMLSectionSplitter
            self.html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            show_progress=False)


        # Initialize ChromaDB vector store
        self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )

    def chunking(self):
        all_chunked_documents: List[Document] = []

        # If no html_articles provided, return empty list (index already exists)
        if self.html_articles is None:
            return all_chunked_documents

        total_articles = len(self.html_articles)

        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_articles, desc="Chunking documents", unit="article")
        except ImportError:
            progress_bar = None
            print("   Processing articles (this may take several minutes)...")

        try:
            for relative_path, content in self.html_articles.items():
                # First pass: Split by HTML sections
                html_sections = self.html_splitter.split_text(content['html_content'])

                # Second pass: Apply RecursiveCharacterTextSplitter for size constraints
                size_constrained_chunks = self.text_splitter.split_documents(html_sections)
                for i, chunk in enumerate(size_constrained_chunks):
                    chunked_document = Document(
                                    page_content=chunk.page_content,
                                    metadata={"source_path": relative_path,
                                              "title":content['metadata']['title'],
                                              "collection":'pcd'}
                                )
                    all_chunked_documents.append(chunked_document)

                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)

        finally:
            if progress_bar:
                progress_bar.close()

        return all_chunked_documents
    
    def index_document(self,all_chunked_documents,batch_size=50):

        total_docs = len(all_chunked_documents)
        processed_count = 0
            
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_docs, desc="Creating embeddings", unit="doc")
        except ImportError:
            progress_bar = None
            print("   Processing in batches (this may take several minutes)...")
            
        try:
            # Process documents in batches
            for i in range(0, len(all_chunked_documents), batch_size):
                batch = all_chunked_documents[i:i + batch_size]
                
                # Add remaining batches to existing vectorstore
                if self.vectorstore is not None:
                    self.vectorstore.add_documents(batch)
                
                processed_count += len(batch)
                
                # Update progress bar
                if progress_bar:
                    progress_bar.update(len(batch))
                else:
                    print(f"   Processed {processed_count}/{total_docs} documents")

        finally:
            if progress_bar:
                progress_bar.close()

    def semantic_search(self,
            query: str,
            k: int = 5,
            search_type: str = "mmr"
        ) -> List[Dict[str, Any]]:
            """
            Perform semantic search on indexed documents.
            
            Args:
                query: Search query string
                k: Number of results to return
                collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')
                search_type: Type of search ('similarity', 'mmr')
                
            Returns:
                List of search results with content and metadata
            """

            
            # Update retriever with search parameters
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k":k}
            )
            
            # Perform search
            docs = retriever.invoke(query)
            
            # Format results
            results: List[Dict[str, Any]] = []
            for doc in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "title": doc.metadata.get("title", "Unknown"),
                    "collection": doc.metadata.get("collection", "unknown"),
                    "url": doc.metadata.get("url", ""),
                    "relevance_score": getattr(doc, 'relevance_score', None)
                }
                results.append(result)
            
            return results

    def _check_index_exists(self) -> bool:
        """Check if a ChromaDB index already exists in the persist directory."""
        persist_path = Path(self.persist_directory)
        
        # Check for ChromaDB files
        chroma_files = [
            persist_path / "chroma.sqlite3",
        ]
        
        # Check if any essential ChromaDB files exist
        has_db_files = any(file.exists() for file in chroma_files)
        
        # Also check for collection directories (ChromaDB creates UUID-named folders)
        has_collections = False
        if persist_path.exists():
            for item in persist_path.iterdir():
                if item.is_dir() and len(item.name) == 36:  # UUID length
                    has_collections = True
                    break
        
        return has_db_files and has_collections

    def should_process_documents(self) -> bool:
        """Determine if documents should be processed (chunked and indexed)."""
        return self.recreate_index or not self.index_exists

    def get_document_count(self) -> int:
        """Get the number of documents in the existing index."""
        if self.vectorstore and self.index_exists:
            try:
                # Try to get collection info
                collection = self.vectorstore._collection
                return collection.count()
            except Exception:
                return 0
        return 0

