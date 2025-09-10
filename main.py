import asyncio
import pandas as pd
from sodapy import Socrata
import pathlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_huggingface import HuggingFaceEmbeddings as lgHuggingFaceEmbeddings
from rag_agent import AgentConfig, AgenticRAG
from loader import download_file, get_data_directory, extract_zip_files, load_html_files
from vectorstore import VectorStorePaper

# =============================================================================
# Configuration Constants
# =============================================================================
_URL_PCD = "https://data.cdc.gov/api/views/ut5n-bmc3/files/c0594869-ba74-4c26-bf54-b2dab3dff971?download=true&filename=pcd_2004-2023.zip"
HTML_ZIP_DIRECTORY = "./cdc-corpus-data/zip"
RECREATE_INDEX = False  # Change to True to force index recreation
CHROMA_PERSIST_DIRECTORY = "./cdc-corpus-data/chroma_db"

# =============================================================================
# Data Setup Functions
# =============================================================================

def download() -> None:
    """Download CDC PCD data if not already present.
    
    Downloads the CDC Preventing Chronic Disease (2004-2023) dataset
    from the official CDC data portal if the data directory doesn't exist.
    """
    if not Path(HTML_ZIP_DIRECTORY).exists():
        print("No data found. Downloading CDC PCD dataset...")
        download_file(url=_URL_PCD, file_name="pcd.zip")

def extract_and_load_data() -> Dict[str, str]:
    """Extract ZIP files and load HTML articles.
    
    Extracts downloaded ZIP files and loads all HTML articles
    into a dictionary for processing.
    
    Returns:
        Dict[str, str]: Dictionary mapping file paths to HTML content
    
    Raises:
        FileNotFoundError: If target directory doesn't exist after extraction
    """
    data_dir = get_data_directory()
    extract_zip_files()
    target_dir = data_dir / "html-outputs/pcd"

    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory {target_dir} not found after extraction")
    
    data_html = load_html_files()
    return data_html

# =============================================================================
# Vector Store Setup Functions
# =============================================================================

def create_index(data_html: Dict[str, str]) -> VectorStorePaper:
    """Create or load the vector store index.
    
    Initializes a VectorStorePaper instance with the provided HTML articles
    and displays the current index status.
    
    Args:
        data_html: Dictionary mapping file paths to HTML content
        
    Returns:
        VectorStorePaper: Initialized vector store instance
    """
    vector_store = VectorStorePaper(
        html_articles=data_html,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        recreate_index=RECREATE_INDEX
    )

    # Display index status
    if vector_store.index_exists:
        doc_count = vector_store.get_document_count()
        print(f"Index contains {doc_count} document chunks")
    else:
        print("No existing index found")

    return vector_store

def chunking(vector_store: VectorStorePaper) -> List[Any]:
    """Process documents into chunks if needed.
    
    Chunks documents only if the index needs to be recreated,
    otherwise skips processing to use existing index.
    
    Args:
        vector_store: Initialized vector store instance
        
    Returns:
        List[Any]: List of chunked documents, empty if using existing index
    """
    if vector_store.should_process_documents():
        print("Chunking documents...")
        documents = vector_store.chunking()
        print(f"Created {len(documents)} document chunks")
    else:
        print("Skipping document chunking (using existing index)")
        documents = []

    return documents

def indexing(vector_store: VectorStorePaper, documents: List[Any]) -> None:
    """Index documents in the vector store if needed.
    
    Creates vector embeddings and indexes documents only if the index
    needs to be recreated, otherwise skips to use existing index.
    
    Args:
        vector_store: Initialized vector store instance
        documents: List of chunked documents to index
    """
    if vector_store.should_process_documents():
        print(f"Indexing {len(documents)} documents (this may take several minutes)...")
        vector_store.index_document(documents)
        print("Indexing completed!")
    else:
        print("Skipping document indexing (using existing index)")
        print("Ready to perform searches!")


# =============================================================================
# Agent Setup Functions
# =============================================================================

def create_agent(vector_store: VectorStorePaper) -> Tuple[AgenticRAG, AgentConfig]:
    """Create and configure the agentic RAG system.
    
    Initializes the AgenticRAG agent with predefined configuration
    optimized for CDC PCD research queries.
    
    Args:
        vector_store: Initialized vector store instance
        
    Returns:
        Tuple[AgenticRAG, AgentConfig]: Configured agent and its configuration
    """
    config = AgentConfig(
        collection_filter='pcd',
        relevance_cutoff=8,
        search_k=10,
        max_evidence_pieces=5,
        max_search_attempts=3
    )
    
    agentic_rag = AgenticRAG(vector_store=vector_store, config=config)
    return agentic_rag, config

async def ask(config: AgentConfig, agent: AgenticRAG, question: str) -> str:
    """Ask a research question using the agentic RAG system.
    
    Processes a research question through the agent's search-gather-response
    workflow to provide evidence-based answers.
    
    Args:
        config: Agent configuration
        agent: Initialized AgenticRAG instance
        question: Research question to answer
        
    Returns:
        str: Generated answer based on evidence from the literature
    """
    print(f"Question: {question}\n")
    answer = await agent.ask_question(question, max_turns=10)
    return answer


# =============================================================================
# Main Execution
# =============================================================================

async def main() -> None:
    """Main execution function for the PaperQA demonstration.
    
    Orchestrates the complete workflow:
    1. Download CDC data if needed
    2. Extract and load HTML articles
    3. Create or load vector store index
    4. Process documents (chunking and indexing)
    5. Initialize the agentic RAG system
    6. Answer a sample research question
    """
    print("=" * 60)
    print("OpenAI Agent PaperQA - CDC Public Health Research System")
    print("=" * 60)

    # Download data if needed
    try:
        download()
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    # Data extraction and loading phase
    try:
        data_html = extract_and_load_data()
        print(f"Successfully loaded {len(data_html)} HTML articles")
    except FileNotFoundError as e:
        print(f"Data loading failed: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during data loading: {e}")
        return
    
    # Vector store setup and indexing phase
    try:
        vector_store = create_index(data_html=data_html)
        chunked_docs = chunking(vector_store)
        indexing(vector_store, chunked_docs)
    except Exception as e:
        print(f"Error in vector store setup or indexing: {e}")
        return
    
    # Agent creation phase
    try:
        agentic_rag, config = create_agent(vector_store)
    except Exception as e:
        print(f"Error creating agent: {e}")
        return


    # Question answering phase
    question = "What are the most common methods used in diabetes prevention to support adolescents in rural areas in the US?"
    
    try:
        print("\n" + "=" * 60)
        print("QUESTION ANSWERING PHASE")
        print("=" * 60)
        answer = await ask(config=config, agent=agentic_rag, question=question)
        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(f"{answer}")
        print("=" * 60)
    except Exception as e:
        print(f"Error during question answering: {e}")
        return
    
    print("\nPaperQA execution completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())




