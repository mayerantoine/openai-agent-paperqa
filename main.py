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
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# =============================================================================
# Configuration Constants
# =============================================================================
_URL_PCD = "https://data.cdc.gov/api/views/ut5n-bmc3/files/c0594869-ba74-4c26-bf54-b2dab3dff971?download=true&filename=pcd_2004-2023.zip"
HTML_ZIP_DIRECTORY = "./cdc-corpus-data/zip"
RECREATE_INDEX = False  # Change to True to force index recreation
CHROMA_PERSIST_DIRECTORY = "./cdc-corpus-data/chroma_db"

# =============================================================================
# Index Check Functions
# =============================================================================

def check_index_exists() -> bool:
    """Check if ChromaDB index exists without loading data.

    Returns:
        bool: True if a valid ChromaDB index exists, False otherwise
    """
    persist_path = Path(CHROMA_PERSIST_DIRECTORY)

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

def create_index(data_html: Optional[Dict[str, str]] = None) -> VectorStorePaper:
    """Create or load the vector store index.

    Initializes a VectorStorePaper instance with the provided HTML articles
    and displays the current index status.

    Args:
        data_html: Dictionary mapping file paths to HTML content.
                   Can be None when using existing index.

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

async def ask(agent: AgenticRAG, question: str) -> Tuple[str, Any]:
    """Ask a research question using the agentic RAG system.

    Processes a research question through the agent's search-gather-response
    workflow to provide evidence-based answers.

    Args:
        agent: Initialized AgenticRAG instance
        question: Research question to answer

    Returns:
        Tuple[str, Any]: Generated answer and session context with evidence
    """
    answer, context = await agent.ask_question(question, max_turns=10)
    return answer, context


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
    6. Interactive Q&A loop with user
    """
    console = Console()

    console.print("\n" + "=" * 60, style="bold cyan")
    console.print("OpenAI Agent PaperQA - Public Health Research System", style="bold cyan")
    console.print("=" * 60 + "\n", style="bold cyan")

    # Early check: if index exists and we're not recreating, skip data loading
    index_exists = check_index_exists()
    skip_data_loading = index_exists and not RECREATE_INDEX

    if skip_data_loading:
        console.print("[green]✓ Found existing index, skipping data download/load[/green]")
        console.print("[dim]   (Set RECREATE_INDEX=True to rebuild from source data)[/dim]\n")
        data_html = None
    else:
        # Download data if needed
        try:
            download()
        except Exception as e:
            console.print(f"[red]Error downloading data: {e}[/red]")
            return

        # Data extraction and loading phase
        try:
            data_html = extract_and_load_data()
            console.print(f"Successfully loaded {len(data_html)} HTML articles", style="green")
        except FileNotFoundError as e:
            console.print(f"[red]Data loading failed: {e}[/red]")
            return
        except Exception as e:
            console.print(f"[red]Unexpected error during data loading: {e}[/red]")
            return

    # Vector store setup and indexing phase
    try:
        vector_store = create_index(data_html=data_html)
        chunked_docs = chunking(vector_store)
        indexing(vector_store, chunked_docs)
    except Exception as e:
        console.print(f"[red]Error in vector store setup or indexing: {e}[/red]")
        return

    # Agent creation phase
    try:
        agentic_rag, config = create_agent(vector_store)
    except Exception as e:
        console.print(f"[red]Error creating agent: {e}[/red]")
        return

    # Interactive Q&A loop
    console.print("\n" + "=" * 60, style="bold green")
    console.print("System Ready - Interactive Q&A Mode", style="bold green")
    console.print("=" * 60, style="bold green")

    # Display welcome message with instructions
    welcome_panel = Panel(
        "[yellow]Welcome to the QA Research Assistant!\n\n"
        "Ask questions about public health research (2004-2023).\n\n"
        "Commands:[/yellow]\n"
        "  [dim]• Type your question and press Enter[/dim]\n"
        "  [dim]• Type 'exit', 'quit', or 'q' to exit[/dim]\n"
        "  [dim]• Press Ctrl+C to exit[/dim]",
        title="[bold cyan]Instructions[/bold cyan]",
        border_style="cyan"
    )
    console.print(welcome_panel)
    console.print()

    try:
        while True:
            # Get user question with colored prompt
            console.print("[bold cyan]Your question:[/bold cyan] ", end="")
            question = input().strip()

            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'q', '']:
                if question == '':
                    continue
                console.print("\n[dim]Thank you. Goodbye![/dim]\n")
                break

            # Process question
            try:
                console.print()
                answer, context = await ask(agent=agentic_rag, question=question)

                # Display answer in formatted panel
                console.print("\n" + "=" * 60, style="bold green")
                console.print("ANSWER", style="bold green")
                console.print("=" * 60, style="bold green")
                console.print(f"[green]{answer}[/green]")
                console.print("=" * 60 + "\n", style="bold green")

                # Display sources if available
                if context.evidence_library:
                    console.print("=" * 60, style="bold blue")
                    console.print("SOURCES", style="bold blue")
                    console.print("=" * 60, style="bold blue")

                    # Show top 3 sources
                    for idx, item in enumerate(context.evidence_library[:3], 1):
                        score, summary, title, content = item
                        console.print(f"\n[bold blue]{idx}. {title}[/bold blue]")
                        console.print(f"[dim]   Relevance Score: {score}/10[/dim]")

                        # Truncate summary to reasonable length
                        truncated_summary = summary[:200] + "..." if len(summary) > 200 else summary
                        console.print(f"[blue]   {truncated_summary}[/blue]")

                    console.print("\n" + "=" * 60 + "\n", style="bold blue")


            except Exception as e:
                console.print(f"\n[red]Error processing question: {e}[/red]\n")

    except KeyboardInterrupt:
        console.print("\n\n[dim]Interrupted. Thank you. Goodbye![/dim]\n")
    except EOFError:
        console.print("\n\n[dim]Thank you. Goodbye![/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())




