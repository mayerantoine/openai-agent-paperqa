from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from typing import TypedDict
import asyncio
import os
from agents import Agent, Runner, RunContextWrapper, function_tool
from pydantic import BaseModel, Field
from rich.console import Console
from vectorstore import VectorStorePaper


class SessionStatus(TypedDict):
    """Status tracking for agent session."""
    Paper: int
    Relevant: int
    Evidence: int


@dataclass
class SessionState:
    """Session state management for agentic RAG workflow."""
    original_question: str
    updated_question: str
    search_results: Optional[List] = field(default_factory=list)
    evidence_library: List[Tuple] = field(default_factory=list)
    status: Optional[Dict[str, int]] = field(default_factory=lambda: {'Paper': 0, 'Relevant': 0, 'Evidence': 0})


class EvidenceSummary(BaseModel):
    """Structured output for evidence analysis."""
    relevant_information_summary: str = Field(description="Summary of the evidence or 'Not applicable'")
    score: int = Field(description="A score from 1-10 indicating relevance to question")

class AgentConfig:
    """Configuration for agentic RAG system."""
    
    def __init__(
        self,
        collection_filter: str = 'pcd',
        relevance_cutoff: int = 8,
        search_k: int = 10,
        max_evidence_pieces: int = 5,
        max_search_attempts: int = 3,
        model_name: Optional[str] = None
    ):
        self.collection_filter = collection_filter
        self.relevance_cutoff = relevance_cutoff
        self.search_k = search_k
        self.max_evidence_pieces = max_evidence_pieces
        self.max_search_attempts = max_search_attempts
        
        # Use same environment variables as RAGEngine for consistency
        if model_name is None:
            # Get provider and model from environment (same as RAGEngine)
            provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
            if provider == "openai":
                self.model_name = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
            elif provider == "anthropic":
                self.model_name = os.getenv("DEFAULT_LLM_MODEL", "claude-3-5-sonnet")
            else:
                # Fallback to environment variable or default
                self.model_name = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        else:
            self.model_name = model_name


class AgentToolFactory:
    """Factory class for creating agent tools."""
    
    def __init__(self,config: AgentConfig, vector_store: VectorStorePaper):
        
        self.config = config
        self._evidence_agent = None
        self.vectorstore = vector_store
    
    @property
    def evidence_agent(self):
        """Lazy initialization of evidence agent."""
        if self._evidence_agent is None:
            instructions = (
                "You are a helpful research librarian assistant. Your role is to summarize chunk of evidence from literature. "
                "Summarize the text below to help answer a question. Do not directly answer the question, "
                "instead summarize to give evidence to help answer the question. Reply 'Not applicable' if text is irrelevant. "
                "Use 2-3 sentences. At the end of your response, provide a score from 1-10 on a newline indicating relevance to question. "
                "Do not explain your score."
            )
            
            self._evidence_agent = Agent(
                name="EvidenceAgent",
                instructions=instructions,
                model=self.config.model_name,
                output_type=EvidenceSummary
            )
        return self._evidence_agent
    
    def create_search_tool(self):
        """Create the search tool for finding relevant papers."""
        
        @function_tool
        async def search(state: "RunContextWrapper[SessionState]", question: str) -> str:
            """Use this tool to search for papers content to help answer the question."""
            
            # Update session state
            if state.context.original_question == "":
                state.context.original_question = question
            else:
                state.context.updated_question = question

            print(f"🟢 [Search] Starting paper search for question:{question}")
            
            # Check if vector index exists, if not exit and ask to create it
            #print(self.vectorstore.vectorstore._collection.count())
            
            # Perform semantic search
            results = self.vectorstore.semantic_search(
                query=question,
                k=self.config.search_k,
            )

            count_results = len(results)
            state.context.search_results.extend(results)
            state.context.status['Paper'] += count_results

            print(f"🟢 [Search] Paper search returned {count_results} passages from papers")
            self._print_status(state.context.status)
            
            return f"🟢 [Search] Found {count_results} text passages from the papers that semantically matches and can help answer the question."
        
        return search
    
    def create_evidence_tool(self):
        """Create the evidence gathering tool."""
        
        async def evidence_summary(evidence: str, question: str) -> EvidenceSummary:
            """Use the evidence agent to gather relevance information about search results."""
            user_instructions = (
                f"Summarize the text below to help answer a question. "
                f"### Evidence: {evidence} #### "
                f"#### Question: {question} #### "
                f"Relevant Information Summary: "
            )
            
            result = await Runner.run(self.evidence_agent, input=user_instructions)
            return result.final_output

        @function_tool
        async def gather_evidence(state: "RunContextWrapper[SessionState]", question: str) -> str:
            """Use this tool to gather evidence to help answer the question."""

            print(f"🟢 [Gather] Gathering evidence for question: {question}")
            chunks = state.context.search_results

            # Process evidence in parallel
            tasks = [
                asyncio.create_task(evidence_summary(item['title'] + item['content'], question)) 
                for item in chunks
            ]
            results = await asyncio.gather(*tasks)
            print(f"🟢 [Gather] Finished gathering evidence for question: {question}")

            # Filter high-quality evidence
            top_evidence_context = [
                (result.score, result.relevant_information_summary) 
                for result in results 
                if result.score >= self.config.relevance_cutoff
            ]
            count_top_evidence = len(top_evidence_context)

            # Update session state
            state.context.evidence_library.extend(top_evidence_context)
            state.context.status['Evidence'] = len(state.context.evidence_library)
            state.context.status['Relevant'] = len(state.context.evidence_library)

            best_evidence = "\n".join([evidence[1] for evidence in state.context.evidence_library])
            print(state.context.status)
            
            return f"🟢 [Gather] Found and added {count_top_evidence}pieces of evidence relevant to the question. Best evidences: {best_evidence}."
        
        return gather_evidence
    
    def create_answer_tool(self):
        """Create the answer generation tool."""
        
        def get_answer_instructions(state: RunContextWrapper[SessionState], agent) -> str:
            """Generate dynamic instructions for answer agent."""
            context_evidence = "\n".join([evidence[1] for evidence in state.context.evidence_library])

            instructions = (
                "Write an answer for the question below based on the provided context. "
                "If the context provides insufficient information, reply 'I cannot answer'. "
                "Answer in an unbiased, comprehensive, and scholarly tone. "
                "If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences."
            )
            instructions += f"\n## Context: {context_evidence}"
            instructions += f"\n## Question: {state.context.original_question}"

            return instructions

        answer_agent = Agent[SessionState](
            name="AnswerAgent",
            instructions=get_answer_instructions,
            model=self.config.model_name,
        )

        generate_answer = answer_agent.as_tool(
            tool_name="generate_answer",
            tool_description="Use this tool to generate a proposed answer to the question when you have collected enough evidence"
        )
        
        return generate_answer
    
    def _print_status(self, status: Dict[str, int]) -> None:
        """Print current session status."""
        print(f"🟢 [Status] Paper Count={status.get('Paper')} | Relevant Papers={status.get('Relevant')} Current Evidence={status.get('Evidence')}")



class AgenticRAG:
    """
    Agentic RAG system for CDC Text Corpora research.
    
    This class implements a multi-agent workflow for answering research questions
    by coordinating specialized agents for search, evidence analysis, and synthesis.
    """
    
    def __init__(
        self, 
        vector_store: VectorStorePaper,
        config: Optional[AgentConfig] = None,
  
    ):
        """
        Initialize the agentic RAG system.
        
        Args:
            corpus: CDCCorpus instance. If None, will create one.
            config: AgentConfig instance. If None, will use defaults.
        """
        self.config = config or AgentConfig()
        self.vectorstore = vector_store
        
        # Initialize tool factory
        self.tool_factory = AgentToolFactory(config=self.config,vector_store=self.vectorstore )
        
        # Create orchestrator agent
        self._orchestrator_agent = None
    
    def _get_collection_instructions(self) -> str:
        """Get collection-specific instructions."""
        collection_map = {
            'pcd': 'Preventing Chronic Disease',
            'eid': 'Emerging Infectious Diseases', 
            'mmwr': 'Morbidity and Mortality Weekly Report',
            'all': 'CDC'
        }
        
        collection_name = collection_map.get('pcd')
        
        return f"""
            You are a senior researcher AI assistant of the {collection_name} journal. Your role is to answer questions based on evidence in the journal papers. 
            Answer in a direct and concise tone. Your audience is an expert, so be highly specific. If there are ambiguous terms or acronyms, first define them.
            You have access to three tools: search, gather_evidence and generate_answer.
            Search for papers, gather evidence, and answer. If you do not have enough evidence,
            you can search for more papers (preferred) or gather more evidence with a different phrase. 
            You may rephrase or break-up the question in those steps. Once you have {self.config.max_evidence_pieces} or more pieces of evidence from multiple sources, 
            or you have tried more than {self.config.max_search_attempts} times, call generate_answer tool. You may reject the answer and try again if it is incomplete.
            Important: remember to answer if you cannot find enough evidence.
            """
    
    @property
    def orchestrator_agent(self):
        """Lazy initialization of orchestrator agent."""
        if self._orchestrator_agent is None:
            # Get all tools
            search_tool = self.tool_factory.create_search_tool()
            evidence_tool = self.tool_factory.create_evidence_tool()
            answer_tool = self.tool_factory.create_answer_tool()
            
            collection_name = self.config.collection_filter.upper() if self.config.collection_filter != 'all' else 'CDC'
            
            self._orchestrator_agent = Agent[SessionState](
                name=f"{collection_name}Agent",
                instructions=self._get_collection_instructions(),
                model=self.config.model_name,
                tools=[search_tool, evidence_tool, answer_tool]
            )
        
        return self._orchestrator_agent
    
    def create_session_state(self) -> SessionState:
        """Create a new session state."""
        return SessionState(
            original_question="",
            updated_question="",
            search_results=[],
            evidence_library=[],
            status={'Paper': 0, 'Relevant': 0, 'Evidence': 0}
        )
    
    async def ask_question(self, question: str, max_turns: int = 10) -> str:
        """
        Ask a research question using the agentic workflow.
        
        Args:
            question: The research question to answer
            max_turns: Maximum number of agent decision cycles
            
        Returns:
            The generated answer
        """
        # Initialize session state
        session_state = self.create_session_state()
        
        # Run the agentic workflow
        result = await Runner.run(
            self.orchestrator_agent,
            input=question,
            context=session_state,
            max_turns=max_turns
        )
        
        return result.final_output
    
    def get_session_status(self, session_state: SessionState) -> Dict[str, int]:
        """Get current session status."""
        return session_state.status.copy()
    

