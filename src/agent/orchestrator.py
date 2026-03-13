import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from src.memory.episodic import EpisodicMemory
from src.memory.procedural import ProceduralMemory
from src.memory.semantic import SemanticMemory
from src.tools.data_query import DuckDBTool
from src.tools.visualization import VisualizationTool
from src.tools.web_search import WebSearchTool
from src.tools.report_generator import ReportGeneratorTool

class RosterIQAgent:
    def __init__(self):
        # 1. Initialize Memory
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()
        self.semantic = SemanticMemory()
        
        # 2. Initialize Tools
        self.db_tool = DuckDBTool()
        self.viz_tool = VisualizationTool()
        self.web_tool = WebSearchTool()
        self.report_tool = ReportGeneratorTool()
        
        # 3. Setup LLM via OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY", "dummy_key")
        model_name = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-large") # Fallback if empty, but you can change in .env or hardcode
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            temperature=0
        )
        
        # 4. Bind LangChain Tools
        self.tools = [
            Tool(
                name="RunSQL",
                func=self.db_tool.run_sql,
                description="Input a SQL SELECT query string to run against duckdb. Tables: roster_processing_details, aggregated_operational_metrics."
            ),
            Tool(
                name="WebSearch",
                func=self.web_tool.search,
                description="Input a search query string to look up regulatory changes, provider info, or Medicaid/CMS rules."
            ),
            Tool(
                name="GenerateChart",
                func=lambda params: self.viz_tool.generate_chart(**eval(params)) if params.startswith("{") else "Error: Use dict format e.g. {'chart_type': 'bar', 'data_json': '[...]', 'title': 'T', 'x_axis': 'X', 'y_axis': 'Y'}",
                description="Input a python dict string to generate a chart. Keys: chart_type, data_json, title, x_axis, y_axis, color_axis."
            ),
            Tool(
                name="GenerateReport",
                func=lambda params: self.report_tool.generate_report(**eval(params)) if params.startswith("{") else "Error: Use dict format e.g. {'target': 'KS', 'flags_json': '[]', 'summary_text': 'Text'}",
                description="Input a python dict string to generate an HTML report. Keys: target, flags_json, summary_text."
            ),
            Tool(
                name="SemanticKnowledge",
                func=self.semantic.retrieve_concept,
                description="Lookup definitions in the domain knowledge base (e.g. 'IS_STUCK', 'DART_GENERATION')."
            ),
            Tool(
                name="GetProcedureDetails",
                func=self.procedural.get_procedure,
                description="Get instructions and SQL query for a procedural workflow."
            )
        ]
        
        # 5. Build LangGraph React Agent
        self.agent = create_react_agent(self.llm, tools=self.tools)

    def run(self, user_query: str) -> str:
        """Main execution entry point for the agent."""
        # 1. Retrieve episodic context & procedures
        episodic_ctx = self.episodic.retrieve_past_context(user_query)
        procs = self.procedural.list_procedures()
        
        # 2. Construct dynamic prompt
        full_query = f"""
You are RosterIQ, an autonomous AI agent for healthcare payer roster operations. Use the provided tools to analyze the data.

AVAILABLE PROCEDURES:
{procs}

RELEVANT EPISODIC CONTEXT (Past Interactions):
{episodic_ctx}

User Question: {user_query}
"""
        # 3. Execute Agent
        try:
            response = self.agent.invoke({"messages": [HumanMessage(content=full_query)]})
            final_answer = response["messages"][-1].content
        except Exception as e:
            final_answer = f"Agent execution failed: {str(e)}"
        
        # 4. Log to episodic memory
        self.episodic.log_interaction(
            query=user_query, 
            response=final_answer, 
            state_snapshot={"stuck_ros": 0} # Dummy state for now
        )
        
        return final_answer
