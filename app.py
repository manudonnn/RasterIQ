import streamlit as st
from dotenv import load_dotenv
import plotly.io as pio
import time

load_dotenv()

# Must import after dotenv so orchestrator gets the env vars
from src.agent.orchestrator import RosterIQAgent

st.set_page_config(page_title="RosterIQ Agent", layout="wide")

@st.cache_resource
def get_agent():
    return RosterIQAgent()

agent = get_agent()

# Sidebar for Memory Inspection
with st.sidebar:
    st.title(" Memory State (For Judges)")
    
    st.subheader("Procedural Memory")
    with st.expander("Available Workflows"):
        st.code(agent.procedural.list_procedures())
        
    st.subheader("Semantic Memory")
    with st.expander("Domain Knowledge Index"):
        st.write(f"{len(agent.semantic.entries)} concepts loaded in FAISS.")
        
    st.subheader("Episodic Memory")
    with st.expander("Recent History"):
        history = agent.episodic.get_recent_history(5)
        for idx, h in enumerate(history):
            st.markdown(f"**Q{idx+1}:** {h['query']}")
            st.caption(f"*A:* {h['response'][:100]}...")

# Main Chat Interface
st.title("RosterIQ Agent")
st.markdown("Ask me to analyze the roster pipeline, check market health, or diagnose stuck records!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am RosterIQ. How can I help you analyze the roster operations pipeline today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What has this agent investigated before?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate agent response
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            # Clear old charts/reports
            agent.viz_tool.generated_charts.clear()
            agent.report_tool.generated_reports.clear()
            
            response = agent.run(prompt)
            st.markdown(response)
            
            # Display generated charts, if any
            for chart_json in agent.viz_tool.generated_charts:
                fig = pio.from_json(chart_json)
                st.plotly_chart(fig, use_container_width=True)
                
            # Display generated reports, if any
            for html_report in agent.report_tool.generated_reports:
                st.components.v1.html(html_report, height=400, scrolling=True)
                
    st.session_state.messages.append({"role": "assistant", "content": response})
