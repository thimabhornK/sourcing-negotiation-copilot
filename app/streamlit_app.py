"""Browser chat for the copilot.

  pip install -e ".[app]"
  streamlit run app/streamlit_app.py
"""

import streamlit as st

from copilot.agents.orchestrator import Session

EXAMPLES = {
    "Ask": ["Top 5 vendors by spend in 2025",
            "Which vendors charge the most above should-cost for steel sheet?",
            "What are the payment terms with V-0003?",
            "Are we close to a rebate with V-0002 in 2025?"],
    "Check a quote": ["Is 1.62 USD fair for 5000 kg of MAT-0001 from V-0002 on 2025-06-15?",
                      "Compare the quotes in RFQ-01990"],
    "Negotiate": ["Build a negotiation playbook for that quote",
                  "Run the negotiation automatically",
                  "Let me practise against the vendor"],
}

st.set_page_config(page_title="Sourcing & Negotiation Copilot", layout="wide")
if "session" not in st.session_state:
    st.session_state.session = Session()
    st.session_state.turns = []
session = st.session_state.session

with st.sidebar:
    st.header("Try")
    picked = None
    for mode, prompts in EXAMPLES.items():
        st.caption(mode)
        for prompt in prompts:
            if st.button(prompt, use_container_width=True):
                picked = prompt
    st.divider()
    st.caption(f"Model: {session.llm.name}")
    if st.button("New conversation"):
        st.session_state.clear()
        st.rerun()

st.title("Sourcing & Negotiation Copilot")
st.caption("Synthetic data. Every figure comes from a tool result, and a reviewer checks "
           "the numbers before an answer is shown.")

message = st.chat_input("Ask about spend, check a quote, or plan a negotiation") or picked
if message:
    st.session_state.turns.append((message, session.handle(message)))

for question, reply in st.session_state.turns:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        st.caption(reply.mode)
        st.code(reply.text, language=None, wrap_lines=True)
        with st.expander(f"How this was answered: {reply.usage['steps']} steps, "
                         f"{reply.usage['ms']:.0f} ms"):
            st.dataframe([{"agent": s["agent"], "step": s["step"], "ms": s["ms"],
                           "tokens": s["tokens_in"] + s["tokens_out"]} for s in reply.trace],
                         hide_index=True)
            if reply.review.get("attempts"):
                st.write("Reviewer: all numbers traced to tool results"
                         if reply.review["ok"] else
                         f"Reviewer rejected: {reply.review['unsupported_numbers']}")
