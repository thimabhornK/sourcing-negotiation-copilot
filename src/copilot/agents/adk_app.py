"""The same agent team defined with the Agent Development Kit (ADK).

The orchestrator in this package is plain Python so it can run and be tested
with no model. This module declares the equivalent ADK agents over the same
tools, which is the form that deploys to Vertex AI Agent Engine.

  pip install -e ".[agents]"
  export GOOGLE_API_KEY=...
  adk web adk

Status: the agents and tool schemas are constructed and checked in the tests.
They have not yet been run against a live model.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent

from copilot import tools
from copilot.negotiation.playbook import build_playbook

MODEL = os.environ.get("COPILOT_GEMINI_MODEL", "gemini-2.5-flash")
GROUNDING = ("Use only figures returned by your tools. Never calculate or estimate a number "
             "yourself. Quote contract citations exactly as the tools return them.")

spend_analyst = Agent(
    name="spend_analyst", model=MODEL,
    description="Answers questions about spend, vendor price levels and delivery performance.",
    instruction="Answer questions about purchasing history with your tools. " + GROUNDING,
    tools=[tools.top_vendors_by_spend, tools.most_expensive_vendors, tools.late_vendors,
           tools.spend_by_category, tools.vendor_profile])

contract_analyst = Agent(
    name="contract_analyst", model=MODEL,
    description="Answers questions from vendor contracts: payment, rebates, penalties, "
                "price adjustment and termination.",
    instruction="Find the clause that answers the question and cite it. " + GROUNDING,
    tools=[tools.search_contracts, tools.contract_terms, tools.rebate_status])

market_analyst = Agent(
    name="market_analyst", model=MODEL,
    description="Checks whether a quoted price is fair against the should-cost model and "
                "compares the quotes in a sourcing event.",
    instruction="Check quotes with your tools and report the verdict, the should-cost range "
                "and the market trend. " + GROUNDING,
    tools=[tools.quote_check, tools.rfq_summary])

strategist = Agent(
    name="strategist", model=MODEL,
    description="Builds a negotiation playbook for a quote: opening ask, target, walk-away "
                "price, arguments and trades.",
    instruction="Call build_playbook and present its result. Do not change the prices it "
                "returns. " + GROUNDING,
    tools=[build_playbook])

root_agent = Agent(
    name="sourcing_copilot", model=MODEL,
    description="Supervisor for the sourcing and negotiation copilot.",
    instruction="You help a buyer. Hand each request to the specialist that fits: "
                "spend_analyst for spend and vendor performance, contract_analyst for contract "
                "terms and rebates, market_analyst for price checks, strategist for "
                "negotiation plans. Ask for the material, quantity, price and vendor if a "
                "quote is incomplete. " + GROUNDING,
    sub_agents=[spend_analyst, contract_analyst, market_analyst, strategist])
