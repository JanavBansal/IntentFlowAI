"""
IntentFlow V4.5 Council of Experts - Agent Module

This module contains all 5 specialized agents that form the Council:
1. Technical Analyst - Price patterns, momentum (LightGBM)
2. Flow Detective - Institutional activity from delivery data (XGBoost)
3. Regime Sentinel - Market state detection (HMM)
4. Risk Contrarian - Anomaly detection with veto power (Isolation Forest)
5. Earnings Oracle - Fundamental screening (Logistic Regression)

Plus:
- Debate Protocol - LLM-based synthesis of agent outputs
- Council Workflow - LangGraph orchestration of agents
"""

from intentflow_ai.agents.base_agent import BaseAgent, AgentOutput
from intentflow_ai.agents.technical_analyst import TechnicalAnalystAgent
from intentflow_ai.agents.flow_detective import FlowDetectiveAgent
from intentflow_ai.agents.regime_sentinel import RegimeSentinelAgent
from intentflow_ai.agents.risk_contrarian import RiskContrarianAgent
from intentflow_ai.agents.earnings_oracle import EarningsOracleAgent
from intentflow_ai.agents.debate_protocol import DebateProtocol
from intentflow_ai.agents.council_workflow import CouncilOfExperts

__all__ = [
    "BaseAgent",
    "AgentOutput",
    "TechnicalAnalystAgent",
    "FlowDetectiveAgent",
    "RegimeSentinelAgent",
    "RiskContrarianAgent",
    "EarningsOracleAgent",
    "DebateProtocol",
    "CouncilOfExperts",
]
