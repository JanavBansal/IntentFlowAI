"""
Debate Protocol for Council of Experts.

Implements structured Bull vs Bear debate with LLM synthesis.
Pattern inspired by TradingAgents paper.

The debate:
1. Separates agents into bull (signal > 0) and bear (signal < 0) camps
2. Builds thesis from each camp's reasoning
3. Uses LLM (GPT-4o-mini) to synthesize a final recommendation
4. Applies weighted voting with risk agent veto check
"""

from typing import Dict, List, Optional, Any
import os

from intentflow_ai.agents.base_agent import AgentOutput
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class DebateProtocol:
    """
    Implements Bull vs Bear structured debate for agent synthesis.
    
    Features:
    - Separates agents by signal direction
    - Builds structured arguments
    - LLM synthesis for human-readable explanation
    - Weighted voting for final signal
    - Risk agent veto check
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        use_llm: bool = True,
        temperature: float = 0.3
    ):
        """
        Initialize debate protocol.
        
        Args:
            llm_model: Which OpenAI model to use for synthesis
            use_llm: If False, skip LLM and use weighted voting only
            temperature: LLM temperature (lower = more deterministic)
        """
        self.llm_model = llm_model
        self.use_llm = use_llm
        self.temperature = temperature
        
        self._llm = None
        self._llm_available = False
        
        if use_llm:
            self._init_llm()
    
    def _init_llm(self) -> None:
        """Initialize LangChain LLM."""
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set. LLM synthesis disabled.")
                return
            
            self._llm = ChatOpenAI(
                model=self.llm_model,
                temperature=self.temperature,
                api_key=api_key
            )
            self._llm_available = True
            logger.info(f"LLM initialized: {self.llm_model}")
            
        except ImportError:
            logger.warning("langchain-openai not installed. LLM synthesis disabled.")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
    
    def conduct_debate(
        self,
        agent_outputs: List[AgentOutput],
        stock_symbol: str,
        current_price: float,
        regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize agent outputs into final decision.
        
        Args:
            agent_outputs: List of outputs from all agents
            stock_symbol: e.g., "RELIANCE"
            current_price: Current stock price
            regime: Optional current market regime
            
        Returns:
            {
                'bull_thesis': str,
                'bear_thesis': str,
                'synthesis': str,
                'final_signal': float,
                'final_confidence': float,
                'vetoed': bool,
                'veto_reason': str
            }
        """
        # Separate bulls and bears
        bulls = [a for a in agent_outputs if a.signal > 0.05]
        bears = [a for a in agent_outputs if a.signal < -0.05]
        neutrals = [a for a in agent_outputs if -0.05 <= a.signal <= 0.05]
        
        # Build thesis for each side
        bull_thesis = self._build_thesis(bulls, "BULLISH")
        bear_thesis = self._build_thesis(bears, "BEARISH")
        
        # Weighted voting
        final_signal, final_confidence = self._weighted_vote(agent_outputs)
        
        # Check for risk veto
        vetoed, veto_reason = self._check_veto(agent_outputs)
        if vetoed:
            final_signal = 0.0
        
        # LLM synthesis (if available)
        if self._llm_available and self.use_llm:
            synthesis = self._llm_synthesis(
                bull_thesis, bear_thesis, 
                stock_symbol, current_price, 
                regime, final_signal, vetoed
            )
        else:
            synthesis = self._simple_synthesis(
                bulls, bears, neutrals,
                final_signal, vetoed, veto_reason
            )
        
        return {
            'bull_thesis': bull_thesis,
            'bear_thesis': bear_thesis,
            'synthesis': synthesis,
            'final_signal': final_signal,
            'final_confidence': final_confidence,
            'vetoed': vetoed,
            'veto_reason': veto_reason if vetoed else None,
            'agent_breakdown': {
                'bulls': len(bulls),
                'bears': len(bears),
                'neutrals': len(neutrals)
            }
        }
    
    def _build_thesis(self, agents: List[AgentOutput], side: str) -> str:
        """Build argument thesis from agents on one side."""
        if not agents:
            return f"No {side} arguments presented."
        
        thesis = f"**{side} CASE** ({len(agents)} agents):\n"
        
        for agent in sorted(agents, key=lambda a: abs(a.signal), reverse=True):
            confidence_emoji = "🟢" if agent.confidence > 0.7 else "🟡" if agent.confidence > 0.4 else "🔴"
            thesis += f"\n• **{agent.agent_name}** {confidence_emoji}\n"
            thesis += f"  Signal: {agent.signal:+.2f} (confidence: {agent.confidence:.0%})\n"
            thesis += f"  {agent.reasoning}\n"
        
        return thesis
    
    # IC-derived weights from backtest (flow=+0.104, tech=-0.015, earnings=0.0, regime excluded)
    # Regime Sentinel is context-only (adjusts thresholds in finalize), not a voter.
    # Negative-IC agents get a small floor weight for diversification.
    IC_WEIGHTS: Dict[str, float] = {
        "Flow Detective":    0.65,
        "Technical Analyst": 0.20,
        "Earnings Oracle":   0.10,
        "Risk Contrarian":   0.05,  # Veto only; minimal signal weight
        "Regime Sentinel":   0.00,  # Excluded from vote — context only
    }

    def _weighted_vote(self, agent_outputs: List[AgentOutput]) -> tuple:
        """IC-weighted average signal.

        Weights are derived from empirical backtest ICs (Flow IC=+0.104 dominates).
        Regime Sentinel is excluded from the vote (IC=-0.040) and acts as a
        threshold modifier in the finalize node instead.
        """
        if not agent_outputs:
            return 0.0, 0.0

        # Exclude agents with zero IC weight (Regime Sentinel)
        voting_agents = [a for a in agent_outputs
                         if self.IC_WEIGHTS.get(a.agent_name, 0.10) > 0]
        if not voting_agents:
            voting_agents = agent_outputs  # fallback

        # Weight = IC_weight * confidence  (confidence scales within the IC tier)
        weights = [
            self.IC_WEIGHTS.get(a.agent_name, 0.10) * max(a.confidence, 0.01)
            for a in voting_agents
        ]
        total_weight = sum(weights)

        if total_weight == 0:
            return 0.0, 0.0

        weighted_signal = sum(
            a.signal * w for a, w in zip(voting_agents, weights)
        ) / total_weight

        avg_confidence = sum(a.confidence for a in voting_agents) / len(voting_agents)

        return float(weighted_signal), float(avg_confidence)
    
    def _check_veto(self, agent_outputs: List[AgentOutput]) -> tuple:
        """Check if Risk Contrarian vetoes the trade."""
        risk_agent = next(
            (a for a in agent_outputs if a.agent_name == "Risk Contrarian"),
            None
        )
        
        if risk_agent is None:
            return False, ""
        
        # Veto if signal strongly negative and high confidence
        if risk_agent.signal < -0.5 and risk_agent.confidence > 0.7:
            return True, risk_agent.reasoning
        
        return False, ""
    
    def _llm_synthesis(
        self,
        bull_thesis: str,
        bear_thesis: str,
        symbol: str,
        price: float,
        regime: Optional[str],
        final_signal: float,
        vetoed: bool
    ) -> str:
        """Use LLM to synthesize arguments."""
        regime_str = f"Current market regime: {regime}" if regime else ""
        veto_str = "⛔ TRADE VETOED BY RISK AGENT\n\n" if vetoed else ""
        
        prompt = f"""You are a trading desk synthesis agent. Analyze this debate and provide a clear recommendation.

**Stock**: {symbol} (₹{price:,.2f})
{regime_str}

{bull_thesis}

{bear_thesis}

**Weighted Vote Result**: Signal = {final_signal:+.2f}

Provide a 2-3 sentence synthesis:
1. What's the primary driver?
2. What's the key risk?
3. Final recommendation (BUY/SELL/HOLD with conviction)"""

        try:
            response = self._llm.invoke(prompt)
            return veto_str + response.content
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return veto_str + f"Signal: {final_signal:+.2f}. See agent breakdown for details."
    
    def _simple_synthesis(
        self,
        bulls: List[AgentOutput],
        bears: List[AgentOutput],
        neutrals: List[AgentOutput],
        final_signal: float,
        vetoed: bool,
        veto_reason: str
    ) -> str:
        """Simple synthesis without LLM."""
        parts = []
        
        if vetoed:
            parts.append(f"⛔ TRADE VETOED: {veto_reason}")
            return " | ".join(parts)
        
        # Direction
        if final_signal > 0.2:
            direction = "BULLISH"
        elif final_signal < -0.2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        parts.append(f"**{direction}** (signal: {final_signal:+.2f})")
        
        # Vote count
        parts.append(f"Votes: {len(bulls)} bulls, {len(bears)} bears, {len(neutrals)} neutral")
        
        # Top driver
        all_agents = bulls + bears
        if all_agents:
            top_agent = max(all_agents, key=lambda a: abs(a.signal) * a.confidence)
            parts.append(f"Primary driver: {top_agent.agent_name}")
        
        return " | ".join(parts)
