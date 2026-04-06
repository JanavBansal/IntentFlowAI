"""
Council of Experts Workflow - LangGraph State Machine Implementation.

This is the main entry point for V4.5 that replaces V3's monolithic model.

Uses LangGraph StateGraph for:
- State machine checkpointing
- Parallel agent execution
- Audit trail persistence
- Conditional routing with risk veto

Workflow:
1. Regime Detection (affects other agent weights)
2. Parallel Agent Analysis (Technical, Flow, Earnings)
3. Debate Synthesis
4. Risk Assessment (can veto)
5. Final Signal
"""

from typing import Any, Dict, List, Optional, Annotated, TypedDict
from operator import add
import pandas as pd
import os

from intentflow_ai.agents.base_agent import AgentOutput
from intentflow_ai.agents.technical_analyst import TechnicalAnalystAgent
from intentflow_ai.agents.flow_detective import FlowDetectiveAgent
from intentflow_ai.agents.regime_sentinel import RegimeSentinelAgent
from intentflow_ai.agents.risk_contrarian import RiskContrarianAgent
from intentflow_ai.agents.earnings_oracle import EarningsOracleAgent
from intentflow_ai.agents.debate_protocol import DebateProtocol
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class CouncilState(TypedDict):
    """State passed through the LangGraph workflow."""
    # Input
    stock_symbol: str
    features: Any  # pd.DataFrame
    current_price: float
    
    # Separate keys for parallel agents (avoids concurrent write conflict)
    regime: str
    regime_output: Optional[Dict]
    technical_output: Optional[Dict]
    flow_output: Optional[Dict]
    earnings_output: Optional[Dict]
    risk_output: Optional[Dict]
    
    # Final output
    debate_result: Optional[Dict]
    final_signal: float
    final_confidence: float
    vetoed: bool
    veto_reason: Optional[str]
    direction: str
    
    # Audit trail (Annotated with add for accumulation)
    audit_trail: Annotated[List[str], add]


# =============================================================================
# COUNCIL OF EXPERTS - MAIN CLASS
# =============================================================================

class CouncilOfExperts:
    """
    Main orchestration class for V4.5 Council of Experts.
    
    Uses LangGraph StateGraph for workflow orchestration.
    
    Usage:
        council = CouncilOfExperts()
        council.train_all_agents(X_train, y_train)
        result = council.get_signal("RELIANCE", features_df)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Council of Experts."""
        self.config = config or {}
        self._load_env()
        
        # Initialize all 5 agents
        self.agents = {
            'technical': TechnicalAnalystAgent(config=self.config.get('technical', {})),
            'flow': FlowDetectiveAgent(config=self.config.get('flow', {})),
            'regime': RegimeSentinelAgent(config=self.config.get('regime', {})),
            'risk': RiskContrarianAgent(config=self.config.get('risk', {})),
            'earnings': EarningsOracleAgent(config=self.config.get('earnings', {})),
        }
        
        # Debate protocol
        self.debate = DebateProtocol(
            llm_model=self.config.get('llm_model', 'gpt-4o-mini'),
            use_llm=self.config.get('use_llm', True)
        )
        
        self._is_trained = False
        self._graph = None
        self._build_graph()
    
    def _load_env(self):
        """Load environment variables from .env file."""
        env_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    def _build_graph(self) -> None:
        """Build the LangGraph state machine."""
        try:
            from langgraph.graph import StateGraph, END
            
            graph = StateGraph(CouncilState)
            
            # Add nodes
            graph.add_node("detect_regime", self._node_detect_regime)
            graph.add_node("run_technical", self._node_run_technical)
            graph.add_node("run_flow", self._node_run_flow)
            graph.add_node("run_earnings", self._node_run_earnings)
            graph.add_node("run_debate", self._node_run_debate)
            graph.add_node("run_risk", self._node_run_risk)
            graph.add_node("finalize", self._node_finalize)
            
            # Set entry point
            graph.set_entry_point("detect_regime")
            
            # Parallel edges from regime to agents
            graph.add_edge("detect_regime", "run_technical")
            graph.add_edge("detect_regime", "run_flow")
            graph.add_edge("detect_regime", "run_earnings")
            
            # Agents to debate
            graph.add_edge("run_technical", "run_debate")
            graph.add_edge("run_flow", "run_debate")
            graph.add_edge("run_earnings", "run_debate")
            
            # Debate to risk to finalize
            graph.add_edge("run_debate", "run_risk")
            graph.add_edge("run_risk", "finalize")
            graph.add_edge("finalize", END)
            
            self._graph = graph.compile()
            logger.info("LangGraph workflow compiled successfully")
            
        except ImportError:
            logger.warning("LangGraph not available. Using simple orchestration.")
            self._graph = None
        except Exception as e:
            logger.warning(f"LangGraph compilation failed: {e}. Using simple orchestration.")
            self._graph = None
    
    # =========================================================================
    # GRAPH NODES
    # =========================================================================
    
    def _node_detect_regime(self, state: CouncilState) -> Dict:
        """Node 1: Detect market regime."""
        features = state['features']
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        output = self.agents['regime'].predict(features)
        regime = output.feature_attribution.get('regime_name', 'Unknown')
        
        return {
            'regime': regime,
            'regime_output': output.to_dict(),
            'audit_trail': [f"1. Regime: {regime}"]
        }
    
    def _node_run_technical(self, state: CouncilState) -> Dict:
        """Node 2a: Run Technical Analyst."""
        features = state['features']
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        try:
            output = self.agents['technical'].predict(features)
        except Exception as e:
            output = AgentOutput(signal=0.0, confidence=0.0, reasoning=str(e), agent_name="Technical Analyst")
        
        return {
            'technical_output': output.to_dict(),
            'audit_trail': [f"2a. Technical: {output.signal:+.2f}"]
        }
    
    def _node_run_flow(self, state: CouncilState) -> Dict:
        """Node 2b: Run Flow Detective."""
        features = state['features']
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        try:
            output = self.agents['flow'].predict(features)
        except Exception as e:
            output = AgentOutput(signal=0.0, confidence=0.0, reasoning=str(e), agent_name="Flow Detective")
        
        return {
            'flow_output': output.to_dict(),
            'audit_trail': [f"2b. Flow: {output.signal:+.2f}"]
        }
    
    def _node_run_earnings(self, state: CouncilState) -> Dict:
        """Node 2c: Run Earnings Oracle."""
        features = state['features']
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        try:
            output = self.agents['earnings'].predict(features)
        except Exception as e:
            output = AgentOutput(signal=0.0, confidence=0.0, reasoning=str(e), agent_name="Earnings Oracle")
        
        return {
            'earnings_output': output.to_dict(),
            'audit_trail': [f"2c. Earnings: {output.signal:+.2f}"]
        }
    
    def _node_run_debate(self, state: CouncilState) -> Dict:
        """Node 3: Debate synthesis - collects outputs from parallel agents.

        Regime Sentinel is intentionally excluded from the voting list here
        (IC=-0.040 in backtest, actively hurts signal). It still runs in
        detect_regime and provides the regime name used for threshold adjustment
        in _node_finalize.
        """
        outputs_list = []

        for key in ['technical_output', 'flow_output', 'earnings_output']:
            if state.get(key):
                outputs_list.append(AgentOutput.from_dict(state[key]))
        
        debate_result = self.debate.conduct_debate(
            agent_outputs=outputs_list,
            stock_symbol=state['stock_symbol'],
            current_price=state['current_price'],
            regime=state.get('regime', 'Unknown')
        )
        
        return {
            'debate_result': debate_result,
            'final_signal': debate_result['final_signal'],
            'final_confidence': debate_result['final_confidence'],
            'vetoed': debate_result['vetoed'],
            'veto_reason': debate_result.get('veto_reason'),
            'audit_trail': [f"3. Debate: {debate_result['final_signal']:+.2f}"]
        }
    
    def _node_run_risk(self, state: CouncilState) -> Dict:
        """Node 4: Risk assessment with veto power."""
        features = state['features']
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        try:
            output = self.agents['risk'].predict(features)
        except Exception as e:
            output = AgentOutput(signal=0.0, confidence=0.0, reasoning=str(e), agent_name="Risk Contrarian")
        
        vetoed = state.get('vetoed', False)
        veto_reason = state.get('veto_reason')
        final_signal = state.get('final_signal', 0.0)
        
        if output.signal < -0.5 and output.confidence > 0.7:
            vetoed = True
            veto_reason = output.reasoning
            final_signal = 0.0
        
        return {
            'risk_output': output.to_dict(),
            'vetoed': vetoed,
            'veto_reason': veto_reason,
            'final_signal': final_signal,
            'audit_trail': [f"4. Risk: {output.signal:+.2f}, veto={vetoed}"]
        }
    
    # Flat BUY/SELL thresholds — Regime Sentinel has IC=-0.040, so regime-dependent
    # thresholds were hurting signal quality. Regime is now informational only
    # (displayed in dashboard for context, not gating signals).
    # 0.12 chosen because horizon sweep shows IC peaks at 15d (IC=0.096) and
    # a 0.15 threshold was filtering profitable signals in the 0.12-0.15 range.
    _REGIME_THRESHOLDS: Dict[str, float] = {
        "Bull":              0.12,
        "Low-Vol Sideways":  0.12,
        "Bear":              0.12,
        "High-Vol Sideways": 0.12,
    }

    def _node_finalize(self, state: CouncilState) -> Dict:
        """Node 5: Finalize the result.

        Uses regime to adjust the BUY/SELL threshold rather than casting a vote.
        This preserves the predictive value of regime classification without
        letting its IC=-0.040 signal pollute the weighted vote.
        """
        final_signal = state.get('final_signal', 0.0)
        vetoed = state.get('vetoed', False)
        regime = state.get('regime', 'Low-Vol Sideways')

        threshold = self._REGIME_THRESHOLDS.get(regime, 0.15)

        if vetoed:
            direction = "HOLD"
        elif final_signal > threshold:
            direction = "BUY"
        elif final_signal < -threshold:
            direction = "SELL"
        else:
            direction = "HOLD"

        return {
            'direction': direction,
            'audit_trail': [f"5. Final: {direction} ({final_signal:+.2f}) threshold={threshold:.2f} regime={regime}"]
        }
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def train_all_agents(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: bool = True
    ) -> Dict[str, bool]:
        """Train all 5 agents."""
        if verbose:
            print("=" * 60)
            print("TRAINING COUNCIL OF EXPERTS")
            print("=" * 60)
        
        results = {}
        for name, agent in self.agents.items():
            if verbose:
                print(f"\n[{name.upper()}] Training {agent.name}...")
            try:
                agent.train(X_train, y_train)
                results[name] = True
                if verbose:
                    print(f"  ✅ {agent.name} trained successfully")
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = False
                if verbose:
                    print(f"  ❌ {agent.name} failed: {e}")
        
        self._is_trained = all(results.values())
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"Training complete: {sum(results.values())}/{len(self.agents)} agents")
            print("=" * 60)
        
        return results
    
    def get_signal(
        self,
        stock_symbol: str,
        features: pd.DataFrame,
        include_audit_trail: bool = True
    ) -> Dict[str, Any]:
        """Main entry point - get trading signal from the Council."""
        if not self._is_trained:
            raise ValueError("Council not trained. Call train_all_agents() first.")
        
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        
        current_price = float(features['close'].iloc[-1]) if 'close' in features.columns else 0.0
        
        initial_state: CouncilState = {
            'stock_symbol': stock_symbol,
            'features': features,
            'current_price': current_price,
            'regime': '',
            'regime_output': None,
            'technical_output': None,
            'flow_output': None,
            'earnings_output': None,
            'risk_output': None,
            'debate_result': None,
            'final_signal': 0.0,
            'final_confidence': 0.0,
            'vetoed': False,
            'veto_reason': None,
            'direction': 'HOLD',
            'audit_trail': [],
        }
        
        if self._graph is not None:
            logger.info("Running LangGraph workflow...")
            final_state = self._graph.invoke(initial_state)
        else:
            final_state = self._run_simple(initial_state)
        
        # Build agent_votes dict from separate outputs
        agent_votes = {}
        for key in ['regime_output', 'technical_output', 'flow_output', 'earnings_output', 'risk_output']:
            if final_state.get(key):
                name = key.replace('_output', '')
                agent_votes[name] = final_state[key]
        
        result = {
            'signal': final_state['final_signal'],
            'confidence': final_state['final_confidence'],
            'reasoning': final_state.get('debate_result', {}).get('synthesis', ''),
            'direction': final_state['direction'],
            'agent_votes': agent_votes,
            'regime': final_state['regime'],
            'vetoed': final_state['vetoed'],
            'veto_reason': final_state.get('veto_reason'),
            'debate_result': final_state.get('debate_result'),
        }
        
        if include_audit_trail:
            result['audit_trail'] = final_state['audit_trail']
        
        return result
    
    def _run_simple(self, state: CouncilState) -> CouncilState:
        """Simple sequential fallback."""
        logger.info("Running simple orchestration...")
        
        # Step 1: Regime
        updates = self._node_detect_regime(state)
        state = {**state, **updates}
        
        # Step 2: Parallel agents (sequential in fallback)
        for node_fn in [self._node_run_technical, self._node_run_flow, self._node_run_earnings]:
            updates = node_fn(state)
            for k, v in updates.items():
                if k == 'audit_trail':
                    state['audit_trail'] = state.get('audit_trail', []) + v
                else:
                    state[k] = v
        
        # Step 3: Debate
        updates = self._node_run_debate(state)
        for k, v in updates.items():
            if k == 'audit_trail':
                state['audit_trail'] = state.get('audit_trail', []) + v
            else:
                state[k] = v
        
        # Step 4: Risk
        updates = self._node_run_risk(state)
        for k, v in updates.items():
            if k == 'audit_trail':
                state['audit_trail'] = state.get('audit_trail', []) + v
            else:
                state[k] = v
        
        # Step 5: Finalize
        updates = self._node_finalize(state)
        for k, v in updates.items():
            if k == 'audit_trail':
                state['audit_trail'] = state.get('audit_trail', []) + v
            else:
                state[k] = v
        
        return state
    
    def get_graph_visualization(self) -> Optional[str]:
        """Get Mermaid diagram of the workflow."""
        if self._graph is None:
            return None
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return None
    
    def save(self, directory: str) -> None:
        """Save all trained agents."""
        from pathlib import Path
        import joblib
        
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, agent in self.agents.items():
            agent.save(str(output_dir / f"{name}_agent.pkl"))
        
        joblib.dump({
            'config': self.config,
            'is_trained': self._is_trained,
        }, str(output_dir / "council_meta.pkl"))
        
        logger.info(f"Council saved to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> "CouncilOfExperts":
        """Load trained council."""
        from pathlib import Path
        import joblib
        
        input_dir = Path(directory)
        meta = joblib.load(str(input_dir / "council_meta.pkl"))
        
        council = cls(config=meta['config'])
        
        for name in council.agents.keys():
            agent_path = input_dir / f"{name}_agent.pkl"
            if agent_path.exists():
                council.agents[name] = type(council.agents[name]).load(str(agent_path))
        
        council._is_trained = meta['is_trained']
        logger.info(f"Council loaded from {directory}")
        return council
    
    def get_agent(self, name: str):
        """Get a specific agent by name."""
        return self.agents.get(name)
    
    def __repr__(self) -> str:
        graph_status = "LangGraph" if self._graph else "Simple"
        status = "trained" if self._is_trained else "untrained"
        return f"CouncilOfExperts({len(self.agents)} agents, {graph_status}, {status})"
