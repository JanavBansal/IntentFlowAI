"""
Council of Experts Workflow - Main Orchestration.

This is the main entry point for V4.5 that replaces V3's monolithic model.

Workflow:
1. Regime Detection (affects other agent weights)
2. Parallel Agent Analysis (Technical, Flow, Earnings)
3. Debate Synthesis
4. Risk Assessment (can veto)
5. Final Signal

Uses LangGraph for state machine orchestration (optional - falls back to simple orchestration).
"""

from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd

from intentflow_ai.agents.base_agent import AgentOutput
from intentflow_ai.agents.technical_analyst import TechnicalAnalystAgent
from intentflow_ai.agents.flow_detective import FlowDetectiveAgent
from intentflow_ai.agents.regime_sentinel import RegimeSentinelAgent
from intentflow_ai.agents.risk_contrarian import RiskContrarianAgent
from intentflow_ai.agents.earnings_oracle import EarningsOracleAgent
from intentflow_ai.agents.debate_protocol import DebateProtocol
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class CouncilState(TypedDict):
    """State passed through the workflow."""
    stock_symbol: str
    date: str
    features: pd.DataFrame
    current_price: float
    regime: str
    agent_outputs: Dict[str, AgentOutput]
    debate_result: Optional[Dict]
    final_signal: Optional[float]
    audit_trail: List[str]


class CouncilOfExperts:
    """
    Main orchestration class for V4.5 Council of Experts.
    
    Replaces the monolithic V3 model with a multi-agent system.
    
    Usage:
        council = CouncilOfExperts()
        council.train_all_agents(X_train, y_train)
        
        result = council.get_signal("RELIANCE", features_df)
        print(result['signal'], result['reasoning'])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Council of Experts.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize all 5 agents
        self.agents = {
            'technical': TechnicalAnalystAgent(
                config=self.config.get('technical', {})
            ),
            'flow': FlowDetectiveAgent(
                config=self.config.get('flow', {})
            ),
            'regime': RegimeSentinelAgent(
                config=self.config.get('regime', {})
            ),
            'risk': RiskContrarianAgent(
                config=self.config.get('risk', {})
            ),
            'earnings': EarningsOracleAgent(
                config=self.config.get('earnings', {})
            ),
        }
        
        # Debate protocol
        self.debate = DebateProtocol(
            llm_model=self.config.get('llm_model', 'gpt-4o-mini'),
            use_llm=self.config.get('use_llm', True)
        )
        
        # State
        self._is_trained = False
        self._langgraph_available = self._check_langgraph()
    
    def _check_langgraph(self) -> bool:
        """Check if LangGraph is available for workflow orchestration."""
        try:
            from langgraph.graph import StateGraph
            return True
        except ImportError:
            logger.info("LangGraph not available. Using simple orchestration.")
            return False
    
    def train_all_agents(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: bool = True
    ) -> Dict[str, bool]:
        """
        Train all 5 agents on the training data.
        
        Args:
            X_train: Training features (should contain features for all agents)
            y_train: Target variable (forward returns)
            verbose: Print training progress
            
        Returns:
            Dict mapping agent name to training success status
        """
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
            success_count = sum(results.values())
            print(f"Training complete: {success_count}/{len(self.agents)} agents trained")
            print("=" * 60)
        
        return results
    
    def get_signal(
        self,
        stock_symbol: str,
        features: pd.DataFrame,
        include_audit_trail: bool = True
    ) -> Dict[str, Any]:
        """
        Main entry point - get trading signal from the Council.
        
        Replaces V3's model.predict() with multi-agent synthesis.
        
        Args:
            stock_symbol: e.g., "RELIANCE"
            features: DataFrame with all required features (can be single row)
            include_audit_trail: Include detailed agent reasoning
            
        Returns:
            {
                'signal': float (-1 to 1),
                'confidence': float (0 to 1),
                'reasoning': str (LLM synthesis),
                'direction': str ('BUY', 'SELL', 'HOLD'),
                'agent_votes': Dict[str, AgentOutput],
                'regime': str,
                'vetoed': bool,
                'audit_trail': List[str] (if requested)
            }
        """
        if not self._is_trained:
            raise ValueError("Council not trained. Call train_all_agents() first.")
        
        audit_trail = []
        
        # Ensure features is a DataFrame with at least 1 row
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        
        # Get current price for debate context
        current_price = float(features['close'].iloc[-1]) if 'close' in features.columns else 0.0
        
        # ==========================================================
        # STEP 1: Regime Detection (runs first, affects other agents)
        # ==========================================================
        regime_output = self.agents['regime'].predict(features)
        regime = regime_output.feature_attribution.get('regime_name', 'Unknown')
        audit_trail.append(f"1. Regime: {regime_output.reasoning}")
        
        # ==========================================================
        # STEP 2: Parallel Agent Analysis
        # ==========================================================
        agent_outputs = {'regime': regime_output}
        
        for name in ['technical', 'flow', 'earnings']:
            agent = self.agents[name]
            try:
                output = agent.predict(features)
                agent_outputs[name] = output
                audit_trail.append(f"2. {agent.name}: {output.reasoning}")
            except Exception as e:
                logger.warning(f"{name} prediction failed: {e}")
                # Create neutral fallback output
                agent_outputs[name] = AgentOutput(
                    signal=0.0,
                    confidence=0.0,
                    reasoning=f"Prediction failed: {str(e)[:50]}",
                    agent_name=agent.name
                )
        
        # ==========================================================
        # STEP 3: Debate Synthesis
        # ==========================================================
        debate_result = self.debate.conduct_debate(
            agent_outputs=list(agent_outputs.values()),
            stock_symbol=stock_symbol,
            current_price=current_price,
            regime=regime
        )
        audit_trail.append(f"3. Debate: {debate_result['synthesis'][:100]}...")
        
        # ==========================================================
        # STEP 4: Risk Assessment (can veto)
        # ==========================================================
        risk_output = self.agents['risk'].predict(features)
        agent_outputs['risk'] = risk_output
        audit_trail.append(f"4. Risk: {risk_output.reasoning}")
        
        # Apply risk veto
        final_signal = debate_result['final_signal']
        vetoed = debate_result['vetoed']
        
        if risk_output.signal < -0.5 and risk_output.confidence > 0.7:
            final_signal = 0.0
            vetoed = True
            audit_trail.append("5. ⛔ VETOED by Risk Contrarian")
        else:
            audit_trail.append(f"5. Final signal: {final_signal:+.2f}")
        
        # ==========================================================
        # Build Result
        # ==========================================================
        # Determine direction
        if vetoed:
            direction = "HOLD"
        elif final_signal > 0.15:
            direction = "BUY"
        elif final_signal < -0.15:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        result = {
            'signal': final_signal,
            'confidence': debate_result['final_confidence'],
            'reasoning': debate_result['synthesis'],
            'direction': direction,
            'agent_votes': {
                name: output.to_dict() for name, output in agent_outputs.items()
            },
            'regime': regime,
            'vetoed': vetoed,
            'debate_result': debate_result,
        }
        
        if include_audit_trail:
            result['audit_trail'] = audit_trail
        
        return result
    
    def save(self, directory: str) -> None:
        """Save all trained agents to directory."""
        from pathlib import Path
        import joblib
        
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, agent in self.agents.items():
            agent.save(str(output_dir / f"{name}_agent.pkl"))
        
        # Save config
        joblib.dump({
            'config': self.config,
            'is_trained': self._is_trained,
        }, str(output_dir / "council_meta.pkl"))
        
        logger.info(f"Council saved to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> "CouncilOfExperts":
        """Load trained council from directory."""
        from pathlib import Path
        import joblib
        
        input_dir = Path(directory)
        
        # Load meta
        meta = joblib.load(str(input_dir / "council_meta.pkl"))
        
        # Create council
        council = cls(config=meta['config'])
        
        # Load agents
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
        status = "trained" if self._is_trained else "untrained"
        return f"CouncilOfExperts({len(self.agents)} agents, status={status})"
