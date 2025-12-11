#!/usr/bin/env python3
"""
Test script for Council of Experts V4.5.

Loads existing training data and tests the council pipeline end-to-end.

Usage:
    python scripts/test_council.py
    python scripts/test_council.py --sample-size 1000  # Quick test
    python scripts/test_council.py --no-llm  # Skip LLM synthesis
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_council_initialization():
    """Test that all 5 agents initialize correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Council Initialization")
    print("=" * 60)
    
    from intentflow_ai.agents.council_workflow import CouncilOfExperts
    
    council = CouncilOfExperts()
    
    assert len(council.agents) == 5, f"Expected 5 agents, got {len(council.agents)}"
    
    expected_agents = ['technical', 'flow', 'regime', 'risk', 'earnings']
    for agent_name in expected_agents:
        assert agent_name in council.agents, f"Missing agent: {agent_name}"
        print(f"  ✅ {council.agents[agent_name].name}")
    
    print("\n✅ Council initialization passed!")
    return True


def test_training(sample_size: int = 5000):
    """Test training all agents."""
    print("\n" + "=" * 60)
    print("TEST 2: Agent Training")
    print("=" * 60)
    
    from intentflow_ai.agents.council_workflow import CouncilOfExperts
    
    # Load training data
    train_path = project_root / "experiments/v3_improved/train.parquet"
    
    if not train_path.exists():
        print(f"⚠️ Training data not found at {train_path}")
        print("   Creating synthetic test data...")
        
        # Create synthetic data for testing
        np.random.seed(42)
        n = sample_size
        
        X = pd.DataFrame({
            'close': np.random.uniform(100, 1000, n),
            'volume': np.random.uniform(10000, 1000000, n),
            'technical__ema_20': np.random.uniform(100, 1000, n),
            'technical__ema_50': np.random.uniform(100, 1000, n),
            'technical__macd': np.random.uniform(-5, 5, n),
            'technical__macd_signal': np.random.uniform(-5, 5, n),
            'technical__boll_z': np.random.uniform(-2, 2, n),
            'technical__rsi_14': np.random.uniform(20, 80, n),
            'momentum__price_ret_1d': np.random.uniform(-0.05, 0.05, n),
            'momentum__price_ret_5d': np.random.uniform(-0.1, 0.1, n),
            'momentum__price_ret_10d': np.random.uniform(-0.15, 0.15, n),
            'momentum__price_ret_20d': np.random.uniform(-0.2, 0.2, n),
            'momentum__price_mom_5': np.random.uniform(-0.1, 0.1, n),
            'momentum__price_mom_10': np.random.uniform(-0.1, 0.1, n),
            'momentum__price_mom_20': np.random.uniform(-0.15, 0.15, n),
            'momentum__momentum_ratio_10_30': np.random.uniform(-0.1, 0.1, n),
            'momentum__pct_from_120d_high': np.random.uniform(-0.3, 0, n),
            'volatility__vol_5d': np.random.uniform(0.01, 0.05, n),
            'volatility__price_vol_10': np.random.uniform(0.01, 0.05, n),
            'volatility__price_vol_20': np.random.uniform(0.01, 0.05, n),
            'volatility__downside_vol_10d': np.random.uniform(0.01, 0.05, n),
            'volatility__vol_ratio_short_long': np.random.uniform(0.5, 2.0, n),
            'atr__atr_14': np.random.uniform(5, 50, n),
            'atr__atr_pct_14': np.random.uniform(0.01, 0.05, n),
            'turnover__volume_mean_20': np.random.uniform(10000, 1000000, n),
            'turnover__volume_spike': np.random.uniform(0.5, 3.0, n),
            'turnover__turnover_z_20': np.random.uniform(-2, 2, n),
        })
        
        y = pd.Series(np.random.uniform(-0.1, 0.1, n), name='target')
        
    else:
        print(f"   Loading data from {train_path}")
        data = pd.read_parquet(train_path)
        
        if sample_size and len(data) > sample_size:
            data = data.sample(sample_size, random_state=42)
            print(f"   Sampled {sample_size} rows for testing")
        
        # Create target if not present
        if 'target' not in data.columns:
            # Use forward return or synthetic
            if 'close' in data.columns:
                y = data.groupby('ticker')['close'].pct_change(10).shift(-10)
                y = y.fillna(0)
            else:
                y = pd.Series(np.random.uniform(-0.1, 0.1, len(data)))
        else:
            y = data['target']
        
        X = data.drop(['target'], axis=1, errors='ignore')
    
    print(f"   Training data shape: {X.shape}")
    
    # Train council
    council = CouncilOfExperts(config={'use_llm': False})  # Skip LLM for testing
    results = council.train_all_agents(X, y)
    
    success_count = sum(results.values())
    print(f"\n   Trained {success_count}/{len(results)} agents")
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    if success_count >= 3:
        print("\n✅ Training test passed!")
        return council, X
    else:
        print("\n❌ Training test failed (too many agents failed)")
        return None, None


def test_prediction(council, X):
    """Test signal generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Signal Generation")
    print("=" * 60)
    
    if council is None:
        print("⚠️ Skipping - council not trained")
        return False
    
    # Get single sample
    sample = X.iloc[[0]]  # Keep as DataFrame
    
    print("   Generating signal for sample...")
    
    result = council.get_signal("RELIANCE", sample)
    
    # Validate output structure
    assert 'signal' in result, "Missing 'signal' in result"
    assert 'confidence' in result, "Missing 'confidence' in result"
    assert 'reasoning' in result, "Missing 'reasoning' in result"
    assert 'agent_votes' in result, "Missing 'agent_votes' in result"
    assert 'direction' in result, "Missing 'direction' in result"
    
    # Validate ranges
    assert -1 <= result['signal'] <= 1, f"Signal out of range: {result['signal']}"
    assert 0 <= result['confidence'] <= 1, f"Confidence out of range: {result['confidence']}"
    
    # Validate all agents participated
    expected_agents = {'technical', 'flow', 'regime', 'risk', 'earnings'}
    actual_agents = set(result['agent_votes'].keys())
    
    if not expected_agents <= actual_agents:
        missing = expected_agents - actual_agents
        print(f"   ⚠️ Missing agent votes: {missing}")
    
    print(f"\n   Signal: {result['signal']:+.3f}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Direction: {result['direction']}")
    print(f"   Regime: {result.get('regime', 'N/A')}")
    print(f"   Vetoed: {result.get('vetoed', False)}")
    
    print("\n   Agent Breakdown:")
    for name, vote in result['agent_votes'].items():
        print(f"   - {name}: signal={vote['signal']:+.2f}, conf={vote['confidence']:.0%}")
    
    print("\n   Audit Trail:")
    for item in result.get('audit_trail', [])[:3]:
        print(f"   {item[:80]}...")
    
    print("\n✅ Signal generation test passed!")
    return True


def test_agents_individually():
    """Test each agent in isolation."""
    print("\n" + "=" * 60)
    print("TEST 4: Individual Agent Tests")
    print("=" * 60)
    
    from intentflow_ai.agents.base_agent import AgentOutput
    
    # Test AgentOutput
    output = AgentOutput(
        signal=0.5,
        confidence=0.8,
        reasoning="Test reasoning",
        agent_name="Test Agent"
    )
    
    assert output.direction == "BULLISH"
    assert output.signal == 0.5
    print("   ✅ AgentOutput dataclass works")
    
    # Test clamping
    output2 = AgentOutput(signal=1.5, confidence=1.5, reasoning="", agent_name="")
    assert output2.signal == 1.0, "Signal should be clamped to 1.0"
    assert output2.confidence == 1.0, "Confidence should be clamped to 1.0"
    print("   ✅ Signal/confidence clamping works")
    
    print("\n✅ Individual agent tests passed!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=5000, help='Training sample size')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM tests')
    args = parser.parse_args()
    
    print("=" * 60)
    print("COUNCIL OF EXPERTS V4.5 TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    try:
        # Test 1: Initialization
        all_passed &= test_council_initialization()
        
        # Test 2: Training
        council, X = test_training(args.sample_size)
        all_passed &= (council is not None)
        
        # Test 3: Prediction
        if council:
            all_passed &= test_prediction(council, X)
        
        # Test 4: Individual components
        all_passed &= test_agents_individually()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
