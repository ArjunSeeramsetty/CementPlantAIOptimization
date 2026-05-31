"""Unit tests for the Tennessee Eastman Process (TEP) Simulator module."""

import pytest
from cement_ai_platform.simulation.tep_simulator import TennesseeEastmanSimulator


def test_tep_simulator_reset():
    """Test that TEP simulator initializes to nominal steady-state values."""
    sim = TennesseeEastmanSimulator()
    
    assert sim.state["reactor_temp"] == 120.4
    assert sim.state["reactor_pressure"] == 2705.0
    assert sim.controls["reactor_cooling_valve"] == 45.0
    assert not sim.safety_tripped
    assert sim.trip_reason == ""


def test_tep_simulator_step():
    """Test that simulator advances states and returns complete telemetry."""
    sim = TennesseeEastmanSimulator()
    telemetry = sim.step()
    
    assert "reactor_temp" in telemetry
    assert "reactor_pressure" in telemetry
    assert "safety_tripped" in telemetry
    assert sim.time_step_counter == 1


def test_tep_simulator_drift_injection():
    """Test injecting process drifts and checking their baseline impacts."""
    sim = TennesseeEastmanSimulator()
    
    # Check stable catalyst decay impact
    sim.inject_drift("catalyst_decay")
    assert "catalyst_decay" in sim.active_drifts
    
    # Step a few times to let drift build up
    for _ in range(5):
        sim.step()
        
    effects = sim._calculate_drift_effects()
    assert effects["reaction_mult"] < 1.0  # Catalyst decay should reduce yield


def test_tep_simulator_safety_trips():
    """Test that safety limits trigger interlock shutdown."""
    sim = TennesseeEastmanSimulator()
    
    # 1. Force reactor temperature breach
    sim.state["reactor_temp"] = 136.0
    sim.step()
    assert sim.safety_tripped
    assert "TEMPERATURE" in sim.trip_reason
    
    # Check shutdown dynamics cooled it down
    assert sim.state["reactor_temp"] < 136.0
    assert sim.state["reactor_pressure"] < 2705.0
    
    # 2. Reset and force reactor pressure breach
    sim.reset()
    sim.state["reactor_pressure"] = 3150.0
    sim.step()
    assert sim.safety_tripped
    assert "PRESSURE" in sim.trip_reason


def test_generate_history():
    """Test generating full consecutive simulation history with drift."""
    sim = TennesseeEastmanSimulator()
    history = sim.generate_history(num_steps=30, drift_to_inject="feed_d_temp_step")
    
    assert len(history) == 30
    assert history[0]["step_idx"] == 0.0
    assert history[-1]["step_idx"] == 29.0
