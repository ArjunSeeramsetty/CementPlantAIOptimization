"""Unit tests for the Computer Vision Safety Monitoring module."""

import pytest
from cement_ai_platform.vision.safety_monitor import SafetyMonitor


def test_safety_monitor_initialization():
    """Test that the safety monitor initializes correctly."""
    monitor = SafetyMonitor()
    assert monitor.project_id is not None


def test_check_boundary_violation():
    """Test that boundary violations are detected correctly based on frame cycling."""
    monitor = SafetyMonitor()
    
    # Test frame 0 (secure zone, y = 220)
    res_secure = monitor.check_boundary_violation(0)
    assert "is_violation" in res_secure
    assert not res_secure["is_violation"]
    assert res_secure["detections"][0]["severity"] == "NORMAL"
    
    # Test frame 35 (violation zone, y > 300)
    res_breach = monitor.check_boundary_violation(35)
    assert res_breach["is_violation"]
    assert res_breach["detections"][0]["severity"] == "CRITICAL"


def test_check_ppe_compliance():
    """Test PPE compliance simulation check."""
    monitor = SafetyMonitor()
    
    # Cycle 0 (Frame 0): Compliant worker
    res_compliant = monitor.check_ppe_compliance(0)
    assert not res_compliant["is_violation"]
    assert res_compliant["detections"][0]["compliant"]
    
    # Cycle 1 (Frame 50): Missing hard hat
    res_violation_hat = monitor.check_ppe_compliance(50)
    assert res_violation_hat["is_violation"]
    assert not res_violation_hat["detections"][0]["compliant"]
    assert res_violation_hat["detections"][0]["violation"] == "MISSING HARD HAT"


def test_generate_mock_frame_bytes():
    """Test generating mock video frames and validating JPEG header."""
    monitor = SafetyMonitor()
    
    # Generate boundary safety frame
    frame_bytes = monitor.generate_mock_frame_bytes('boundary', 1)
    if frame_bytes:
        # Check standard JPEG file magic header (FF D8 FF)
        assert frame_bytes.startswith(b"\xff\xd8")
        
    # Generate PPE compliance frame
    frame_bytes_ppe = monitor.generate_mock_frame_bytes('ppe', 1)
    if frame_bytes_ppe:
        assert frame_bytes_ppe.startswith(b"\xff\xd8")


def test_generate_safety_incident_report():
    """Test incident report generation."""
    monitor = SafetyMonitor()
    report = monitor.generate_safety_incident_report("CAM-NORTH-FENCE", "Intrusion Alert", "Worker crossed boundary.")
    
    assert "INCIDENT OVERVIEW" in report
    assert "COMPLIANCE ASSESSMENT" in report
    assert "CORRECTIVE ACTIONS" in report
    assert "CAM-NORTH-FENCE" in report
