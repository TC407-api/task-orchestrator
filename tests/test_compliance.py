import pytest
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.governance.compliance import (
    ReportGenerator,
    ComplianceFramework
)

@pytest.fixture
def sample_audit_logs() -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": datetime.now().isoformat(),
            "action": "access_record",
            "user": "admin@example.com",
            "status": "success",
            "details": "Accessed patient record 123-45-6789"
        },
        {
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            "action": "delete_db",
            "user": "unknown",
            "status": "failure",
            "error": "Permission denied"
        }
    ]

@pytest.fixture
def generator() -> ReportGenerator:
    return ReportGenerator()

def test_soc2_report_has_required_sections(
    generator: ReportGenerator,
    sample_audit_logs: List[Dict[str, Any]]
):
    """Test that SOC2 reports contain specific required headers."""
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()

    report = generator.generate(
        ComplianceFramework.SOC2, start, end, sample_audit_logs
    )

    assert "security_controls" in report.sections
    assert "availability_metrics" in report.sections
    assert "access_logs" in report.sections
    assert report.framework == ComplianceFramework.SOC2

def test_hipaa_report_redacts_phi(
    generator: ReportGenerator,
    sample_audit_logs: List[Dict[str, Any]]
):
    """Test that HIPAA reports redact SSNs and Emails."""
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()

    # Inject PHI into logs specifically for this test
    logs = sample_audit_logs + [{
        "details": "Sent email to patient@hospital.com regarding SSN 999-00-1234"
    }]

    report = generator.generate(
        ComplianceFramework.HIPAA, start, end, logs
    )

    # Flatten findings to check strings
    report_text = json.dumps(report.findings)

    assert "patient@hospital.com" not in report_text
    assert "999-00-1234" not in report_text
    assert "[REDACTED]" in report_text

def test_report_includes_time_range_summary(
    generator: ReportGenerator,
    sample_audit_logs: List[Dict[str, Any]]
):
    """Test that the report accurately reflects the requested time window."""
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 31)

    report = generator.generate(
        ComplianceFramework.ISO27001, start, end, sample_audit_logs
    )

    assert report.start_date == start
    assert report.end_date == end
    # Ensure logs outside range are excluded (logic to be implemented)
    assert isinstance(report.sections.get("meta"), dict)

def test_report_includes_failure_analysis(
    generator: ReportGenerator,
    sample_audit_logs: List[Dict[str, Any]]
):
    """Test that failed actions are aggregated in findings."""
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()

    report = generator.generate(
        ComplianceFramework.SOC2, start, end, sample_audit_logs
    )

    failures = [f for f in report.findings if f.get("type") == "security_incident"]
    assert len(failures) > 0
    assert failures[0]["severity"] == "high"
    assert "delete_db" in str(failures[0])

def test_report_exports_pdf_and_json(
    generator: ReportGenerator,
    sample_audit_logs: List[Dict[str, Any]],
    tmp_path: Any
):
    """Test that export methods create files."""
    start = datetime.now()
    end = datetime.now()
    report = generator.generate(
        ComplianceFramework.SOC2, start, end, sample_audit_logs
    )

    json_path = tmp_path / "report.json"
    pdf_path = tmp_path / "report.pdf"

    report.export_json(str(json_path))
    report.export_pdf(str(pdf_path))

    assert os.path.exists(json_path)
    assert os.path.exists(pdf_path)
