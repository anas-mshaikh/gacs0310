"""Tests for the secret scanner (check_secrets.py)."""

import subprocess
import sys

# import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


class TestSecretScanner:
    """Test check_secrets.py detection capabilities."""

    def test_scanner_exists(self):
        assert (PROJECT_ROOT / "check_secrets.py").exists()

    def test_scanner_runs_clean(self):
        """The current repo should pass the secret scan."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "check_secrets.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Secret scan failed:\n{result.stdout}\n{result.stderr}"

    def test_scanner_import(self):
        """check_secrets.py should be importable."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_secrets", str(PROJECT_ROOT / "check_secrets.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "COMPILED_PATTERNS")
        assert hasattr(mod, "scan_file")


class TestSecretPatterns:
    """Test that secret patterns catch real secrets."""

    @pytest.fixture
    def scanner(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_secrets", str(PROJECT_ROOT / "check_secrets.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_detects_anthropic_key(self, scanner, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('API_KEY = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA"\n')
        findings = scanner.scan_file(str(f))
        assert len(findings) > 0, "Should detect Anthropic API key"

    def test_detects_aws_key(self, scanner, tmp_path):
        f = tmp_path / "test.py"
        # Note: avoid "EXAMPLE" in value as it matches placeholder filter
        f.write_text('aws_key = "AKIAIOSFODNN7ABCDEFG"\n')
        findings = scanner.scan_file(str(f))
        assert len(findings) > 0, "Should detect AWS key"

    def test_detects_github_token(self, scanner, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"\n')
        findings = scanner.scan_file(str(f))
        assert len(findings) > 0, "Should detect GitHub token"

    def test_detects_private_key(self, scanner, tmp_path):
        f = tmp_path / "test.pem"
        f.write_text("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n")
        findings = scanner.scan_file(str(f))
        assert len(findings) > 0, "Should detect private key"

    def test_detects_hardcoded_password(self, scanner, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('password = "SuperSecret123!"\n')
        findings = scanner.scan_file(str(f))
        assert len(findings) > 0, "Should detect hardcoded password"

    def test_ignores_placeholder(self, scanner, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('API_KEY = "your-api-key-here"\n')
        findings = scanner.scan_file(str(f))
        assert len(findings) == 0, "Should ignore placeholder values"

    def test_ignores_env_example(self, scanner, tmp_path):
        f = tmp_path / ".env.example"
        f.write_text("ANTHROPIC_API_KEY=your-anthropic-api-key-here\n")
        findings = scanner.scan_file(str(f))
        assert len(findings) == 0, "Should ignore .env.example"
