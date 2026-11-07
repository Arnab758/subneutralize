"""
Unit Tests for SubNeutralize CLI interface and Aider Proxy integration.
"""
import pytest
from unittest.mock import patch, MagicMock
from subneutralize.cli import run_aider_proxy


def test_run_aider_proxy_missing_binary():
    with patch("shutil.which", return_value=None):
        success = run_aider_proxy(host="127.0.0.1", port=8000)
        assert success is False


def test_run_aider_proxy_success():
    with patch("shutil.which", return_value="/usr/local/bin/aider"), \
         patch("subprocess.run") as mock_run:
        success = run_aider_proxy(host="127.0.0.1", port=8000, extra_args=["--yes"])
        assert success is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "--openai-api-base" in args
        assert "http://127.0.0.1:8000/v1" in args
        assert "--openai-api-key" in args
        assert "subneutralize" in args
        assert "--yes" in args
