"""Final delivery checks exercise the real offline runner and read-only evidence CLIs."""
import hashlib
from pathlib import Path
import subprocess
import sys

from helpdesk.paths import ROOT
from scripts.verify_all import verify_all


def test_all_fourteen_offline_scenarios_without_provider_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', '')
    monkeypatch.setenv('MODEL_NAME', '')
    report = verify_all(tmp_path / 'summary.json')
    assert [r['scenario_id'] for r in report['scenarios']] == [f'S{i:02}' for i in range(1, 15)]
    assert all(r['status'] == 'PASS' and not r['real_model_used'] for r in report['scenarios'])
    assert all((ROOT / r['evidence_file']).is_file() for r in report['scenarios'])


def test_saved_evidence_check_commands_do_not_write_artifacts():
    def hashes():
        return {p: hashlib.sha256(p.read_bytes()).digest() for p in (ROOT / 'outputs').rglob('*') if p.is_file()}
    before = hashes()
    for module in ('scripts.part2.verify_live', 'scripts.part3.verify_evidence'):
        subprocess.run([sys.executable, '-m', module, '--check'], cwd=ROOT, check=True, capture_output=True, timeout=40)
    assert hashes() == before


def test_final_cli_failure_returns_nonzero(monkeypatch, capsys):
    import scripts.verify_all as module
    monkeypatch.setattr(sys, 'argv', ['verify_all'])
    monkeypatch.setattr(module, 'verify_all', lambda _: {'scenarios': [{'scenario_id': 'S06', 'status': 'FAIL'}]})
    assert module.main() == 1
    assert 'S06 FAIL' in capsys.readouterr().out
