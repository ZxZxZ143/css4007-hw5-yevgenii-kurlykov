"""Replay all fixture scenarios offline in disposable storage; never overwrite live evidence."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from helpdesk.part1.cli import PREDICTIONS
from helpdesk.part1.graphs import build_main_graph, build_original_graph
from helpdesk.part1.trace import execute
from helpdesk.part2.runner import run_scenario
from helpdesk.part2.tools import read_fixture
from helpdesk.part3.scenarios import run_memory_scenario
from scripts.part3.verify_evidence import verify_part3
from tests.part2.fake_model import DemoFakeModel


def check_agent(log, scenario, storage):
    turns = log['turns']
    assert [t['user_message'] for t in turns] == [e['message'] for e in scenario['events']]
    assert all(t['control']['stop_reason'] == 'final_response' and not t['control']['errors'] for t in turns)
    entries = [entry for turn in turns for step in turn['steps']
               for entry in step['state_after']['tool_results'].values()]
    statuses = [e['data'] for e in entries if e['tool'] == 'check_service_status']
    sid = scenario['id']
    drafts = list(storage.rglob('DRAFT-*.json'))
    assert len(drafts) == (1 if sid in ('S06', 'S09') else 0)
    if sid != 'S09':
        expected = {'S06': 'operational', 'S07': 'outage', 'S08': 'unknown', 'S14': 'outage'}[sid]
        assert any(s['status'] == expected for s in statuses)
    if sid == 'S14':
        assert all(s['service'] == 'account_recovery' for s in statuses)
    if sid == 'S09':
        assert turns[0]['control']['tool_calls'] == 0 and 'steps' in turns[0]['final_response'].lower()
    for path in drafts:
        draft = json.loads(path.read_text(encoding='utf-8'))
        assert all(span in '\n'.join(t['user_message'] for t in turns) for span in draft['steps_tried'])
        confirmation = json.loads(turns[-1]['final_response'].split('\n', 1)[1])
        assert all(confirmation[k] == draft[k] for k in ('ticket_id', 'category', 'summary', 'steps_tried'))
    return {'draft_count': len(drafts), 'statuses': statuses, 'turns': len(turns)}


def verify_all(output=None):
    results = []
    with TemporaryDirectory(prefix='hw5-final-') as temporary:
        root = Path(temporary)
        for scenario in read_fixture('scenarios.json')['scenarios']:
            sid = scenario['id']
            row = {'scenario_id': sid, 'expected_behavior': scenario['title'],
                   'execution_method': 'real LangGraph; deterministic fake model' if scenario['part'] != 1 else 'real deterministic LangGraph; no model',
                   'real_model_used': False, 'evidence_file': f'outputs/part1/{"improved" if sid == "S05" else "original"}/{sid}.json' if scenario['part'] == 1 else f'outputs/part{scenario["part"]}/fake/{sid}.json'}
            try:
                if scenario['part'] == 1:
                    version, target = PREDICTIONS[sid]
                    log = execute(build_original_graph() if version == 'original' else build_main_graph(),
                                  scenario['events'][0]['message'], version=version,
                                  predicted_route=['read_request', target], metadata={})
                    assert log['prediction_matched'] and log['final_response']
                    row['observed_behavior'] = {'route': log['nodes_executed']}
                elif scenario['part'] == 2:
                    storage = root / sid
                    log = run_scenario(sid, model=DemoFakeModel(), model_name='DemoFakeModel (offline)',
                                       evidence_kind='deterministic_fake', storage_dir=storage, output=root / f'{sid}.json')
                    row['observed_behavior'] = check_agent(log, scenario, storage)
                else:
                    log = run_memory_scenario(sid, storage_dir=root, output=root / 'part3' / f'{sid}.json', model_mode='fake')
                    answer = log['events'][1]['final_response']
                    if sid in ('S10', 'S11'):
                        assert 'enabled' in answer and 'reconnect' in answer
                        assert 'airplane mode' not in answer
                    if sid == 'S10':
                        assert 'forgot the saved network' in answer
                    if sid in ('S11', 'S12'):
                        assert answer.startswith('FAKE kk')
                    if sid == 'S13':
                        assert 'No preference or previous problem' in answer
                    row['observed_behavior'] = {'restart': log['restart_occurred'], 'process_ids': [p['pid'] for p in log['processes']],
                                                'restored': log['events'][1]['memory']['checkpoint_restored'],
                                                'preferences': log['events'][1]['memory']['user_preferences_after']}
                    # The complete cross-scenario validator below also checks tool IDs and source separation.
                row['status'] = 'PASS'
            except Exception as error:
                row.update(status='FAIL', observed_behavior={'error': type(error).__name__, 'detail': str(error)})
            results.append(row)
        if all(r['status'] == 'PASS' for r in results if r['scenario_id'] in ('S10','S11','S12','S13')):
            try:
                verify_part3(root / 'part3', evidence_kind='deterministic_fake')
            except Exception as error:
                for row in results:
                    if row['scenario_id'] in ('S10','S11','S12','S13'):
                        row.update(status='FAIL', observed_behavior={'error': type(error).__name__})
    report = {'evidence_kind': 'fresh_offline_verification', 'limitations': 'Fake models verify application behavior, not new OpenAI behavior. Full temporary traces are discarded; evidence_file links retained independent examples.', 'scenarios': results}
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional summary destination; default writes no repository files')
    args = parser.parse_args()
    report = verify_all(args.output)
    for row in report['scenarios']:
        print(row['scenario_id'], row['status'])
    return int(any(row['status'] != 'PASS' for row in report['scenarios']))


if __name__ == '__main__':
    raise SystemExit(main())
