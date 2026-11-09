"""Check local Markdown destinations and unfinished final report sections, read-only."""
from pathlib import Path
import re
from urllib.parse import unquote

from helpdesk.paths import ROOT


def main():
    documents = ['README.md', 'SUBMISSION.md', 'outputs/README.md', 'outputs/final/VERIFICATION.md']
    for name in documents:
        path = ROOT / name
        text = path.read_text(encoding='utf-8')
        for destination in re.findall(r'\]\(([^)]+)\)', text):
            destination = destination.split('#', 1)[0]
            if not destination or re.match(r'\w+://', destination):
                continue
            assert (path.parent / unquote(destination)).exists(), (name, destination)
        if name == 'SUBMISSION.md':
            assert not re.search(r'\b(?:TODO|TBD|FIXME)\b', text)
            for number in range(1, 15):
                row = next(line for line in text.splitlines() if line.startswith(f'| S{number:02} '))
                assert 'PASS' in row and '`python -m helpdesk.cli' in row
    print('Documentation links and fourteen complete scenario rows: PASS')


if __name__ == '__main__':
    main()
