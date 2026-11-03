#!/usr/bin/env python3
"""Fetch reference sources for the local reference audit."""
import html.parser
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data/papers'

class Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.hidden += 1
        if tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3'): self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.hidden -= 1
        if tag in ('p', 'div', 'li'): self.parts.append('\n')
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    last_request = 0.0
    failures = []
    for entry in yaml.safe_load((ROOT / 'paper/refs/verified.yaml').read_text()):
        ident = str(entry.get('arxiv_id', ''))
        target = OUT / (ident + '.pdf' if ident else entry['key'] + '.txt')
        url = 'https://arxiv.org/pdf/' + ident if ident else entry.get('url')
        try:
            if not target.exists():
                if not url: raise ValueError('no source URL')
                time.sleep(max(0, 3 - (time.monotonic() - last_request)))
                last_request = time.monotonic()
                result = subprocess.run(['curl', '-fsSL', '--max-time', '90', url], check=True, capture_output=True)
                if ident: target.write_bytes(result.stdout)
                else:
                    parser = Text(); parser.feed(result.stdout.decode('utf-8', errors='replace'))
                    target.write_text(''.join(parser.parts))
            if ident:
                subprocess.run(['pdftotext', str(target), str(OUT / (ident + '.txt'))], check=True)
            print(entry['key'], 'ok', flush=True)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            print(entry['key'], 'FAILED', exc, flush=True); failures.append(entry['key'])
    return bool(failures)

if __name__ == '__main__': raise SystemExit(main())
