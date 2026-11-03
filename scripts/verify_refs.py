#!/usr/bin/env python3
"""Check contiguous excerpts and arXiv metadata; failures block paper review."""
import difflib
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('\u00ad', '')
    text = re.sub(r'-\s*\n\s*(?=\w)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def bib_fields(text, key):
    match = re.search(r'@\w+\s*\{\s*' + re.escape(key) + r'\s*,(.*?)(?=\n\s*@\w+\s*\{|\Z)', text, re.S | re.I)
    if not match: return None, None
    author = re.search(r'\bauthor\s*=\s*\{([^{}]*)\}', match[1], re.I)
    year = re.search(r'\byear\s*=\s*\{([^{}]*)\}', match[1], re.I)
    return ([x.strip() for x in author[1].split(' and ')] if author else None, year[1] if year else None)

def main():
    entries = yaml.safe_load((ROOT / 'paper/refs/verified.yaml').read_text())
    bib = (ROOT / 'paper/refs/refs.bib').read_text()
    ids = [str(e['arxiv_id']) for e in entries if e.get('arxiv_id')]
    api = {}
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'id_list': ','.join(ids), 'max_results': len(ids)})
        with urllib.request.urlopen(url, timeout=45) as response: root = ET.fromstring(response.read())
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        for item in root.findall('a:entry', ns):
            ident = item.findtext('a:id', namespaces=ns).rsplit('/', 1)[-1]
            ident = re.sub(r'v\d+$', '', ident)
            api[ident] = ([x.findtext('a:name', namespaces=ns) for x in item.findall('a:author', ns)], item.findtext('a:published', namespaces=ns)[:4], normalize(item.findtext('a:title', namespaces=ns)))
    except Exception as exc:
        print('arXiv API error:', exc, file=sys.stderr)
    failed = False
    print(f'{"key":38} result')
    for entry in entries:
        problems = []
        if entry.get('status') != 'verified': problems.append('status is not verified')
        ident = str(entry.get('arxiv_id', ''))
        path = ROOT / entry.get('source_file', f'data/papers/{ident or entry["key"]}.txt')
        try: source = normalize(path.read_text(errors='replace'))
        except OSError: source = ''; problems.append('source missing')
        for field in ('supporting_excerpt', 'supporting_excerpt_2'):
            excerpt = entry.get(field)
            if not excerpt:
                if field == 'supporting_excerpt': problems.append('excerpt missing')
                continue
            needle = normalize(excerpt)
            if len(needle.split()) > 60: problems.append(field + ': exceeds 60 words')
            if '...' in needle or '…' in needle: problems.append(field + ': ellipsis')
            elif needle not in source:
                problems.append(field + ': mismatch')
                if source:
                    words = source.split(); n = max(1, len(needle.split()))
                    anchors = set(needle.lower().split()[:5])
                    positions = [i for i, word in enumerate(words) if word.lower() in anchors][:500]
                    candidates = [' '.join(words[i:i+n]) for i in positions]
                    close = difflib.get_close_matches(needle, candidates, n=1, cutoff=0)
                    if close: print(f'  {entry["key"]} nearest: {close[0]}')
        if ident:
            meta = api.get(ident)
            if not meta: problems.append('API metadata missing')
            else:
                authors, year, title = meta
                ba, by = bib_fields(bib, entry['key'])
                if entry.get('authors') != authors: problems.append('YAML authors differ')
                if str(entry.get('year')) != year: problems.append('YAML year differs')
                if normalize(entry.get('title', '')) != title: problems.append('YAML title differs')
                if ba != authors: problems.append('BibTeX authors differ')
                if by != year: problems.append('BibTeX year differs')
        print(f'{entry["key"]:38} {", ".join(problems) if problems else "verified"}')
        failed |= bool(problems)
    return int(failed)

if __name__ == '__main__': raise SystemExit(main())
