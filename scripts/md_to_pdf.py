#!/usr/bin/env python3
"""Convert Markdown to HTML and inline images (SVGs) for PDF rendering.

Writes `docs/ML_PIPELINE.html` from `docs/ML_PIPELINE.md` and inlines local SVG images.
"""
import os
import io
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / 'docs' / 'ML_PIPELINE.md'
OUT_HTML = ROOT / 'docs' / 'ML_PIPELINE.html'

def inline_svgs(html: str, base_dir: Path) -> str:
    # replace <img src="./diagrams/foo.svg"> or ![](...) markdown conversion artifacts
    # simple approach: find occurrences of src="./diagrams/*.svg" and replace with inline content
    import re
    def repl(m):
        src = m.group(1)
        path = (base_dir / src).resolve()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                svg = f.read()
            # remove xml header if present to avoid duplicates
            svg = svg.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
            return svg
        except Exception:
            return m.group(0)

    # handle src="..."
    html = re.sub(r'src=["\'](\.\/diagrams\/.+?\.svg)["\']', repl, html)
    # handle src="/absolute/path/..."
    html = re.sub(r'src=["\'](docs\/diagrams\/.+?\.svg)["\']', lambda m: repl((m.group(1))), html)
    return html

def main():
    md_text = MD.read_text(encoding='utf-8')
    # Replace markdown image references to local diagrams with inline SVG content
    import re
    def _embed_svg_markdown(m):
        fname = m.group(1)
        svg_path = (MD.parent / 'diagrams' / fname).resolve()
        try:
            svg_text = svg_path.read_text(encoding='utf-8')
            svg_text = svg_text.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
            return '\n' + svg_text + '\n'
        except Exception:
            return m.group(0)

    md_text = re.sub(r'!\[[^\]]*\]\(\.?\/?diagrams\/(.+?\.svg)\)', _embed_svg_markdown, md_text)

    html = markdown.markdown(md_text, extensions=['fenced_code', 'codehilite', 'tables', 'nl2br'])
    # wrap in a minimal HTML document
    full = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NextHorizon ML Pipeline</title>
<style>body{{font-family: Arial, Helvetica, sans-serif; margin: 28px;}} pre{{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto}} code{{font-family: monospace;}}</style>
</head>
<body>
{html}
</body>
</html>
"""

    # Inline local SVGs
    out_html = inline_svgs(full, ROOT / 'docs')
    OUT_HTML.write_text(out_html, encoding='utf-8')
    print('Wrote', OUT_HTML)

if __name__ == '__main__':
    main()
