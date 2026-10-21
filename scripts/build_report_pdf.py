"""Gera docs/relatorio_tecnico.pdf a partir do Markdown (diagramas Mermaid renderizados).

    uv run python scripts/build_report_pdf.py
    uv run python scripts/build_report_pdf.py --capa entrega/capa.md --saida entrega/entrega.pdf

Com ``--capa``, o PDF começa por esse Markdown e segue com o relatório em nova página.

Markdown → HTML (python-markdown) → PDF pelo Google Chrome headless. O Mermaid é carregado
do jsDelivr no momento da impressão, então é preciso estar com internet.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "relatorio_tecnico.md"
HTML_OUT = ROOT / "docs" / "relatorio_tecnico.html"
PDF_OUT = ROOT / "docs" / "relatorio_tecnico.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

STYLE = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1f2328; max-width: 100%; }
h1 { font-size: 20pt; margin-bottom: 0; } h2 { font-size: 14pt; border-bottom: 1px solid #d0d7de; padding-bottom: 3px;
     margin-top: 22px; page-break-after: avoid; } h3 { font-size: 12pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 4px 7px; vertical-align: top; } th { background: #f6f8fa; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f8fa; padding: 8px; border-radius: 4px; font-size: 8.5pt; white-space: pre-wrap; }
pre.mermaid { background: none; text-align: center; page-break-inside: avoid; }
pre.mermaid svg { max-height: 175mm; max-width: 100% !important; height: auto; }
blockquote { border-left: 3px solid #d0d7de; margin-left: 0; padding-left: 10px; color: #57606a; }
img { max-width: 100%; }
"""

_MERMAID = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def build(cover: Path | None = None, out: Path = PDF_OUT) -> Path:
    text = SOURCE.read_text(encoding="utf-8")
    if cover is not None:
        text = cover.read_text(encoding="utf-8") + '\n\n<div style="page-break-after: always"></div>\n\n' + text
    blocks: list[str] = []

    def stash(match: re.Match[str]) -> str:
        blocks.append(match.group(1))
        return f"\n\nMERMAIDBLOCK{len(blocks) - 1}\n\n"

    text = _MERMAID.sub(stash, text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    for i, block in enumerate(blocks):
        body = body.replace(f"<p>MERMAIDBLOCK{i}</p>", f'<pre class="mermaid">{html.escape(block)}</pre>')

    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório Técnico — Tech Challenge Fase 3</title><style>{STYLE}</style></head><body>{body}
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({{ startOnLoad: true, theme: "neutral", flowchart: {{ htmlLabels: true }} }});
</script></body></html>"""
    HTML_OUT.write_text(page, encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={out.resolve()}", HTML_OUT.as_uri()], check=True, capture_output=True)
    HTML_OUT.unlink()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capa", type=Path, help="Markdown da capa, inserido antes do relatório")
    parser.add_argument("--saida", type=Path, default=PDF_OUT)
    args = parser.parse_args()
    out = build(args.capa, args.saida)
    print(f"PDF gerado: {out} ({out.stat().st_size / 1024:.0f} KB)")
    sys.exit(0)
