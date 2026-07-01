"""
propepx_visualizer.py
======================
ProPepX Binding-Site Sequence Visualiser
-----------------------------------------
Renders protein and peptide binding-site predictions as richly formatted
output suitable for:
    - Terminal / CLI   (ANSI colour codes, 80-column wrapped)
    - Plain text       (no colour, for logging)
    - HTML             (colour-coded spans for web-server use)

The HTML output is ready to be embedded in the ProPepX web server.

Colour scheme
-------------
Binding     : bold bright red background   (terminal) / #E53E3E  (HTML)
Non-binding : dim grey                     (terminal) / #718096  (HTML)

Authors: Syed Kumail Hussain Naqvi et al.
"""

import textwrap
import os

"""
propepx_visualizer.py
======================
ProPepX Binding-Site Sequence Visualiser
"""
import os


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_RED = "\033[91m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_WHITE = "\033[97m"

_BG_RED = "\033[41m"

LINE_WIDTH = 100
HTML_LINE_WIDTH = 90

AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}


def _binding_stats(seq: str, pred: list) -> dict:
    total = len(seq)
    n_bind = sum(pred)
    residues = [seq[i] for i, p in enumerate(pred) if p == 1]
    return {
        "total": total,
        "n_binding": n_bind,
        "n_nonbinding": total - n_bind,
        "pct_binding": 100.0 * n_bind / total if total > 0 else 0.0,
        "binding_residues": residues,
        "unique_binding_aa": sorted(set(residues)),
    }


def _section_header(title: str, width: int = LINE_WIDTH) -> str:
    pad = (width - len(title) - 2) // 2
    return (
        f"\n{_BOLD}{_CYAN}"
        f"{'━' * pad} {title} {'━' * (width - pad - len(title) - 2)}"
        f"{_RESET}\n"
    )


def _format_binding_residues_inline(seq: str, pred: list, prob=None) -> str:
    items = []

    for i, p in enumerate(pred):
        if p == 1:
            aa1 = seq[i]
            aa3 = AA3.get(aa1, aa1)
            pos = i + 1

            if prob is not None:
                try:
                    probability = float(prob[i])
                    items.append(f"{aa3}{pos} ({probability:.3f})")
                except Exception:
                    items.append(f"{aa3}{pos}")
            else:
                items.append(f"{aa3}{pos}")

    return "  ".join(items) if items else "None"


def _binding_residues_inline_html(seq: str, pred: list, prob=None, title: str = "Binding Residues") -> str:
    items = []

    for i, p in enumerate(pred):
        if p == 1:
            aa1 = seq[i]
            aa3 = AA3.get(aa1, aa1)
            pos = i + 1

            if prob is not None:
                try:
                    probability = float(prob[i])
                    text = f"{aa3}{pos} ({probability:.3f})"
                except Exception:
                    text = f"{aa3}{pos}"
            else:
                text = f"{aa3}{pos}"

            items.append(f'<span class="br-chip">{text}</span>')

    chips = "".join(items) if items else "<em>None</em>"

    return (
        f'<div style="margin-top:14px;font-size:.8rem;color:#4A5568;font-weight:600;">'
        f'{title} with ProPepX probability</div>'
        f'<div class="binding-residues">{chips}</div>'
    )


def _render_sequence_terminal(seq: str, pred: list, label: str, line_width: int = LINE_WIDTH) -> str:
    lines = []
    lines.append(_section_header(label))

    total_len = len(seq)

    lines.append(
        f"  {_BOLD}{_BG_RED} B {_RESET} = Binding residue   "
        f"{_DIM}[ ]= Non-binding residue\n"
    )

    for start in range(0, total_len, line_width):
        end = min(start + line_width, total_len)
        chunk = seq[start:end]
        preds = pred[start:end]

        pos_label = f"{start + 1:>5}  "
        lines.append(f"{_DIM}{pos_label}{_RESET}")

        residue_str = ""
        for aa, p in zip(chunk, preds):
            if p == 1:
                residue_str += f"{_BOLD}{_BG_RED}{_WHITE}{aa}{_RESET}"
            else:
                residue_str += f"{_DIM}{aa}{_RESET}"

        lines.append(residue_str + f"  {_DIM}{end}{_RESET}\n")

        pred_row = ""
        for p in preds:
            pred_row += f"{_BOLD}{_RED}*{_RESET}" if p == 1 else f"{_DIM}.{_RESET}"
        lines.append(f"       {pred_row}\n")

    stats = _binding_stats(seq, pred)

    lines.append(
        f"\n  {_BOLD}Summary:{_RESET}\n"
        f"    Total residues    : {_WHITE}{stats['total']}{_RESET}\n"
        f"    Binding sites     : {_BOLD}{_RED}{stats['n_binding']}{_RESET} "
        f"({stats['pct_binding']:.1f}%)\n"
        f"    Non-binding sites : {_DIM}{stats['n_nonbinding']}{_RESET}\n"
        f"    Binding residues  : {_BOLD}{_RED}"
        + "".join(stats["binding_residues"]) + f"{_RESET}\n"
        f"    Unique binding AA : {_YELLOW}"
        + ", ".join(stats["unique_binding_aa"]) + f"{_RESET}\n"
    )

    return "".join(lines)


def render_terminal(
    protein_seq: str,
    peptide_seq: str,
    prot_pred: list,
    pep_pred: list,
    mode: str,
    embedding_model: str,
    dataset: str,
    metrics: dict = None,
) -> None:
    title_bar = "=" * LINE_WIDTH

    print(f"\n{_BOLD}{_CYAN}{title_bar}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'ProPepX — Binding Site Prediction':^{LINE_WIDTH}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{title_bar}{_RESET}\n")

    print(f"  {_BOLD}Mode          :{_RESET}  {_GREEN}{mode}{_RESET}")
    print(f"  {_BOLD}Embedding     :{_RESET}  {_GREEN}{embedding_model.upper()}{_RESET}")
    print(f"  {_BOLD}Dataset       :{_RESET}  {_GREEN}{dataset.upper()}{_RESET}")
    print(f"  {_BOLD}Threshold     :{_RESET}  0.50\n")

    if prot_pred is not None:
        print(_render_sequence_terminal(protein_seq, prot_pred, "PROTEIN  BINDING  SITES"))

    if pep_pred is not None:
        print(_render_sequence_terminal(peptide_seq, pep_pred, "PEPTIDE  BINDING  SITES"))

    if metrics:
        print(_section_header("PERFORMANCE  METRICS"))
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"    {_BOLD}{key:<25}{_RESET} {_WHITE}{val:.4f}{_RESET}")
            else:
                print(f"    {_BOLD}{key:<25}{_RESET} {_WHITE}{val}{_RESET}")

    print(f"\n{_BOLD}{_CYAN}{title_bar}{_RESET}\n")


_HTML_CSS = """
<style>
  :root {
    --bind-bg: #FEB2B2;
    --bind-fg: #742A2A;
    --nobind-bg: #F7FAFC;
    --nobind-fg: #4A5568;
    --ruler-fg: #A0AEC0;
    --header-bg: #2D3748;
    --header-fg: #E2E8F0;
    --card-bg: #FFFFFF;
    --border: #E2E8F0;
    --metric-bg: #EBF8FF;
    --metric-fg: #2B6CB0;
    --font-mono: 'JetBrains Mono', 'Fira Mono', 'Cascadia Code', monospace;
    --font-sans: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }

  body { font-family: var(--font-sans); background: #F7FAFC; margin: 0; padding: 24px; }

  .propepx-container {
    max-width: 1100px; margin: 0 auto;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,.08);
    overflow: hidden;
  }

  .propepx-header {
    background: var(--header-bg);
    color: var(--header-fg);
    padding: 20px 28px;
    display: flex; align-items: center; gap: 14px;
  }

  .propepx-header h1 { margin: 0; font-size: 1.4rem; font-weight: 700; letter-spacing: .5px; }

  .propepx-header .badge {
    background: #E53E3E; color: #fff;
    padding: 2px 10px; border-radius: 9999px;
    font-size: .75rem; font-weight: 600;
  }

  .propepx-meta {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1px; background: var(--border);
    border-bottom: 1px solid var(--border);
  }

  .propepx-meta-cell { background: #F7FAFC; padding: 10px 20px; }

  .propepx-meta-cell .label {
    font-size: .7rem; color: #718096;
    text-transform: uppercase; letter-spacing: .05em;
  }

  .propepx-meta-cell .value {
    font-size: .95rem; font-weight: 600; color: #2D3748; margin-top: 2px;
  }

  .propepx-section { padding: 24px 28px; border-bottom: 1px solid var(--border); }

  .propepx-section h2 {
    margin: 0 0 16px; font-size: 1rem; font-weight: 700;
    color: #2D3748; text-transform: uppercase; letter-spacing: .08em;
    display: flex; align-items: center; gap: 8px;
  }

  .propepx-section h2::before {
    content: ''; display: inline-block;
    width: 4px; height: 18px; border-radius: 2px;
    background: #E53E3E;
  }

  .seq-block {
    font-family: var(--font-mono);
    font-size: .85rem;
    line-height: 1.9;
    background: #FAFAFA;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    overflow-x: auto;
    white-space: pre;
  }

  .ruler { color: var(--ruler-fg); font-size: .75rem; }
  .pred-row { font-size: .7rem; letter-spacing: .1em; }

  .b {
    background: var(--bind-bg);
    color: var(--bind-fg);
    font-weight: 800;
    border-radius: 3px;
    padding: 0 1px;
  }

  .nb { color: var(--nobind-fg); }

  .legend {
    display: flex; gap: 20px; margin-bottom: 12px;
    font-size: .8rem; color: #4A5568;
  }

  .legend-item { display: flex; align-items: center; gap: 6px; }

  .legend-swatch {
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid rgba(0,0,0,.1);
  }

  .stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin-top: 16px;
  }

  .stat-card {
    background: var(--metric-bg);
    border: 1px solid #BEE3F8;
    border-radius: 8px;
    padding: 12px 16px;
  }

  .stat-card .s-label {
    font-size: .7rem; color: #2B6CB0;
    text-transform: uppercase; letter-spacing: .05em;
  }

  .stat-card .s-value {
    font-size: 1.25rem; font-weight: 700; color: var(--metric-fg);
  }

  .binding-residues {
    font-family: var(--font-mono);
    font-size: .85rem;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 12px;
    line-height: 1.7;
  }

  .br-chip {
    background: var(--bind-bg);
    color: var(--bind-fg);
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: 700;
    font-size: .82rem;
  }


  .details-btn {
    margin-top: 18px;
    background: #2B6CB0;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 9px 16px;
    font-size: .82rem;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(43,108,176,.18);
  }

  .details-btn:hover {
    background: #2C5282;
  }

  .details-panel {
    margin-top: 14px;
  }

  .details-title {
    margin-top: 6px;
    margin-bottom: 10px;
    font-size: .95rem;
    color: #2D3748;
    font-weight: 700;
    letter-spacing: .03em;
  }

  .details-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .85rem;
    margin-top: 6px;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .details-table th {
    background: #EDF2F7;
    color: #4A5568;
    text-align: left;
    padding: 9px 14px;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    border-bottom: 2px solid var(--border);
  }

  .details-table td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    color: #2D3748;
  }

  .details-table tr:last-child td {
    border-bottom: none;
  }

  .details-table .aa3 {
    font-weight: 700;
    color: #742A2A;
  }

  .details-table .aa1 {
    font-family: var(--font-mono);
    font-weight: 800;
    background: var(--bind-bg);
    color: var(--bind-fg);
    border-radius: 4px;
    padding: 2px 7px;
    display: inline-block;
  }

  .details-table .prob {
    font-family: var(--font-mono);
    font-weight: 700;
    color: #2B6CB0;
  }

  .metrics-table {
    width: 100%; border-collapse: collapse; font-size: .875rem; margin-top: 4px;
  }

  .metrics-table th {
    background: #EDF2F7; color: #4A5568;
    text-align: left; padding: 8px 14px;
    font-size: .75rem; text-transform: uppercase; letter-spacing: .05em;
    border-bottom: 2px solid var(--border);
  }

  .metrics-table td {
    padding: 8px 14px; border-bottom: 1px solid var(--border); color: #2D3748;
  }

  .metrics-table tr:last-child td { border-bottom: none; }
  .metrics-table .mv { font-weight: 700; color: #2B6CB0; }

  .propepx-footer {
    background: #F7FAFC;
    padding: 14px 28px;
    font-size: .75rem;
    color: #A0AEC0;
    display: flex;
    justify-content: space-between;
  }
</style>
"""


def _seq_block_html(seq: str, pred: list, line_width: int = HTML_LINE_WIDTH) -> str:
    chunks = []
    total = len(seq)

    for start in range(0, total, line_width):
        end = min(start + line_width, total)
        chunk = seq[start:end]
        ps = pred[start:end]

        ruler = f'<span class="ruler">{str(start + 1).rjust(5)}  </span>'

        res_html = ""
        for aa, p in zip(chunk, ps):
            css = "b" if p == 1 else "nb"
            res_html += f'<span class="{css}">{aa}</span>'

        end_num = f'<span class="ruler">  {end}</span>'

        pred_html = '<span class="pred-row">       '
        for p in ps:
            pred_html += '<span class="b">*</span>' if p == 1 else '<span class="nb">.</span>'
        pred_html += "</span>"

        chunks.append(f"{ruler}{res_html}{end_num}\n{pred_html}\n")

    return '<div class="seq-block">' + "\n".join(chunks) + "</div>"


def _stats_html(seq: str, pred: list) -> str:
    stats = _binding_stats(seq, pred)

    cards = [
        ("Total Residues", stats["total"]),
        ("Binding Sites", f"{stats['n_binding']} ({stats['pct_binding']:.1f}%)"),
        ("Non-Binding Sites", stats["n_nonbinding"]),
    ]

    cards_html = "\n".join(
        f'<div class="stat-card">'
        f'<div class="s-label">{label}</div>'
        f'<div class="s-value">{value}</div>'
        f'</div>'
        for label, value in cards
    )

    binding_text = _binding_residues_inline_html(
        seq=seq,
        pred=pred,
        title="Binding residues",
    )

    return f'<div class="stats-grid">{cards_html}</div>{binding_text}'


def _binding_residues_inline_html(seq: str, pred: list, title: str = "Binding residues") -> str:
    """Show binding residues as chips WITHOUT probability, e.g. Lys34."""
    items = []

    for i, p in enumerate(pred):
        if p == 1:
            aa1 = seq[i]
            aa3 = AA3.get(aa1, aa1)
            pos = i + 1
            label = f"{aa3}{pos}"
            items.append(f'<span class="br-chip">{label}</span>')

    chips = "".join(items) if items else "<em>None</em>"

    return (
        f'<div style="margin-top:14px;font-size:.8rem;color:#4A5568;font-weight:600;">'
        f'{title}</div>'
        f'<div class="binding-residues">{chips}</div>'
    )


def _binding_residue_table_html(
    seq: str,
    pred: list,
    prob=None,
    title: str = "Binding Residue Details",
    details_id: str = "binding-details",
) -> str:
    rows = []

    for i, p in enumerate(pred):
        if p == 1:
            aa1 = seq[i]
            aa3 = AA3.get(aa1, aa1)
            position = i + 1

            if prob is not None:
                try:
                    probability = float(prob[i])
                    prob_text = f"{probability:.4f}"
                except Exception:
                    prob_text = "NA"
            else:
                prob_text = "NA"

            rows.append(
                "<tr>"
                f"<td>{position}</td>"
                f'<td class="aa3">{aa3}</td>'
                f'<td><span class="aa1">{aa1}</span></td>'
                f'<td class="prob">{prob_text}</td>'
                "</tr>"
            )

    if not rows:
        rows.append(
            "<tr><td colspan='4'><em>No binding residues predicted.</em></td></tr>"
        )

    return f"""
    <button class="details-btn" type="button" onclick="toggleBindingDetails('{details_id}', this)">
      Show {title}
    </button>

    <div id="{details_id}" class="details-panel" style="display:none;">
      <div class="details-title">{title}</div>
      <table class="details-table">
        <thead>
          <tr>
            <th>Position</th>
            <th>3-letter code</th>
            <th>1-letter code</th>
            <th>ProPepX probability</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


def _details_toggle_script() -> str:
    return """
<script>
  function toggleBindingDetails(detailsId, btn) {
    const panel = document.getElementById(detailsId);
    if (!panel) return;

    const isHidden = panel.style.display === "none" || panel.style.display === "";
    panel.style.display = isHidden ? "block" : "none";
    btn.textContent = (isHidden ? "Hide " : "Show ") + btn.textContent.replace(/^Show |^Hide /, "");
  }
</script>
"""


def _metrics_html(metrics: dict) -> str:
    if not metrics:
        return ""

    rows = ""
    for k, v in metrics.items():
        value = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows += f'<tr><td>{k}</td><td class="mv">{value}</td></tr>'

    return (
        '<table class="metrics-table">'
        '<thead><tr><th>Metric</th><th>Value</th></tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table>'
    )


def render_html(
    protein_seq: str,
    peptide_seq: str,
    prot_pred: list,
    pep_pred: list,
    mode: str,
    embedding_model: str,
    dataset: str,
    metrics: dict = None,
    output_path: str = None,
    prot_prob=None,
    pep_prob=None,
) -> str:
    import datetime

    meta_cells = "".join([
        f'<div class="propepx-meta-cell"><div class="label">Mode</div>'
        f'<div class="value">{mode}</div></div>',
        f'<div class="propepx-meta-cell"><div class="label">Embedding</div>'
        f'<div class="value">{embedding_model.upper()}</div></div>',
        f'<div class="propepx-meta-cell"><div class="label">Dataset</div>'
        f'<div class="value">{dataset.upper()}</div></div>',
        f'<div class="propepx-meta-cell"><div class="label">Threshold</div>'
        f'<div class="value">0.50</div></div>',
        f'<div class="propepx-meta-cell"><div class="label">Protein length</div>'
        f'<div class="value">{len(protein_seq)} aa</div></div>',
        f'<div class="propepx-meta-cell"><div class="label">Peptide length</div>'
        f'<div class="value">{len(peptide_seq)} aa</div></div>',
    ])

    legend = (
        '<div class="legend">'
        '<div class="legend-item">'
        '<div class="legend-swatch" style="background:#FEB2B2;"></div>'
        'Binding residue</div>'
        '<div class="legend-item">'
        '<div class="legend-swatch" style="background:#F7FAFC;border-color:#CBD5E0;"></div>'
        'Non-binding residue</div>'
        '</div>'
    )

    sections = []

    if prot_pred is not None:
        sections.append(
            f'<div class="propepx-section">'
            f'<h2>Protein Binding Sites</h2>'
            f'{legend}'
            f'{_seq_block_html(protein_seq, prot_pred)}'
            f'{_stats_html(protein_seq, prot_pred)}'
            f'{_binding_residue_table_html(protein_seq, prot_pred, prot_prob, "Protein Binding Residue Details", "protein-binding-details")}'
            f'</div>'
        )

    if pep_pred is not None:
        sections.append(
            f'<div class="propepx-section">'
            f'<h2>Peptide Binding Sites</h2>'
            f'{legend}'
            f'{_seq_block_html(peptide_seq, pep_pred)}'
            f'{_stats_html(peptide_seq, pep_pred)}'
            f'{_binding_residue_table_html(peptide_seq, pep_pred, pep_prob, "Peptide Binding Residue Details", "peptide-binding-details")}'
            f'</div>'
        )

    if metrics:
        sections.append(
            f'<div class="propepx-section">'
            f'<h2>Performance Metrics</h2>'
            f'{_metrics_html(metrics)}'
            f'</div>'
        )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ProPepX — Binding Site Prediction</title>
  {_HTML_CSS}
  {_details_toggle_script()}
</head>
<body>
  <div class="propepx-container">

    <div class="propepx-header">
      <div>
        <h1>ProPepX</h1>
        <div style="font-size:.8rem;color:#A0AEC0;margin-top:2px;">
          Protein–Peptide Binding Site Predictor
        </div>
      </div>
      <span class="badge">PREDICTION REPORT</span>
    </div>

    <div class="propepx-meta">{meta_cells}</div>

    {"".join(sections)}

    <div class="propepx-footer">
      <span>ProPepX · Syed Kumail Hussain Naqvi et al.</span>
      <span>Generated: {timestamp}</span>
    </div>

  </div>
</body>
</html>"""

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  📄  HTML report saved → {output_path}")

    return html