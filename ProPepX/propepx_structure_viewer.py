"""
propepx_structure_viewer.py
============================
Nature-style ProPepX 3D Protein–Peptide Binding Site Viewer

Default style:
- White background
- Protein cartoon/ribbon: teal (#2A9D8F)
- Peptide cartoon/ribbon: orange (#C96A10)
- Protein binding residues: red sticks + spheres (#007FFF)
- Peptide binding residues: yellow/orange sticks + spheres (#D11D80)
- Publication-friendly modes:
    1. AI Interface View
    2. Interaction Surface
    3. Binding Hotspots
    4. Pocket Focus
    5. Interface Contacts
    6. Publication Figure

IMPORTANT:
For real ProPepX predictions, do NOT run this file directly for final results.
Call generate_html_3d_viewer() from propepx_predict.py and pass:
    infer["prot_pred"], infer["pep_pred"], infer["prot_prob"], infer["pep_prob"]
    
**************************************************************************************************************
"""

import os
import sys
import json
import base64
import datetime
import subprocess
import textwrap


AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}


# ==========================================================
# Stage A: structure prediction helpers
# ==========================================================

def prepare_colabfold_fasta(
    protein_seq: str,
    peptide_seq: str,
    output_fasta: str,
    job_name: str = "propepx_complex",
) -> str:
    """
    Write paired FASTA for ColabFold multimer prediction.

    Format:
        >propepx_complex
        PROTEIN_SEQUENCE:PEPTIDE_SEQUENCE
    """
    output_fasta = os.path.abspath(output_fasta)
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

    protein_seq = protein_seq.strip().replace(" ", "").replace("\n", "")
    peptide_seq = peptide_seq.strip().replace(" ", "").replace("\n", "")

    with open(output_fasta, "w", encoding="utf-8") as fh:
        fh.write(f">{job_name}\n{protein_seq}:{peptide_seq}\n")

    print(f"  FASTA written -> {output_fasta}")
    print(f"  Protein : {len(protein_seq)} aa")
    print(f"  Peptide : {len(peptide_seq)} aa")

    return output_fasta


def run_colabfold_prediction(
    fasta_path: str,
    output_dir: str,
    num_models: int = 5,
    num_recycles: int = 3,
    use_amber: bool = False,
    use_templates: bool = False,
    model_type: str = "alphafold2_multimer_v3",
    colabfold_bin: str = "colabfold_batch",
    conda_env: str = None,
) -> str:
    """
    Run ColabFold and return best PDB path.
    Recommended: keep ColabFold in a separate conda env and set conda_env="colabfold".
    """
    os.makedirs(output_dir, exist_ok=True)

    base_cmd = [
        colabfold_bin,
        fasta_path,
        output_dir,
        "--model-type", model_type,
        "--num-models", str(num_models),
        "--num-recycle", str(num_recycles),
    ]

    if conda_env:
        cmd = ["conda", "run", "-n", conda_env] + base_cmd
    else:
        cmd = base_cmd

    if use_templates:
        cmd.append("--templates")
    if use_amber:
        cmd.append("--amber")

    print("\n" + "=" * 70)
    print("  ColabFold / AlphaFold2-Multimer Structure Prediction")
    print("=" * 70)
    print("  Command:")
    print("  " + " ".join(cmd) + "\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ColabFold exited with code {result.returncode}.")

    candidates = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".pdb") and ("rank_001" in f or "rank_1" in f or "ranked_0" in f)
    ])

    if not candidates:
        candidates = sorted([
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".pdb") and "rank" in f
        ])

    if not candidates:
        raise FileNotFoundError(f"No ranked PDB file found in {output_dir}")

    print(f"  Best structure -> {candidates[0]}")
    return candidates[0]


def print_alphafold_server_guide(protein_seq: str = None, peptide_seq: str = None) -> None:
    guide = """
AlphaFold Server / ColabFold workflow
=====================================

1. Open AlphaFold Server.
2. Add Entity 1 as Protein and paste the protein sequence.
3. Add Entity 2 as Protein and paste the peptide sequence.
4. Run prediction.
5. Download model_0.cif or model_0.pdb.
6. Pass that file to generate_html_3d_viewer().
"""
    print(textwrap.dedent(guide))

    if protein_seq:
        print("\nProtein sequence:")
        print(protein_seq)

    if peptide_seq:
        print("\nPeptide sequence:")
        print(peptide_seq)


# ==========================================================
# Internal helpers
# ==========================================================

def _clean_seq(seq: str) -> str:
    return seq.strip().replace(" ", "").replace("\n", "")


def _validate_predictions(seq: str, pred, name: str):
    if pred is None:
        return None

    pred = list(pred)

    if len(pred) != len(seq):
        print(
            f"  WARNING: {name} prediction length ({len(pred)}) does not match "
            f"{name} sequence length ({len(seq)}). Truncating to shortest length."
        )
        n = min(len(pred), len(seq))
        pred = pred[:n]

    return pred


def _resi_list(pred):
    if pred is None:
        return []
    return [i + 1 for i, p in enumerate(pred) if int(p) == 1]


def _prob_map(pred, prob):
    if pred is None:
        return {}

    out = {}
    for i, p in enumerate(pred):
        if int(p) != 1:
            continue

        if prob is None:
            out[str(i + 1)] = None
            continue

        try:
            out[str(i + 1)] = round(float(prob[i]), 4)
        except Exception:
            out[str(i + 1)] = None

    return out


def _build_binding_site_js(
    prot_pred,
    pep_pred,
    prot_prob,
    pep_prob,
    prot_chain_id: str,
    pep_chain_id: str,
) -> str:
    prot_resi = _resi_list(prot_pred)
    pep_resi = _resi_list(pep_pred)
    prot_prob_map = _prob_map(prot_pred, prot_prob)
    pep_prob_map = _prob_map(pep_pred, pep_prob)

    return f"""
const PROT_CHAIN = "{prot_chain_id}";
const PEP_CHAIN = "{pep_chain_id}";
const PROT_BINDING_RESI = {json.dumps(prot_resi)};
const PEP_BINDING_RESI = {json.dumps(pep_resi)};
const PROT_PROB_MAP = {json.dumps(prot_prob_map)};
const PEP_PROB_MAP = {json.dumps(pep_prob_map)};
"""


def _build_binding_table_html(
    protein_seq: str,
    peptide_seq: str,
    prot_pred,
    pep_pred,
    prot_prob,
    pep_prob,
    prot_chain_id: str,
    pep_chain_id: str,
) -> str:
    def rows(seq, pred, prob, chain, cls):
        html_rows = []
        if pred is None:
            return html_rows

        for i, p in enumerate(pred):
            if int(p) != 1:
                continue

            aa1 = seq[i] if i < len(seq) else "X"
            aa3 = AA3.get(aa1, aa1)
            pos = i + 1

            if prob is not None:
                try:
                    pv = float(prob[i])
                    prob_text = f"{pv:.4f}"
                    pct = max(0, min(100, int(pv * 100)))
                    bar = (
                        f'<div class="tb-bar-wrap">'
                        f'<div class="tb-bar {cls}" style="width:{pct}%"></div>'
                        f'</div>'
                    )
                except Exception:
                    prob_text = "N/A"
                    bar = ""
            else:
                prob_text = "N/A"
                bar = ""

            html_rows.append(
                f"<tr class='seq-row' data-chain='{chain}' data-resi='{pos}'>"
                f"<td><span class='chain-badge {cls}'>{chain}</span></td>"
                f"<td>{pos}</td>"
                f"<td class='aa3'>{aa3}</td>"
                f"<td><span class='aa1-sm'>{aa1}</span></td>"
                f"<td>{prob_text}{bar}</td>"
                "</tr>"
            )

        return html_rows

    all_rows = []
    all_rows += rows(protein_seq, prot_pred, prot_prob, prot_chain_id, "prot-cls")
    all_rows += rows(peptide_seq, pep_pred, pep_prob, pep_chain_id, "pep-cls")

    if not all_rows:
        all_rows = ["<tr><td colspan='5'><em>No binding residues predicted.</em></td></tr>"]

    return f"""
<div class="panel-title">Predicted binding residues</div>
<div class="table-wrap">
  <table class="bind-table">
    <thead>
      <tr>
        <th>Chain</th>
        <th>Pos</th>
        <th>AA3</th>
        <th>AA1</th>
        <th>ProPepX probability</th>
      </tr>
    </thead>
    <tbody>
      {''.join(all_rows)}
    </tbody>
  </table>
</div>
"""


_VIEWER_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    height: 100%;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background: #ffffff;
  }

  .app-wrap {
    display: flex;
    height: 100vh;
    overflow: hidden;
    background: #ffffff;
  }

  .left-panel {
    width: 320px;
    flex-shrink: 0;
    background: #f8fafc;
    border-right: 1px solid #d8dee9;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .panel-header {
    background: #ffffff;
    padding: 18px 20px;
    border-bottom: 1px solid #d8dee9;
  }

  .panel-header h1 {
    font-size: 1.2rem;
    font-weight: 900;
    color: #102a43;
    letter-spacing: .3px;
  }

  .panel-header .sub {
    font-size: .75rem;
    color: #52616b;
    margin-top: 4px;
  }

  .panel-meta {
    padding: 14px 20px;
    border-bottom: 1px solid #d8dee9;
    background: #f1f5f9;
  }

  .meta-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 7px;
    gap: 12px;
  }

  .meta-label {
    font-size: .68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .06em;
  }

  .meta-value {
    font-size: .74rem;
    font-weight: 850;
    color: #0f172a;
    text-align: right;
  }

  .controls {
    padding: 14px 20px;
    border-bottom: 1px solid #d8dee9;
    background: #ffffff;
  }

  .controls-title,
  .panel-title {
    font-size: .68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .09em;
    margin-bottom: 10px;
    font-weight: 800;
  }

  .btn-group {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .ctrl-btn {
    background: #eef2f7;
    color: #243b53;
    border: 1px solid #d8dee9;
    border-radius: 9px;
    padding: 9px 11px;
    font-size: .78rem;
    font-weight: 800;
    cursor: pointer;
    text-align: left;
    transition: all .16s ease;
  }

  .ctrl-btn:hover {
    background: #e2e8f0;
    transform: translateY(-1px);
  }

  .ctrl-btn.active {
    background: #2A9D8F;
    color: #ffffff;
    border-color: #2A9D8F;
    box-shadow: 0 8px 18px rgba(42,157,143,.18);
  }

  .legend {
    padding: 14px 20px;
    border-bottom: 1px solid #d8dee9;
    background: #ffffff;
  }

  .leg-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: .74rem;
    color: #334e68;
  }

  .leg-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    border: 1px solid rgba(0,0,0,.08);
  }

  .panel-title {
    padding: 14px 20px 6px;
    margin-bottom: 0;
    background: #f8fafc;
  }

  .table-wrap {
    flex: 1;
    overflow: auto;
    padding: 0 12px 18px;
    background: #f8fafc;
  }

  .bind-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .73rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
  }

  .bind-table th {
    background: #f1f5f9;
    color: #52616b;
    text-align: left;
    padding: 8px 7px;
    font-size: .62rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    position: sticky;
    top: 0;
    border-bottom: 1px solid #d8dee9;
  }

  .bind-table td {
    padding: 8px 7px;
    border-bottom: 1px solid #edf2f7;
    color: #102a43;
  }

  .bind-table tr:hover td,
  .bind-table tr.active-row td {
    background: #fff5f5;
  }

  .chain-badge {
    border-radius: 4px;
    padding: 1px 7px;
    font-weight: 900;
    font-size: .7rem;
  }

  .prot-cls {
    background: #2A9D8F;
    color: #ffffff;
  }

  .pep-cls {
    background: #C96A10;
    color: #ffffff;
  }

  .aa3 {
    font-weight: 850;
    color: #007FFF;
  }

  .aa1-sm {
    font-family: 'JetBrains Mono', 'Fira Mono', monospace;
    font-size: .8rem;
    background: #e2e8f0;
    border-radius: 4px;
    padding: 1px 7px;
    color: #102a43;
    font-weight: 900;
  }

  .tb-bar-wrap {
    background: #e2e8f0;
    border-radius: 3px;
    height: 5px;
    margin-top: 4px;
  }

  .tb-bar {
    height: 5px;
    border-radius: 3px;
  }

  .tb-bar.prot-cls {
    background: #007FFF;
  }

  .tb-bar.pep-cls {
    background: #D11D80;
  }

  .viewport-wrap {
    flex: 1;
    position: relative;
    background: #ffffff;
  }

  #ngl-viewport {
    width: 100%;
    height: 100%;
    background: #ffffff;
  }

  .top-bar {
    position: absolute;
    top: 14px;
    right: 16px;
    z-index: 10;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
  }

  .top-bar button {
    background: rgba(255,255,255,.92);
    color: #102a43;
    border: 1px solid #d8dee9;
    border-radius: 8px;
    padding: 7px 12px;
    font-size: .73rem;
    font-weight: 850;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(15,23,42,.08);
  }

  .top-bar button:hover {
    background: #f1f5f9;
  }

  #ngl-tooltip {
    position: absolute;
    pointer-events: none;
    display: none;
    background: #ffffff;
    border: 1px solid #d8dee9;
    border-radius: 9px;
    padding: 9px 13px;
    font-size: .76rem;
    color: #102a43;
    box-shadow: 0 10px 28px rgba(15,23,42,.16);
    z-index: 100;
    min-width: 180px;
  }

  #ngl-tooltip strong {
    display: block;
    font-size: .86rem;
    color: #007FFF;
    margin-bottom: 4px;
  }

  #ngl-tooltip .prob-line {
    color: #2A9D8F;
    margin-top: 4px;
    font-weight: 800;
  }

  .viewer-footer {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255,255,255,.92);
    border: 1px solid #d8dee9;
    border-radius: 999px;
    padding: 6px 18px;
    font-size: .68rem;
    color: #52616b;
    white-space: nowrap;
    z-index: 10;
    box-shadow: 0 4px 16px rgba(15,23,42,.08);
  }

  .sequence-strip {
    position: absolute;
    left: 340px;
    right: 20px;
    bottom: 48px;
    z-index: 10;
    background: rgba(255,255,255,.94);
    border: 1px solid #d8dee9;
    border-radius: 12px;
    padding: 8px 12px;
    font-family: 'JetBrains Mono', 'Fira Mono', monospace;
    font-size: .72rem;
    box-shadow: 0 4px 16px rgba(15,23,42,.08);
    overflow-x: auto;
    white-space: nowrap;
  }

  .seq-chip {
    display: inline-block;
    padding: 2px 5px;
    margin: 1px;
    border-radius: 5px;
    cursor: pointer;
    color: #334e68;
  }

  .seq-chip.bind-prot {
    background: #ffd7d7;
    color: #8a1c1c;
    font-weight: 900;
  }

  .seq-chip.bind-pep {
    background: #ffe6c7;
    color: #7a3b00;
    font-weight: 900;
  }

  .seq-chip:hover,
  .seq-chip.active-chip {
    outline: 2px solid #007FFF;
  }
</style>
"""


def generate_html_3d_viewer(
    pdb_path: str,
    protein_seq: str,
    peptide_seq: str,
    prot_pred,
    pep_pred,
    prot_prob=None,
    pep_prob=None,
    prot_chain_id: str = "A",
    pep_chain_id: str = "B",
    output_html: str = None,
    mode: str = "mode-global",
    embedding_model: str = "prottrans",
    dataset: str = "test167",
    title: str = "ProPepX — Nature-style 3D Binding Site Viewer",
) -> str:
    """
    Generate Nature-style interactive NGL viewer HTML.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB/mmCIF file not found: {pdb_path}")

    protein_seq = _clean_seq(protein_seq)
    peptide_seq = _clean_seq(peptide_seq)

    prot_pred = _validate_predictions(protein_seq, prot_pred, "protein")
    pep_pred = _validate_predictions(peptide_seq, pep_pred, "peptide")

    with open(pdb_path, "rb") as fh:
        pdb_b64 = base64.b64encode(fh.read()).decode("ascii")

    fmt = "cif" if pdb_path.lower().endswith((".cif", ".mmcif")) else "pdb"

    js_data = _build_binding_site_js(
        prot_pred=prot_pred,
        pep_pred=pep_pred,
        prot_prob=prot_prob,
        pep_prob=pep_prob,
        prot_chain_id=prot_chain_id,
        pep_chain_id=pep_chain_id,
    )

    binding_table_html = _build_binding_table_html(
        protein_seq=protein_seq,
        peptide_seq=peptide_seq,
        prot_pred=prot_pred,
        pep_pred=pep_pred,
        prot_prob=prot_prob,
        pep_prob=pep_prob,
        prot_chain_id=prot_chain_id,
        pep_chain_id=pep_chain_id,
    )

    n_prot_bind = sum(int(x) for x in prot_pred) if prot_pred is not None else 0
    n_pep_bind = sum(int(x) for x in pep_pred) if pep_pred is not None else 0

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    structure_name = os.path.basename(pdb_path)

    prot_seq_html = "".join(
        f"<span class='seq-chip {'bind-prot' if prot_pred and int(prot_pred[i]) == 1 else ''}' "
        f"data-chain='{prot_chain_id}' data-resi='{i+1}'>{aa}</span>"
        for i, aa in enumerate(protein_seq)
    )
    pep_seq_html = "".join(
        f"<span class='seq-chip {'bind-pep' if pep_pred and int(pep_pred[i]) == 1 else ''}' "
        f"data-chain='{pep_chain_id}' data-resi='{i+1}'>{aa}</span>"
        for i, aa in enumerate(peptide_seq)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {_VIEWER_CSS}
  <script src="https://cdnjs.cloudflare.com/ajax/libs/ngl/2.0.0-dev.37/ngl.js"></script>
</head>

<body>
<div class="app-wrap">

  <div class="left-panel">

    <div class="panel-header">
      <h1>ProPepX</h1>
      <div class="sub">AI-guided protein–peptide interface viewer</div>
    </div>

    <div class="panel-meta">
      <div class="meta-row">
        <span class="meta-label">Mode</span>
        <span class="meta-value">{mode}</span>
      </div>
      <div class="meta-row">
        <span class="meta-label">Embedding</span>
        <span class="meta-value">{embedding_model.upper()}</span>
      </div>
      <div class="meta-row">
        <span class="meta-label">Dataset</span>
        <span class="meta-value">{dataset.upper()}</span>
      </div>
      <div class="meta-row">
        <span class="meta-label">Protein binding</span>
        <span class="meta-value" style="color:#007FFF;">{n_prot_bind} / {len(prot_pred) if prot_pred else 0}</span>
      </div>
      <div class="meta-row">
        <span class="meta-label">Peptide binding</span>
        <span class="meta-value" style="color:#D11D80;">{n_pep_bind} / {len(pep_pred) if pep_pred else 0}</span>
      </div>
    </div>

    <div class="controls">
      <div class="controls-title">Representation</div>
      <div class="btn-group">
        <button class="ctrl-btn active" id="btn-interface" onclick="setRepr('interface')"> ProPepX Interface View</button>
        <button class="ctrl-btn" id="btn-surface" onclick="setRepr('surface')">Interaction Surface</button>
        <button class="ctrl-btn" id="btn-hotspots" onclick="setRepr('hotspots')"> Binding Hotspots</button>
        <button class="ctrl-btn" id="btn-pocket" onclick="setRepr('pocket')"> Pocket Focus</button>
        <button class="ctrl-btn" id="btn-contacts" onclick="setRepr('contacts')"> Interface Contacts</button>
        <button class="ctrl-btn" id="btn-pub" onclick="setRepr('publication')"> Publication Figure</button>
      </div>
    </div>

    <div class="legend">
      <div class="leg-item">
        <div class="leg-dot" style="background:#2A9D8F;"></div>
        Protein chain {prot_chain_id} — cartoon
      </div>
      <div class="leg-item">
        <div class="leg-dot" style="background:#C96A10;"></div>
        Peptide chain {pep_chain_id} — cartoon
      </div>
      <div class="leg-item">
        <div class="leg-dot" style="background:#007FFF;"></div>
        Protein ProPepX binding residue
      </div>
      <div class="leg-item">
        <div class="leg-dot" style="background:#D11D80;"></div>
        Peptide ProPepX binding residue
      </div>
    </div>

    {binding_table_html}

  </div>

  <div class="viewport-wrap">
    <div class="top-bar">
      <button onclick="stage.autoView()">Reset</button>
      <button onclick="zoomBinding()">Zoom interface</button>
      <button onclick="toggleSpin()">Rotate</button>
      <button onclick="saveImage()">Save PNG</button>
    </div>

    <div id="ngl-viewport"></div>
    <div id="ngl-tooltip"></div>

    <div class="sequence-strip">
      <b style="color:#2A9D8F;">Protein A:</b> {prot_seq_html}
      <br>
      <b style="color:#C96A10;">Peptide B:</b> {pep_seq_html}
    </div>

    <div class="viewer-footer">
      ProPepX · {timestamp} · NGL Viewer · {structure_name}
    </div>
  </div>

</div>

<script>
const PDB_B64 = "{pdb_b64}";
const PDB_FMT = "{fmt}";

{js_data}

const COLORS = {{
  protein: "#2A9D8F",
  peptide: "#C96A10",
  protBind: "#007FFF",
  pepBind: "#D11D80",
  surface: "#D9D9D9",
  gray: "#E5E7EB"
}};

function b64toBlob(b64, type) {{
  const bytes = atob(b64);
  const ab = new ArrayBuffer(bytes.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < bytes.length; i++) ia[i] = bytes.charCodeAt(i);
  return new Blob([ab], {{type: type}});
}}

let stage = new NGL.Stage("ngl-viewport", {{
  backgroundColor: "white",
  clipNear: 0,
  clipFar: 100,
  clipDist: 10,
  fogNear: 100,
  fogFar: 100
}});

stage.setParameters({{ backgroundColor: "white" }});

window.addEventListener("resize", function() {{
  stage.handleResize();
}});

let compRef = null;
let currentRepr = "interface";
let spinning = false;
let hoverRepr = null;

function makeSel(resList, chain) {{
  if (!resList || resList.length === 0) return "none";
  return "(" + resList.map(r => "(:" + chain + " and " + r + ")").join(" or ") + ")";
}}

function makeSingleSel(chain, resi) {{
  return "(:" + chain + " and " + resi + ")";
}}

function makeProtSel() {{
  return makeSel(PROT_BINDING_RESI, PROT_CHAIN);
}}

function makePepSel() {{
  return makeSel(PEP_BINDING_RESI, PEP_CHAIN);
}}

function makeAllBindingSel() {{
  const p = makeProtSel();
  const q = makePepSel();

  if (p === "none" && q === "none") return "none";
  if (p === "none") return q;
  if (q === "none") return p;

  return "(" + p + " or " + q + ")";
}}

function makePocketSel() {{
  return "(:" + PEP_CHAIN + " or (:" + PROT_CHAIN + " and within 6 of :" + PEP_CHAIN + "))";
}}

function baseCartoon(comp, proteinOpacity=1, peptideOpacity=1) {{
  comp.addRepresentation("cartoon", {{
    sele: ":" + PROT_CHAIN,
    color: COLORS.protein,
    opacity: proteinOpacity,
    smoothSheet: true,
    aspectRatio: 5.0,
    radius: 0.18
  }});

  comp.addRepresentation("cartoon", {{
    sele: ":" + PEP_CHAIN,
    color: COLORS.peptide,
    opacity: peptideOpacity,
    smoothSheet: true,
    aspectRatio: 5.0,
    radius: 0.20
  }});
}}

function bindingResidues(comp, sphereRadius=0.38, stickScale=0.45) {{
  if (PROT_BINDING_RESI.length > 0) {{
    comp.addRepresentation("licorice", {{
      sele: makeProtSel(),
      color: COLORS.protBind,
      multipleBond: true,
      radiusScale: stickScale
    }});

    comp.addRepresentation("spacefill", {{
      sele: makeProtSel(),
      color: COLORS.protBind,
      radius: sphereRadius,
      opacity: 1.0
    }});
  }}

  if (PEP_BINDING_RESI.length > 0) {{
    comp.addRepresentation("licorice", {{
      sele: makePepSel(),
      color: COLORS.pepBind,
      multipleBond: true,
      radiusScale: stickScale
    }});

    comp.addRepresentation("spacefill", {{
      sele: makePepSel(),
      color: COLORS.pepBind,
      radius: sphereRadius,
      opacity: 1.0
    }});
  }}
}}

function renderInterface(comp) {{
  comp.removeAllRepresentations();
  stage.setParameters({{ backgroundColor: "white" }});
  baseCartoon(comp, 1, 1);
  bindingResidues(comp, 0.35, 0.42);
}}

function renderSurface(comp) {{
  comp.removeAllRepresentations();
  stage.setParameters({{ backgroundColor: "white" }});

  comp.addRepresentation("surface", {{
    sele: ":" + PROT_CHAIN,
    color: COLORS.surface,
    opacity: 0.25,
    probeRadius: 1.4,
    smooth: 2
  }});

  baseCartoon(comp, 0.85, 1);
  bindingResidues(comp, 0.38, 0.45);
}}

function renderHotspots(comp) {{
  comp.removeAllRepresentations();
  stage.setParameters({{ backgroundColor: "white" }});

  comp.addRepresentation("cartoon", {{
    sele: ":" + PROT_CHAIN,
    color: COLORS.gray,
    opacity: 0.35,
    smoothSheet: true
  }});

  comp.addRepresentation("cartoon", {{
    sele: ":" + PEP_CHAIN,
    color: COLORS.peptide,
    opacity: 1,
    smoothSheet: true
  }});

  bindingResidues(comp, 0.48, 0.55);
  zoomBinding();
}}

function renderPocket(comp) {{
  comp.removeAllRepresentations();
  stage.setParameters({{ backgroundColor: "white" }});

  comp.addRepresentation("cartoon", {{
    sele: makePocketSel(),
    color: "chainid",
    opacity: 0.95,
    smoothSheet: true
  }});

  comp.addRepresentation("surface", {{
    sele: "(:" + PROT_CHAIN + " and within 6 of :" + PEP_CHAIN + ")",
    color: COLORS.surface,
    opacity: 0.22,
    probeRadius: 1.4
  }});

  bindingResidues(comp, 0.45, 0.55);
  zoomBinding();
}}

function renderContacts(comp) {{
  comp.removeAllRepresentations();
  stage.setParameters({{ backgroundColor: "white" }});

  baseCartoon(comp, 0.65, 1);
  bindingResidues(comp, 0.38, 0.45);

  // Contact-style dashed distance lines around the peptide interface.
  // This is visual guidance; exact hydrogen/salt/hydrophobic classification
  // should be computed separately if needed.
  comp.addRepresentation("contact", {{
    sele: "(:" + PROT_CHAIN + " or :" + PEP_CHAIN + ")",
    radiusSize: 0.04,
    weakHydrogenBond: true,
    hydrogenBond: true,
    hydrophobic: true,
    halogenBond: true,
    ionicInteraction: true,
    metalCoordination: true,
    color: "#64748B",
    labelVisible: false
  }});

  zoomBinding();
}}

function renderPublication(comp) {{
  comp.removeAllRepresentations();
  stage.setParameters({{
    backgroundColor: "white",
    sampleLevel: 2,
    impostor: true,
    quality: "high"
  }});

  baseCartoon(comp, 1, 1);
  bindingResidues(comp, 0.32, 0.40);

  comp.addRepresentation("surface", {{
    sele: ":" + PROT_CHAIN,
    color: COLORS.surface,
    opacity: 0.13,
    probeRadius: 1.4,
    smooth: 2
  }});

  zoomBinding();
}}

function setRepr(repr) {{
  if (!compRef) return;

  currentRepr = repr;

  document.querySelectorAll(".ctrl-btn").forEach(btn => btn.classList.remove("active"));

  const idMap = {{
    "interface": "btn-interface",
    "surface": "btn-surface",
    "hotspots": "btn-hotspots",
    "pocket": "btn-pocket",
    "contacts": "btn-contacts",
    "publication": "btn-pub"
  }};

  if (idMap[repr]) {{
    document.getElementById(idMap[repr]).classList.add("active");
  }}

  if (repr === "interface") renderInterface(compRef);
  if (repr === "surface") renderSurface(compRef);
  if (repr === "hotspots") renderHotspots(compRef);
  if (repr === "pocket") renderPocket(compRef);
  if (repr === "contacts") renderContacts(compRef);
  if (repr === "publication") renderPublication(compRef);
}}

function zoomBinding() {{
  if (!compRef) return;
  const sel = makeAllBindingSel();
  if (sel !== "none") {{
    compRef.autoView(sel, 1000);
  }} else {{
    compRef.autoView();
  }}
}}

function toggleSpin() {{
  spinning = !spinning;
  stage.setSpin(spinning ? [0, 1, 0] : null);
}}

function saveImage() {{
  setRepr("publication");
  setTimeout(function() {{
    stage.makeImage({{factor: 4, antialias: true, trim: false}}).then(function(blob) {{
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "propepx_nature_style_binding_sites.png";
      a.click();
    }});
  }}, 450);
}}

function highlightResidue(chain, resi) {{
  if (!compRef) return;

  if (hoverRepr) {{
    compRef.removeRepresentation(hoverRepr);
    hoverRepr = null;
  }}

  hoverRepr = compRef.addRepresentation("spacefill", {{
    sele: makeSingleSel(chain, resi),
    color: "#7C3AED",
    radius: 0.65,
    opacity: 1.0
  }});
}}

function clearHighlight() {{
  if (hoverRepr && compRef) {{
    compRef.removeRepresentation(hoverRepr);
    hoverRepr = null;
  }}
}}

function b64ToText(b64) {{
  return atob(b64);
}}

// ── Load structure from embedded base64 using Blob ─────────────
// This avoids NGL.StringStreamer, which is not supported in your NGL version.

const blob = b64toBlob(
  PDB_B64,
  PDB_FMT === "cif" ? "chemical/x-cif" : "chemical/x-pdb"
);

stage.loadFile(blob, {{
  ext: PDB_FMT === "cif" ? "cif" : "pdb",
  defaultRepresentation: false
}}).then(function(comp) {{
  compRef = comp;
  renderInterface(comp);
  zoomBinding();
}}).catch(function(err) {{
  console.error("NGL loading error:", err);
  alert("Could not load structure. Try converting CIF to PDB, or check that the CIF/PDB file is valid.");
}});


// ── Hover tooltip ─────────────────────────────────────────────

const tooltip = document.getElementById("ngl-tooltip");

stage.signals.hovered.add(function(pickingProxy) {{
  if (pickingProxy && pickingProxy.atom) {{
    const atom = pickingProxy.atom;
    const chain = atom.chainname;
    const resi = atom.resno;
    const res = atom.resname;

    const isProt = chain === PROT_CHAIN;
    const probMap = isProt ? PROT_PROB_MAP : PEP_PROB_MAP;
    const bindArr = isProt ? PROT_BINDING_RESI : PEP_BINDING_RESI;

    const isBind = bindArr.includes(resi);
    const prob = probMap[String(resi)];

    const status = isBind
      ? '<span style="color:#007FFF;font-weight:900;">● ProPepX binding residue</span>'
      : '<span style="color:#64748b;">○ Non-binding residue</span>';

    const probLine = (prob !== null && prob !== undefined)
      ? '<div class="prob-line">Probability: ' + Number(prob).toFixed(4) + '</div>'
      : '';

    tooltip.innerHTML =
      '<strong>' + res + resi + ' · Chain ' + chain + '</strong>' +
      status +
      probLine;

    tooltip.style.display = "block";

    const pos = pickingProxy.canvasPosition;
    tooltip.style.left = (pos.x + 16) + "px";
    tooltip.style.top = (pos.y + 16) + "px";
  }} else {{
    tooltip.style.display = "none";
  }}
}});

// Linked sequence/table ↔ structure interaction
document.querySelectorAll(".seq-chip, .seq-row").forEach(function(el) {{
  el.addEventListener("mouseenter", function() {{
    const chain = this.getAttribute("data-chain");
    const resi = parseInt(this.getAttribute("data-resi"));
    highlightResidue(chain, resi);
    document.querySelectorAll("[data-chain='" + chain + "'][data-resi='" + resi + "']")
      .forEach(x => x.classList.add("active-chip", "active-row"));
  }});

  el.addEventListener("mouseleave", function() {{
    clearHighlight();
    document.querySelectorAll(".active-chip, .active-row")
      .forEach(x => x.classList.remove("active-chip", "active-row"));
  }});

  el.addEventListener("click", function() {{
    const chain = this.getAttribute("data-chain");
    const resi = parseInt(this.getAttribute("data-resi"));
    if (compRef) compRef.autoView(makeSingleSel(chain, resi), 800);
  }});
}});
</script>

</body>
</html>
"""

    if output_html:
        output_html = os.path.abspath(output_html)
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        with open(output_html, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"  Nature-style 3D Viewer HTML saved -> {output_html}")

    return html


# ==========================================================
# Optional CLI
# ==========================================================

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="ProPepX Nature-style 3D Structure Viewer")

    parser.add_argument("--protein", required=True, help="Protein sequence")
    parser.add_argument("--peptide", required=True, help="Peptide sequence")
    parser.add_argument("--pdb", default=None, help="Existing PDB/mmCIF file")
    parser.add_argument("--predict", action="store_true", help="Run ColabFold first")
    parser.add_argument("--output_dir", default="./propepx_3d_output")
    parser.add_argument("--prot_chain", default="A")
    parser.add_argument("--pep_chain", default="B")
    parser.add_argument("--mode", default="mode-global")
    parser.add_argument("--embedding", default="prottrans")
    parser.add_argument("--dataset", default="test167")
    parser.add_argument("--guide", action="store_true")
    parser.add_argument("--conda_env", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.guide:
        print_alphafold_server_guide(args.protein, args.peptide)
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.predict:
        fasta_path = prepare_colabfold_fasta(
            protein_seq=args.protein,
            peptide_seq=args.peptide,
            output_fasta=os.path.join(args.output_dir, "input.fasta"),
        )

        pdb_path = run_colabfold_prediction(
            fasta_path=fasta_path,
            output_dir=os.path.join(args.output_dir, "colabfold_output"),
            conda_env=args.conda_env,
        )

    elif args.pdb:
        pdb_path = args.pdb

    else:
        print("ERROR: provide --pdb or use --predict.")
        sys.exit(1)

    print(
        "\nWARNING: CLI mode does not run ProPepX inference. "
        "It marks no binding residues unless you call generate_html_3d_viewer() "
        "from propepx_predict.py with real infer['prot_pred'] and infer['pep_pred'].\n"
    )

    prot_pred = [0] * len(_clean_seq(args.protein))
    pep_pred = [0] * len(_clean_seq(args.peptide))

    output_html = os.path.join(args.output_dir, "propepx_3d_viewer.html")

    generate_html_3d_viewer(
        pdb_path=pdb_path,
        protein_seq=args.protein,
        peptide_seq=args.peptide,
        prot_pred=prot_pred,
        pep_pred=pep_pred,
        prot_prob=None,
        pep_prob=None,
        prot_chain_id=args.prot_chain,
        pep_chain_id=args.pep_chain,
        output_html=output_html,
        mode=args.mode,
        embedding_model=args.embedding,
        dataset=args.dataset,
    )
