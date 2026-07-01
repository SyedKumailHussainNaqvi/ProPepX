# Reproducibility

## Reviewer checklist

| Item | Status |
|---|:---:|
| Source code | Available |
| Conda environment | Available |
| Command-line inference | Available |
| Web-server interface | Available |
| Pretrained/fine-tuned checkpoints | Available on Hugging Face |
| Test embeddings | Available on Hugging Face |
| Training scripts | Available |
| Fine-tuning scripts | Available |
| Example commands | Available |
| Docker instructions | Available |

## Before running predictions

- Confirm the Conda environment was created from `propepx.yml`.
- Confirm the checkpoint matches the selected prediction mode.
- Confirm the embedding backend matches the checkpoint family.
- Confirm protein and peptide sequences contain valid amino-acid characters.
- Confirm the output directory exists and is writable.
- Confirm `--gpu_id` is valid when using CUDA.

## Output interpretation

| Output | Meaning |
|---|---|
| Residue probability | Probability that each residue participates in binding |
| Binary label | Binding/non-binding residue prediction after thresholding |
| HTML report | Interactive summary for protein, peptide, and figures |
| Heatmap | Residue-level interaction or interpretability map |
| CSV file | Machine-readable prediction table |
| JSON file | Reproducible run metadata |
