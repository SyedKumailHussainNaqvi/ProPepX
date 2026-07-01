# FAQ

## Can I run ProPepX on CPU?

Yes, for small tests. GPU inference is recommended for ESM-3 and batch prediction.

## Which embedding should I use?

Use the embedding backend that matches the checkpoint family: `prottrans` for ProtTransT5 checkpoints and `esm` for ESM-3 checkpoints.

## Why does inference take longer the first time?

The first run may load checkpoints and initialize embedding/model resources. Later runs can be faster if files are already cached.

## Can I use FASTA input?

Yes. ProPepX accepts raw amino-acid sequences and FASTA-like sequences.

## Where should I put checkpoints?

Use a clear structure such as:

```text
checkpoints/
├── joint/
├── protein_side/
├── peptide_side/
└── zero_shot/
```

## Where are the model files?

The model weights and test embeddings are hosted on Hugging Face:

- https://huggingface.co/syedkumailhussain/ProPepX/tree/main
