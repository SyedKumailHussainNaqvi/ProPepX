# Models and datasets

Main model hub:

- [ProPepX on Hugging Face](https://huggingface.co/syedkumailhussain/ProPepX/tree/main)

## Local organization

```text
checkpoints/
├── joint/
├── protein_side/
├── peptide_side/
└── zero_shot/

data/
├── embeddings/
├── benchmark/
└── examples/
```

## Available checkpoint families

| Task | Backbone | Dataset/setting |
|---|---|---|
| Joint protein–peptide prediction | ESM-3 600M | LEADS TS251 |
| Joint protein–peptide prediction | ProtTransT5 | LEADS TS251 |
| Joint protein–peptide prediction | ESM-3 600M | Test167 |
| Joint protein–peptide prediction | ProtTransT5 | Test167 |
| Protein-side prediction | ESM-3 600M | TS092, TS125, TS251, TS639 |
| Protein-side prediction | ProtTransT5 | TS092, TS125, TS251, TS639 |
| Peptide-side prediction | ESM-3 600M | CAMP TS231 |
| Peptide-side prediction | ProtTransT5 | CAMP TS231 |
| Zero-shot prediction | ESM-3 600M | TS167 zero-shot |
| Zero-shot prediction | ProtTransT5 | TS167 zero-shot |
