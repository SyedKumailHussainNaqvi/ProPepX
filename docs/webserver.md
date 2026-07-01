# Web server

## Static demo

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## FastAPI application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## Workflow

```text
Protein sequence
      ↓
Peptide sequence
      ↓
Embedding backend
      ↓
Prediction mode
      ↓
Model checkpoint
      ↓
Interactive HTML report
```
