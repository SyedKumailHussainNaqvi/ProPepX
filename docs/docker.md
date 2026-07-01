# Docker

## GPU

```bash
docker build -t propepx:latest .
docker run --gpus all -p 8000:8000 propepx:latest
```

## CPU

```bash
docker run -p 8000:8000 propepx:latest
```

Open:

```text
http://localhost:8000
```
