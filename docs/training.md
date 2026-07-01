# Training and fine-tuning

## Pretraining: protein-side model

```bash
python pretrain_propepx.py \
  --mode prot \
  --emb_dim 1024 \
  --train_h5 "data/embeddings/Transfer_Learning_Training_embeddings.h5" \
  --val_h5 "data/embeddings/Transfer_Learning_Validation_embeddings.h5" \
  --ckpt_dir "checkpoints/pretrain/"
```

## Pretraining: peptide-side model

```bash
python pretrain_propepx.py \
  --mode pep \
  --emb_dim 1024 \
  --train_h5 "data/embeddings/TL_train_Prottran_embeddings.h5" \
  --val_h5 "data/embeddings/TL_validation_Prottran_embeddings.h5" \
  --ckpt_dir "checkpoints/pretrain/"
```

## Pretraining: joint model

```bash
python pretrain_propepx.py \
  --mode mode-GLOBAL \
  --emb_dim 1024 \
  --train_h5 "data/embeddings/Transfer_Learning_Training_embeddings.h5" \
  --val_h5 "data/embeddings/TL_validation_Prottran_embeddings.h5" \
  --ckpt_dir "checkpoints/pretrain/" \
  --gpu_id 0
```

## Fine-tuning: protein-side model

```bash
python finetune_propepx.py \
  --mode prot \
  --train_h5 "data/embeddings/ESM_train_TS092_Dataset.h5" \
  --val_h5 "data/embeddings/ESM_val_TS092_Dataset.h5" \
  --test_h5 "data/embeddings/ESM_test_TS092_Dataset.h5" \
  --pretrained_ckpt "checkpoints/pretrain/best_pretrain_prot.pt" \
  --ckpt_dir "checkpoints/finetune/prot_ts092/" \
  --emb_dim 1152 \
  --epochs 3 \
  --gpu_id 0
```

## Fine-tuning: peptide-side model

```bash
python finetune_propepx.py \
  --mode pep \
  --train_h5 "data/embeddings/CAMP_Train_Prottran_embeddings.h5" \
  --val_h5 "data/embeddings/CAMP_Test_Prottran_embeddings.h5" \
  --test_h5 "data/embeddings/CAMP_Test_Prottran_embeddings.h5" \
  --pretrained_ckpt "checkpoints/pretrain/best_pretrain_pep.pt" \
  --ckpt_dir "checkpoints/finetune/pep_camp231/" \
  --emb_dim 1024 \
  --epochs 3 \
  --gpu_id 0
```

## Fine-tuning: joint model

```bash
python finetune_propepx.py \
  --mode mode-GLOBAL \
  --train_h5 "data/embeddings/TrainingDataset-KGIPA_prot1418_embedding.h5" \
  --test_h5 "data/embeddings/Test167_embeddings.h5" \
  --pretrained_ckpt "checkpoints/pretrain/best_pretrain_global.pt" \
  --ckpt_dir "checkpoints/finetune/joint_test167/" \
  --emb_dim 1024 \
  --epochs 3 \
  --n_splits 5 \
  --gpu_id 0
```
