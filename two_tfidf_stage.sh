python scripts/exp_align_two_stage.py \
  --dataset Amazon_Beauty \
  --mode base \
  --stage1_epochs 8 \
  --stage2_epochs 50 \
  --stage2_lr 3e-5 \
  --temperatures 0.05 0.07 \
  --weights 0.05 0.1 0.2 \
  --exclude_topk 0
