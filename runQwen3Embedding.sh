python tools/build_item_text_emb_qwen3_hf.py \
  --mapping /home/charlie/project/recbole/dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output /home/charlie/project/recbole/dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
  --batch_size 16 --max_length 128 --dtype float16 \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config /home/charlie/project/recbole/sasrec_align_qwen3.yaml
