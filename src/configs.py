# transformer_model = "google/muril-large-cased"
# tr_out = 1024
# transformer_model = "bert-base-uncased"
transformer_model = "bert-base-multilingual-uncased"
tr_out = 768
include_langs = ["eng", "hin", "mal", "tam"]
mention_model_path = "../saved_ckpts/all_mbert_mention.ckpt"
pairscore_model_path = "../saved_ckpts/all_mbert_pairscore.ckpt"
