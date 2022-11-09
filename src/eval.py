import torch
import numpy as np
from model import MentionModel, PairScoreModel
import file_parser
import dataset
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_model_weights(weights: dict):
    new_wts = type(weights)()
    new_wts.update(
        {
            k.lstrip("model."): v
            for k, v in weights["state_dict"].items()
            if k.startswith("model.")
        }
    )
    return new_wts


def get_mention_model(wts_path):
    model = MentionModel().to(device)
    model.load_state_dict(
        filter_model_weights(torch.load(wts_path, map_location=device))
    )
    return model


def get_pairscore_model(wts_path):
    model = PairScoreModel().to(device)
    model.load_state_dict(
        filter_model_weights(torch.load(wts_path, map_location=device))
    )
    return model


def process_file(ds: dataset.MentionDataset, file: str):
    parsed_file = file_parser.parse_file(ds, file)
    tokenizer_out = ds.tokenizer.batch_encode_plus(
        [parsed_file["token_id_list"]],
        add_special_tokens=False,
        is_split_into_words=True,
        padding="max_length",
        max_length=ds.MAX_SEQ_LEN,
        return_tensors="pt",
    )
    ret = {}
    ret["token_id_list"] = tokenizer_out["input_ids"]
    ret["mask_lists"] = tokenizer_out["attention_mask"]

    mention_list = parsed_file["is_mention"]
    # mention_list += [0] * (ds.MAX_SEQ_LEN - len(mention_list))
    # ret["mention_list"] = torch.as_tensor(mention_list, dtype=torch.float32)
    ret["mention_list"] = mention_list
    first_token_idx = parsed_file["first_token_idx"]
    first_token_idx += [-1] * (ds.MAX_SEQ_LEN - len(first_token_idx))
    ret["first_token_idx"] = torch.as_tensor(first_token_idx, dtype=torch.long)
    ret["mention_clusters"] = parsed_file["mention_clusters"]

    return ret


@torch.no_grad()
def run_on_file(file):
    ds = dataset.MentionDataset(include_lang=["eng", "hin"])
    mention_model = get_mention_model("../men_hin.ckpt")
    pairscore_model = get_pairscore_model("../ps_hi_en.ckpt")
    processed_file = process_file(ds, file)

    mention_logits, word_embeds = mention_model(
        processed_file["token_id_list"].to(device),
        processed_file["mask_lists"].to(device),
    )
    sel_words = torch.sigmoid(mention_logits[0]) >= 0.9
    selected_indices = torch.arange(len(sel_words))[sel_words].tolist()

    print(
        "real mention list: ", np.flatnonzero(processed_file["mention_list"]).tolist()
    )
    print("predicted mention list: ", selected_indices)

    pred_clusters = []

    def add_pair_to_clusters(idx1, idx2):
        idx1_list = None
        idx2_list = None
        for cluster in pred_clusters:
            if idx1 in cluster:
                idx1_list = cluster
                break
        for cluster in pred_clusters:
            if idx2 in cluster:
                idx2_list = cluster
                break

        if idx1_list is None and idx2_list is None:
            pred_clusters.append([idx1, idx2])
        elif idx1_list is None:
            idx2_list.append(idx1)
        elif idx2_list is None:
            idx1_list.append(idx2)
        else:
            return 0
        return 1

    for i, j in itertools.combinations(selected_indices, 2):
        word1_embed = word_embeds[:, i]
        word2_embed = word_embeds[:, j]
        pairscore = pairscore_model(word1_embed, word2_embed)
        pairscore = torch.sigmoid(pairscore)
        if pairscore > 0.6:
            add_pair_to_clusters(i, j)

    real_clusters = processed_file["mention_clusters"]
    real_clusters = sorted([sorted(i) for i in real_clusters])
    pred_clusters = sorted([sorted(i) for i in pred_clusters])
    print("real_clusters: ", real_clusters)
    print("pred_clusters: ", pred_clusters)


def main():
    run_on_file("../data/clean/eng_train_files/EngFile_col11.txt")


if __name__ == "__main__":
    main()
