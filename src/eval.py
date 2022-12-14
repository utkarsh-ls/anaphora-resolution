import torch
import numpy as np
from model import MentionModel, PairScoreModel
import file_parser
import dataset
import itertools
import tempfile
import argparse
import configs
import bisect
import eval_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self):
        self.ds = dataset.MentionDataset(include_lang=configs.include_langs)
        self.mention_model = self._get_mention_model(configs.mention_model_path)
        self.pairscore_model = self._get_pairscore_model(configs.pairscore_model_path)

    def _get_mention_model(self, wts_path):
        model = MentionModel().to(device)
        model.load_state_dict(
            self._filter_model_weights(torch.load(wts_path, map_location=device))
        )
        return model

    def _get_pairscore_model(self, wts_path):
        model = PairScoreModel().to(device)
        model.load_state_dict(
            self._filter_model_weights(torch.load(wts_path, map_location=device))
        )
        return model

    def _filter_model_weights(self, weights: dict):
        new_wts = type(weights)()
        new_wts.update(
            {
                k.lstrip("model."): v
                for k, v in weights["state_dict"].items()
                if k.startswith("model.")
            }
        )
        return new_wts

    def process_file(self, file: str):
        parsed_file = file_parser.parse_file(self.ds, file)
        tokenizer_out = self.ds.tokenizer.batch_encode_plus(
            [parsed_file["token_id_list"]],
            add_special_tokens=False,
            is_split_into_words=True,
            padding="max_length",
            max_length=self.ds.MAX_SEQ_LEN,
            return_tensors="pt",
        )
        ret = {}
        ret["token_id_list"] = tokenizer_out["input_ids"]
        ret["mask_lists"] = tokenizer_out["attention_mask"]

        mention_list = parsed_file["is_mention"]
        # mention_list += [0] * (self.ds.MAX_SEQ_LEN - len(mention_list))
        # ret["mention_list"] = torch.as_tensor(mention_list, dtype=torch.float32)
        # first_token_idx = parsed_file["first_token_idx"]
        # # first_token_idx += [-1] * (self.ds.MAX_SEQ_LEN - len(first_token_idx))
        # ret["first_token_idx"] = torch.as_tensor(first_token_idx, dtype=torch.long)
        ret["mention_list"] = mention_list
        ret["mention_clusters"] = parsed_file["mention_clusters"]
        ret["org_words"] = parsed_file["org_words"]
        ret["first_token_idx"] = parsed_file["first_token_idx"]
        ret["tok_cnt"] = len(parsed_file["token_id_list"])

        return ret

    def run_on_file(self, file):
        processed_file = self.process_file(file)

        mention_logits, word_embeds = self.mention_model(
            processed_file["token_id_list"].to(device),
            processed_file["mask_lists"].to(device),
        )
        sel_words = torch.sigmoid(mention_logits[0]) >= 0.7
        selected_indices = torch.arange(len(sel_words))[sel_words].tolist()

        pred_clusters = []
        pair_decisions = {}

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
                p = tuple(sorted([idx1, idx2]))
                if p in pair_decisions and (not pair_decisions[p]):
                    return False
                pred_clusters.append(list(p))
            elif idx1_list is None:
                for id2_mem in idx2_list:
                    p = tuple(sorted((id2_mem, idx1)))
                    if p in pair_decisions and (not pair_decisions[p]):
                        return False
                idx2_list.append(idx1)
            elif idx2_list is None:
                for id1_mem in idx1_list:
                    p = tuple(sorted((id1_mem, idx2)))
                    if p in pair_decisions and (not pair_decisions[p]):
                        return False
                idx1_list.append(idx2)
            else:
                if idx1_list is idx2_list:
                    return True
                for id1_mem, id2_mem in itertools.product(idx1_list, idx2_list):
                    p = tuple(sorted((id1_mem, id2_mem)))
                    if p in pair_decisions and (not pair_decisions[p]):
                        return False
                idx1_list.extend(idx2_list)
                pred_clusters.remove(idx2_list)
            return True

        gt_is_pair = []
        pred_is_pair = []

        def get_gt_is_pair(i, j):
            for c in processed_file["mention_clusters"]:
                if i in c and j in c:
                    return 1
            return 0

        pairscore_preds = []
        for i, j in itertools.combinations(selected_indices, 2):
            word1_embed = word_embeds[:, i]
            word2_embed = word_embeds[:, j]
            pairscore = self.pairscore_model(word1_embed, word2_embed)
            pairscore = torch.sigmoid(pairscore).item()
            pairscore_preds.append([pairscore, True, i, j])
            pairscore_preds.append([1 - pairscore, False, i, j])
            gt_is_pair.append(get_gt_is_pair(i, j))
            if pairscore > 0.7:
                # add_pair_to_clusters(i, j)
                pred_is_pair.append(1)
            else:
                pred_is_pair.append(0)

        pairscore_preds = np.asarray(pairscore_preds)
        if pairscore_preds.shape != (0,):
            pairscore_preds = pairscore_preds[np.argsort(-pairscore_preds[:, 0])]
        for pairscore, will_join, i, j in pairscore_preds:
            if pairscore < 0.7:
                break
            if (i, j) in pair_decisions:
                continue

            if will_join:
                mergeable = add_pair_to_clusters(i, j)
                if mergeable:
                    pair_decisions[(i, j)] = True
            else:
                merged = False
                for c in pred_clusters:
                    if i in c and j in c:
                        merged = True
                        break
                if not merged:
                    pair_decisions[(i, j)] = False

        real_clusters = processed_file["mention_clusters"]
        real_clusters = sorted([sorted(i) for i in real_clusters])
        pred_clusters = sorted([sorted(i) for i in pred_clusters])
        gt_selected_indices = np.flatnonzero(processed_file["mention_list"]).tolist()

        accs_mention = eval_metric.get_accs_mention(
            gt_selected_indices, selected_indices, processed_file["tok_cnt"]
        )
        accs_pairscore = eval_metric.get_accs_pairscore(gt_is_pair, pred_is_pair)
        return (
            processed_file["org_words"],
            processed_file["first_token_idx"],
            gt_selected_indices,
            selected_indices,
            real_clusters,
            pred_clusters,
            accs_mention,
            accs_pairscore,
        )

    def eval_single_file(self, file_name, verbose=False):
        (
            org_words,
            first_token_idx,
            mention_list_gt,
            mention_list_pred,
            clusters_gt,
            clusters_pred,
            accs_mention,
            accs_pairscore,
        ) = self.run_on_file(file_name)

        to_word_idx = lambda tok_idx: max(
            0, bisect.bisect_right(first_token_idx, tok_idx) - 1
        )
        words_mt_list_gt = []
        words_mt_list_pred = []
        words_cluster_gt = []
        words_cluster_pred = []
        for tok_idx in mention_list_gt:
            word_idx = to_word_idx(tok_idx)
            words_mt_list_gt.append(word_idx)
        for tok_idx in mention_list_pred:
            word_idx = to_word_idx(tok_idx)
            words_mt_list_pred.append(word_idx)

        words_mt_list_pred = list(set(words_mt_list_pred))

        for c in clusters_gt:
            word_c = []
            for tok_idx in c:
                word_idx = to_word_idx(tok_idx)
                word_c.append(word_idx)
            words_cluster_gt.append(word_c)
        for c in clusters_pred:
            word_c = []
            for tok_idx in c:
                word_idx = to_word_idx(tok_idx)
                word_c.append(word_idx)
            words_cluster_pred.append(word_c)

        words_cluster_pred = [sorted(set(c)) for c in words_cluster_pred]

        if verbose:
            org_sentence = np.array(org_words, dtype=str)
            print("Input Sentence: \n\t", end="")
            print(" ".join(org_sentence))
            print("Mention list ground truth:\t", end="")
            print(org_sentence[words_mt_list_gt].tolist())
            print("Mention list predicted:   \t", end="")
            print(org_sentence[words_mt_list_pred].tolist())
            print("Clusters ground truth:    \t", end="")
            print([org_sentence[c].tolist() for c in words_cluster_gt])
            print("Clusters predicted:       \t", end="")
            print([org_sentence[c].tolist() for c in words_cluster_pred])

        return (
            eval_metric.get_all_scores(words_cluster_gt, words_cluster_pred),
            accs_mention,
            accs_pairscore,
        )

    def predict_single_input(self, input_str, verbose=False):
        with tempfile.NamedTemporaryFile("w+") as f:
            for word in input_str.split(" "):
                if word != "\n":
                    word = word + "\t-\t-\n"
                f.write(word)
            f.flush()
            (
                org_words,
                first_token_idx,
                _,
                mention_list_pred,
                _,
                clusters_pred,
                _,
                _,
            ) = self.run_on_file(f.name)

        to_word_idx = lambda tok_idx: max(
            0, bisect.bisect_right(first_token_idx, tok_idx) - 1
        )
        words_mt_list_pred = []
        words_cluster_pred = []
        for tok_idx in mention_list_pred:
            word_idx = to_word_idx(tok_idx)
            words_mt_list_pred.append(word_idx)

        words_mt_list_pred = list(set(words_mt_list_pred))

        for c in clusters_pred:
            word_c = []
            for tok_idx in c:
                word_idx = to_word_idx(tok_idx)
                word_c.append(word_idx)
            words_cluster_pred.append(word_c)

        org_sentence = np.array(org_words, dtype=str)
        cleaned_inp_sentece = " ".join(org_sentence)
        mention_words_list_pred = org_sentence[words_mt_list_pred].tolist()
        word_clusters_pred = [org_sentence[c].tolist() for c in words_cluster_pred]
        if verbose:
            print(f"Cleaned Input Sentence Sentence: \n\t{cleaned_inp_sentece}")
            print(f"Mention list predicted:   \t{mention_words_list_pred}")
            print(f"Clusters predicted:       \t{word_clusters_pred}")

        return (cleaned_inp_sentece, mention_words_list_pred, word_clusters_pred)

    def eval_files(self):
        test_ds = dataset.get_dataloaders(
            self.ds, {"val_split": 0.2, "test_split": 0, "batch_size": 128}
        )[1].dataset

        test_files = [
            test_ds.dataset.files[test_ds.indices[i]] for i in range(len(test_ds))
        ]
        metrics = []
        accs_mention = []
        accs_pairscore = []
        for file in test_files:
            (
                metric_file,
                accs_mention_file,
                accs_pairscore_file,
            ) = self.eval_single_file(file)

            metrics.append(metric_file)
            accs_mention.append(accs_mention_file)
            accs_pairscore.append(accs_pairscore_file)

        mention_accs = (np.mean(accs_mention, axis=0) * 100).round(3).tolist()
        print("For mention model:")
        print(
            f"accuracy: {mention_accs[0]}, precision:{mention_accs[1]},"
            f" recall:{mention_accs[2]}, f1:{mention_accs[3]}"
        )

        pairscore_accs = (np.mean(accs_pairscore, axis=0) * 100).round(3).tolist()
        print("For pairscore model:")
        print(
            f"accuracy: {pairscore_accs[0]}, precision:{pairscore_accs[1]},"
            f" recall:{pairscore_accs[2]}, f1:{pairscore_accs[3]}"
        )

        coref_metrics = (np.mean(metrics, axis=0) * 100).round(3).tolist()
        print("MUC Score: ", coref_metrics[0])
        print("B3 Score: ", coref_metrics[1])
        print("BLANC Score: ", coref_metrics[2])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_all_file",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="To eval all file",
    )
    parser.add_argument(
        "--eval_single_file",
        type=str,
        default=None,
        nargs="?",
        const="../data/clean/eng_train_files/EngFile_col7.txt",
        help="Path to eval single file",
    )
    parser.add_argument(
        "--predict_from_file",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="To eval all file",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        nargs="?",
        const="John is arrogant. \n Yes, he thinks he can win.",
        help="prediction text",
    )
    args = parser.parse_args()

    if args.eval_all_file:
        evl = Evaluator()
        evl.eval_files()
    elif args.eval_single_file:
        evl = Evaluator()
        evl.eval_single_file(args.eval_single_file, True)
    elif args.predict_from_file:
        with open("inp.txt", "r") as f:
            input_str = ""
            for line in f:
                if line == "\n":
                    line = "\n "
                else:
                    line = line[:-1] + " "
                input_str += line

        evl = Evaluator()
        evl.predict_single_input(input_str, True)
    elif args.predict:
        evl = Evaluator()
        evl.predict_single_input(args.predict, verbose=True)
    else:
        print("No option provided !!")
        parser.print_help()


if __name__ == "__main__":
    main()
