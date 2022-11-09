from itertools import combinations
from scipy.special import comb
import os
import numpy as np
import torch
from transformers import BertTokenizer
import configs
import file_parser

from pl_module import PLModuleMention


class MentionDataset(torch.utils.data.Dataset):
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MAX_SEQ_LEN = 410

    def __init__(self, ds_folder="../data/clean", include_lang=["eng"], pad=True):
        super().__init__()
        self.pad = pad
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            configs.transformer_model
        )
        self.include_lang = include_lang
        if include_lang:
            all_folders = [os.path.join(ds_folder, f) for f in os.listdir(ds_folder)]
            folders = []
            for folder in all_folders:
                for lang in include_lang:
                    if lang in folder:
                        folders.append(folder)
                        break
            self.files = []
            for folder in folders:
                self.files.extend([os.path.join(folder, f) for f in os.listdir(folder)])
        else:
            self.files = [os.path.join(ds_folder, f) for f in os.listdir(ds_folder)]
        self.files = sorted([f for f in self.files if f.endswith(".txt")])

        token_lists = []
        ref_idx_lists = []
        mention_lists = []
        first_token_idx_lists = []
        for i in range(len(self)):
            parsed_file = file_parser.parse_file(self, self.files[i])
            tok_list, ref_idx_list, mention_list, first_token_idx = (
                parsed_file["token_id_list"],
                parsed_file["ref_index_list"],
                parsed_file["is_mention"],
                parsed_file["first_token_idx"],
            )
            token_lists.append(tok_list)
            ref_idx_lists.append(ref_idx_list)
            mention_lists.append(mention_list)
            first_token_idx_lists.append(first_token_idx)

        tokenizer_out = self.tokenizer.batch_encode_plus(
            token_lists,
            add_special_tokens=False,
            is_split_into_words=True,
            padding=self.pad,
            return_tensors="pt",
        )
        self.token_id_lists = tokenizer_out["input_ids"]
        self.MAX_SEQ_LEN = len(self.token_id_lists[0])
        self.mask_lists = tokenizer_out["attention_mask"]
        # for i in range(len(token_lists)):
        #     if len(token_lists[i]) == self.MAX_SEQ_LEN-2:
        #         print(i, self.files[i])
        if self.pad:
            for i in range(len(self.files)):
                ref_idx_lists[i] += [0] * (self.MAX_SEQ_LEN - len(ref_idx_lists[i]))
                mention_lists[i] += [0] * (self.MAX_SEQ_LEN - len(mention_lists[i]))
                first_token_idx_lists[i] += [-1] * (
                    self.MAX_SEQ_LEN - len(first_token_idx_lists[i])
                )

        self.ref_idx_lists = torch.as_tensor(ref_idx_lists, dtype=torch.long)
        self.mention_lists = torch.as_tensor(mention_lists, dtype=torch.float32)
        self.first_token_idx_lists = torch.as_tensor(
            first_token_idx_lists, dtype=torch.long
        )
        # self.one_hot_lists = []
        # _eye = np.eye(self.MAX_SEQ_LEN)
        # for ref_idx_list in ref_idx_lists:
        #     self.one_hot_lists.append(_eye[ref_idx_list])

        # self.one_hot_lists = torch.as_tensor(self.one_hot_lists, dtype=torch.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return (
            self.token_id_lists[idx],
            self.mask_lists[idx],
            self.mention_lists[idx],
            self.first_token_idx_lists[idx],
        )


class PairScoreDataset(torch.utils.data.Dataset):
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MAX_SEQ_LEN = 410

    def __init__(self, ds_folder="../data/clean", include_lang=["eng"], pad=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad = pad
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            configs.transformer_model
        )
        self.include_lang = include_lang
        if include_lang:
            all_folders = [os.path.join(ds_folder, f) for f in os.listdir(ds_folder)]
            folders = []
            for folder in all_folders:
                for lang in include_lang:
                    if lang in folder:
                        folders.append(folder)
                        break
            self.files = []
            for folder in folders:
                self.files.extend([os.path.join(folder, f) for f in os.listdir(folder)])
        else:
            self.files = [os.path.join(ds_folder, f) for f in os.listdir(ds_folder)]
        self.files = sorted([f for f in self.files if f.endswith(".txt")])

        token_lists = []
        ref_idx_lists = []
        mentn_cluster_lists = []
        for i in range(len(self.files)):
            parsed_file = file_parser.parse_file(self, self.files[i])
            tok_list, ref_idx_list, mention_cluster = (
                parsed_file["token_id_list"],
                parsed_file["ref_index_list"],
                parsed_file["mention_clusters"],
            )
            token_lists.append(tok_list)
            ref_idx_lists.append(ref_idx_list)
            mentn_cluster_lists.append(mention_cluster)
        self.mentn_cluster_lists = [[list(c) for c in mc] for mc in mentn_cluster_lists]

        tokenizer_out = self.tokenizer.batch_encode_plus(
            token_lists,
            add_special_tokens=False,
            is_split_into_words=True,
            padding=self.pad,
            return_tensors="pt",
        )
        self.token_id_lists = tokenizer_out["input_ids"]
        self.MAX_SEQ_LEN = len(self.token_id_lists[0])
        self.mask_lists = tokenizer_out["attention_mask"]
        # for i in range(len(token_lists)):
        #     if len(token_lists[i]) == self.MAX_SEQ_LEN-2:
        #         print(i, self.files[i])
        if self.pad:
            for i in range(len(self.files)):
                ref_idx_lists[i] += [0] * (self.MAX_SEQ_LEN - len(ref_idx_lists[i]))

        self.MAX_CLUSTER_COUNT = max([len(c) for c in self.mentn_cluster_lists])
        for i in range(len(self.mentn_cluster_lists)):
            self.mentn_cluster_lists[i] += [[]] * (
                self.MAX_CLUSTER_COUNT - len(self.mentn_cluster_lists[i])
            )

        self.MAX_CLUSTER_SIZE = max(
            [
                max([len(c) for c in cluster_list])
                for cluster_list in self.mentn_cluster_lists
            ]
        )
        for i in range(len(self.mentn_cluster_lists)):
            for j in range(len(self.mentn_cluster_lists[i])):
                self.mentn_cluster_lists[i][j] += [-1] * (
                    self.MAX_CLUSTER_SIZE - len(self.mentn_cluster_lists[i][j])
                )

        self.ref_idx_lists = torch.as_tensor(ref_idx_lists, dtype=torch.long)
        # self.one_hot_lists = []
        # _eye = np.eye(self.MAX_SEQ_LEN)
        # for ref_idx_list in ref_idx_lists:
        #     self.one_hot_lists.append(_eye[ref_idx_list])

        # self.one_hot_lists = torch.as_tensor(self.one_hot_lists, dtype=torch.float32)
        mention_model = PLModuleMention(self.MAX_SEQ_LEN)
        mention_model.load_state_dict(
            torch.load("../men_hin.ckpt", map_location=self.device)["state_dict"]
        )
        mention_model.eval()
        mention_model.model.eval()
        mention_model.to(self.device)
        for p in mention_model.parameters():
            p.requires_grad = False

        word1_embed_list = []
        word2_embed_list = []
        is_pair_list = []
        mention_logits = []
        word_embeds = []
        with torch.no_grad():
            for tok_id_l, mask_l in zip(
                self.token_id_lists.to(self.device).chunk(10),
                self.mask_lists.to(self.device).chunk(10),
            ):
                mention_logits_i, word_embeds_i = mention_model(tok_id_l, mask_l)
                mention_logits_i = mention_logits_i.cpu().numpy()
                word_embeds_i = word_embeds_i.cpu().numpy()
                mention_logits.extend(list(mention_logits_i))
                word_embeds.extend(list(word_embeds_i))
            mention_logits = torch.tensor(np.asarray(mention_logits))
            word_embeds = torch.tensor(np.asarray(word_embeds))

        selected_words = mention_logits >= 0.5

        def is_in_same_cluster(i, j, clist):
            for c in clist:
                if i in c and j in c:
                    return 1.0
            return 0.0

        def stats():
            def get_all_correct_pairs_cnt(men_clists):
                cnt = 0
                for clist in men_clists:
                    for c in clist:
                        clen = len([i for i in c if i != -1])
                        cnt += clen * (clen - 1) / 2
                return cnt

            def get_all_acc_mention(mention_logits, men_clists):
                sel_words = mention_logits >= 0.5
                non_sel_words = mention_logits < 0.5
                total_cnt, pred_pair_cnt, total_neg_cnt = (
                    mention_logits.numel(),
                    1e-6,
                    non_sel_words.sum(),
                )
                pred_cnt, pred_pos_cnt = 0, 0
                act_pos_cnt = 0
                for swords, men_clist in zip(sel_words, men_clists):
                    sel_indices = torch.arange(len(swords))[swords]
                    act_pos_cnt += np.sum(
                        [comb(len([i for i in c if i != -1]), 2) for c in men_clist]
                    )
                    for i, j in combinations(sel_indices, 2):
                        true_class = is_in_same_cluster(i, j, men_clist)
                        if true_class == 1:
                            pred_pos_cnt += 1
                            pred_cnt += 1
                        pred_pair_cnt += 1

                print(f"Total predicted pairs  :{pred_pair_cnt}")
                print(
                    f"Positive Acc: {pred_pos_cnt}/{pred_pair_cnt} : {pred_pos_cnt/pred_pair_cnt}"
                )
                print(f"Actual Positive pairs: {act_pos_cnt}")
                # print(f"Negative Acc: {pred_neg_cnt}/{total_neg_cnt} : {pred_neg_cnt/total_neg_cnt}")

            get_all_acc_mention(mention_logits, self.mentn_cluster_lists)

            all_correct_pairs_cnt = get_all_correct_pairs_cnt(self.mentn_cluster_lists)
            print(
                mention_logits.numel(),
                selected_words.sum().item(),
                all_correct_pairs_cnt,
            )
            # exit(0)

        # stats()
        for w_embeds, sel_words, clist in zip(
            word_embeds, selected_words, self.mentn_cluster_lists
        ):
            indices = torch.arange(len(sel_words))[sel_words]
            for i, j in combinations(indices, 2):
                word1_embed_list.append(w_embeds[i].numpy())
                word2_embed_list.append(w_embeds[j].numpy())
                is_pair_list.append(is_in_same_cluster(i, j, clist))

        self.word1_embed_list = torch.tensor(np.asarray(word1_embed_list))
        self.word2_embed_list = torch.tensor(np.asarray(word2_embed_list))
        self.is_pair_list = torch.tensor(is_pair_list)
        print(len(is_pair_list))
        # exit(0)

    def __len__(self):
        return len(self.is_pair_list)

    def __getitem__(self, idx):
        return (
            self.word1_embed_list[idx],
            self.word2_embed_list[idx],
            self.is_pair_list[idx],
        )


def get_dataloaders(ds, config):
    # ds = ds_cls(include_lang=config["langs"])
    total_size = len(ds)
    test_size = int(total_size * config["test_split"])
    val_size = int(total_size * config["val_split"])
    train_size = total_size - test_size - val_size
    # print("TOTAL SIZE:", total_size)
    # print("TRAIN SIZE:", train_size)
    # print("VAL SIZE:", val_size)
    # print("TEST SIZE:", test_size)

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=8
    )
    # print(len(train_ds), len(val_ds), len(test_ds))
    return train_loader, val_loader, test_loader


def view_ds(ds: MentionDataset):
    for tok_id_list, mask, ref_idx_list in ds:
        tok_list = ds.tokenizer.convert_ids_to_tokens(tok_id_list)
        tok_list = list(filter(lambda tok: tok != "[PAD]", tok_list))
        print(tok_list)


def view_ds_ps(ds: PairScoreDataset):
    for tok_id_list, mask, ref_idx_list, mentn_cluster_list, f in ds:
        tok_list = ds.tokenizer.convert_ids_to_tokens(tok_id_list)
        tok_list = list(filter(lambda tok: tok != "[PAD]", tok_list))
        print(f, mentn_cluster_list)


def get_mention_ratio(ds: MentionDataset):
    all_mentions = ds.mention_lists
    pos_wt = (all_mentions == 0).sum() / (all_mentions == 1).sum()
    pos_wt = pos_wt.item()
    print(
        (all_mentions == 0).sum(),
        (all_mentions == 1).sum(),
        all_mentions.numel(),
        pos_wt,
    )
    return pos_wt


if __name__ == "__main__":
    # ds = mLangDataset("ds/", include_lang=["eng"])
    # print(len(ds))
    # mx = 0
    # # for i in range(len(ds)):
    # #     # print(i,ds.files[i], end='\n')
    # #     print(i, end="\r")
    # #     b = ds[i]
    # #     mx = max(len(b[0]), mx)
    # # print()
    # # print(mx)
    # print(ds[0][0].size(), ds[0][1].size())
    # view_ds(ds)
    # get_mention_ratio(MentionDataset())
    ds = PairScoreDataset(include_lang=["eng", "hin"])
    # view_ds(ds)
