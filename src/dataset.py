from itertools import combinations
from scipy.special import comb
import os
import numpy as np
import torch
from transformers import BertTokenizer
import configs

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
        for i in range(len(self)):
            tok_list, ref_idx_list, mention_list = self._parse_file(i)
            token_lists.append(tok_list)
            ref_idx_lists.append(ref_idx_list)
            mention_lists.append(mention_list)

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

        self.ref_idx_lists = torch.as_tensor(ref_idx_lists, dtype=torch.long)
        self.mention_lists = torch.as_tensor(mention_lists, dtype=torch.float32)
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
        )

    def _parse_file(self, index):
        with open(self.files[index]) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        token_list = [self.CLS_TOKEN]
        ref_id_list = [-1]
        is_mention = [0]
        word_ids_to_idx = {-1: 0}
        user_name_to_id = {}

        def update_list(tok, ref_id, is_m):
            token_list.append(tok)
            ref_id_list.append(ref_id)
            is_mention.append(is_m)

        was_last_line_empty = True
        for line in lines:
            if not line:
                update_list(self.SEP_TOKEN, -1, 0)
                was_last_line_empty = True
                continue
            line = line.split()
            if len(line) != 3:
                continue
            word, word_id, ref_id = line
            if was_last_line_empty and word == "0000":
                was_last_line_empty = False
                continue
            word_id = int(word_id) if word_id.isnumeric() else -1
            ref_id = int(ref_id) if ref_id.isnumeric() else -1
            word = word.lower()
            # if ref_id == 8:
            # print(line, self.files[index], flush=True)
            if word[0] == "@":
                if word not in user_name_to_id:
                    user_name_to_id[word] = "user" + str(len(user_name_to_id))
                word = user_name_to_id[word]

            if word_id != -1:
                word_ids_to_idx[word_id] = len(token_list)
            tokens = self.tokenizer.tokenize(word)
            update_list(tokens[0], ref_id, 1 if word_id != -1 else 0)
            was_last_line_empty = False
            tokens = tokens[1:]
            for tok in tokens:
                update_list(tok, -1, 0)

        token_id_list = self.tokenizer.convert_tokens_to_ids(token_list)
        # tokenizer_out = self.tokenizer.encode_plus(token_list, add_special_tokens=False,padding=False,return_tensors='pt')
        ref_index_list = [word_ids_to_idx.get(ref_id, 0) for ref_id in ref_id_list]

        assert len(token_id_list) == len(ref_index_list) == len(is_mention)
        return token_id_list, ref_index_list, is_mention
        # return tokenizer_out['input_ids'], tokenizer_out['attention_mask'], ref_index_list


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
            tok_list, ref_idx_list, mention_cluster = self._parse_file(i)
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
            torch.load("../mention.ckpt", map_location=self.device)["state_dict"]
        )
        mention_model.eval()
        mention_model.model.eval()
        mention_model.to(self.device)
        for p in mention_model.parameters():
            p.requires_grad = False

        word1_embed_list = []
        word2_embed_list = []
        is_pair_list = []
        with torch.no_grad():
            mention_logits, word_embeds = mention_model(
                self.token_id_lists.to(self.device), self.mask_lists.to(self.device)
            )
            mention_logits = mention_logits.cpu()
            word_embeds = word_embeds.cpu()

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

    def _parse_file(self, index):
        with open(self.files[index]) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        token_list = [self.CLS_TOKEN]
        ref_id_list = [-1]
        word_ids_to_idx = {-1: 0}
        user_name_to_id = {}
        mention_clusters = []

        def update_list(tok, ref_id):
            token_list.append(tok)
            ref_id_list.append(ref_id)

        was_last_line_empty = True
        for line in lines:
            if not line:
                update_list(self.SEP_TOKEN, -1)
                was_last_line_empty = True
                continue
            line = line.split()
            if len(line) != 3:
                continue
            word, word_id, ref_id = line
            if was_last_line_empty and word == "0000":
                was_last_line_empty = False
                continue
            word_id = int(word_id) if word_id.isnumeric() else -1
            ref_id = int(ref_id) if ref_id.isnumeric() else -1

            word = word.lower()
            # if ref_id == 8:
            # print(line, self.files[index], flush=True)
            if word[0] == "@":
                if word not in user_name_to_id:
                    user_name_to_id[word] = "user" + str(len(user_name_to_id))
                word = user_name_to_id[word]

            if word_id != -1:
                word_ids_to_idx[word_id] = len(token_list)
            # Create clusters of co-references
            if word_id != -1 and ref_id != -1 and ref_id in word_ids_to_idx:
                word_set = set()
                ref_set = set()
                for c in mention_clusters:
                    if word_ids_to_idx[word_id] in c:
                        word_set = c
                    if word_ids_to_idx[ref_id] in c:
                        ref_set = c
                if len(word_set) == 0 and len(ref_set) == 0:
                    c = set()
                    c.add(word_ids_to_idx[word_id])
                    c.add(word_ids_to_idx[ref_id])
                    mention_clusters.append(c)
                elif len(word_set) == 0:
                    ref_set.add(word_ids_to_idx[word_id])
                elif len(ref_set) == 0:
                    word_set.add(word_ids_to_idx[ref_id])
                elif word_set is not ref_set:
                    mention_clusters.remove(word_set)
                    mention_clusters.remove(ref_set)
                    mention_clusters.append(word_set | ref_set)

            tokens = self.tokenizer.tokenize(word)
            update_list(tokens[0], ref_id)
            was_last_line_empty = False
            tokens = tokens[1:]
            for tok in tokens:
                update_list(tok, -1)
        token_id_list = self.tokenizer.convert_tokens_to_ids(token_list)
        # tokenizer_out = self.tokenizer.encode_plus(token_list, add_special_tokens=False,padding=False,return_tensors='pt')
        ref_index_list = [word_ids_to_idx.get(ref_id, 0) for ref_id in ref_id_list]

        assert len(token_id_list) == len(ref_index_list)
        return token_id_list, ref_index_list, mention_clusters
        # return tokenizer_out['input_ids'], tokenizer_out['attention_mask'], ref_index_list


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
        ds, [train_size, val_size, test_size]
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
    get_mention_ratio(MentionDataset())
    ds = MentionDataset()
    view_ds(ds)
