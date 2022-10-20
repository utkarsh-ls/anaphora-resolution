import os
import numpy as np
import torch
from transformers import BertTokenizer


class mLangDataset(torch.utils.data.Dataset):
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MAX_SEQ_LEN = 410

    def __init__(self, ds_folder="ds", include_lang=["eng"], pad=True):
        super().__init__()
        self.pad = pad
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
            # "google/muril-base-cased"
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
        for i in range(len(self)):
            tok_list, ref_idx_list = self._parse_file(i)
            token_lists.append(tok_list)
            ref_idx_lists.append(ref_idx_list)

        tokenizer_out = self.tokenizer.batch_encode_plus(
            token_lists,
            add_special_tokens=False,
            is_split_into_words=True,
            padding=self.pad,
            return_tensors="pt",
        )
        self.token_id_lists = tokenizer_out['input_ids']
        self.MAX_SEQ_LEN = len(self.token_id_lists[0])
        self.mask_lists = tokenizer_out['attention_mask']
        # for i in range(len(token_lists)):
        #     if len(token_lists[i]) == self.MAX_SEQ_LEN-2:
        #         print(i, self.files[i])
        if self.pad:
            for i in range(len(self.files)):
                ref_idx_lists[i] += [0]*(self.MAX_SEQ_LEN - len(ref_idx_lists[i]))

        self.ref_idx_lists = torch.as_tensor(ref_idx_lists,dtype=torch.long)
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
            self.ref_idx_lists[idx],
        )

    def _parse_file(self, index):
        with open(self.files[index]) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        token_list = [self.CLS_TOKEN]
        ref_id_list = [-1]
        word_ids_to_idx = {-1: 0}
        user_name_to_id = {}

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
            word= word.lower()
            # if ref_id == 8:
            # print(line, self.files[index], flush=True)
            if word[0] == '@':
                if word not in user_name_to_id:
                    user_name_to_id[word] = "user"+str(len(user_name_to_id))
                word = user_name_to_id[word]
                
            if word_id != -1:
                word_ids_to_idx[word_id] = len(token_list)
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
        return token_id_list, ref_index_list
        # return tokenizer_out['input_ids'], tokenizer_out['attention_mask'], ref_index_list


def get_dataloaders(ds_cls, config):
    ds = ds_cls(include_lang=["eng"])
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
    return train_loader, val_loader, test_loader, ds.MAX_SEQ_LEN

def view_ds(ds:mLangDataset):
    for tok_id_list, mask, ref_idx_list in ds:
        tok_list= ds.tokenizer.convert_ids_to_tokens(tok_id_list)
        tok_list= list(filter(lambda tok: tok!='[PAD]', tok_list))
        print(tok_list)


if __name__ == "__main__":
    ds = mLangDataset("ds/", include_lang=["eng"])
    print(len(ds))
    mx = 0
    # for i in range(len(ds)):
    #     # print(i,ds.files[i], end='\n')
    #     print(i, end="\r")
    #     b = ds[i]
    #     mx = max(len(b[0]), mx)
    # print()
    # print(mx)
    print(ds[0][0].size(), ds[0][1].size())
    view_ds(ds)
