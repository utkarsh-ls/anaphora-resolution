def parse_file(ds, file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    token_list = [ds.CLS_TOKEN]
    org_words = []
    first_token_idx = []
    ref_id_list = [-1]
    is_mention = [0]
    word_ids_to_idx = {-1: 0}
    user_name_to_id = {}
    mention_clusters = []

    def update_list(tok, ref_id, is_m):
        token_list.append(tok)
        ref_id_list.append(ref_id)
        is_mention.append(is_m)

    was_last_line_empty = True
    for line in lines:
        if not line:
            update_list(ds.SEP_TOKEN, -1, 0)
            was_last_line_empty = True
            continue
        line = line.split()
        if len(line) != 3:
            continue
        word, word_id, ref_id = line
        if (not word_id.isnumeric()) and word_id != "-":
            continue
        if (not ref_id.isnumeric()) and ref_id != "-":
            continue
        word = word.lower()
        tokens = ds.tokenizer.tokenize(word)
        if len(tokens) == 0:
            continue
        if was_last_line_empty and word == "0000":
            was_last_line_empty = False
            continue
        word_id = int(word_id) if word_id.isnumeric() else -1
        ref_id = int(ref_id) if ref_id.isnumeric() else -1
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

        first_token_idx.append(len(token_list))
        org_words.append(word)
        update_list(tokens[0], ref_id, 1 if word_id != -1 else 0)
        was_last_line_empty = False
        tokens = tokens[1:]
        for tok in tokens:
            update_list(tok, -1, 0)

    token_id_list = ds.tokenizer.convert_tokens_to_ids(token_list)
    # tokenizer_out = self.tokenizer.encode_plus(token_list, add_special_tokens=False,padding=False,return_tensors='pt')
    ref_index_list = [word_ids_to_idx.get(ref_id, 0) for ref_id in ref_id_list]

    assert len(token_id_list) == len(ref_index_list) == len(is_mention)
    return {
        "token_id_list": token_id_list,
        "ref_index_list": ref_index_list,
        "mention_clusters": mention_clusters,
        "is_mention": is_mention,
        "first_token_idx": first_token_idx,
        "org_words": org_words,
    }
    # return tokenizer_out['input_ids'], tokenizer_out['attention_mask'], ref_index_list
