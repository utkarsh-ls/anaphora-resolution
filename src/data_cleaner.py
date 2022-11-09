from cleantext import clean
import re
import os

# from natsort import natsorted
import numpy as np
from tqdm import tqdm

DATA = "../data/original"


def remove_emojis(data):
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", data)


if __name__ == "__main__":
    data_dirs = os.listdir(DATA)
    data_dirs = ["mal_train_files"]
    for data_dir in data_dirs:
        DATA_PATH = os.path.join(DATA, data_dir)
        NEW_DATA_PATH = os.path.join(DATA, "../clean/", data_dir)

        os.makedirs(NEW_DATA_PATH, exist_ok=True)

        files = sorted(os.listdir(DATA_PATH))
        for file in files:
            file_path = os.path.join(DATA_PATH, file)
            f = open(file_path, "r")
            write_file_path = os.path.join(NEW_DATA_PATH, file)
            fw = open(write_file_path, "w")

            data = f.readlines()

            for line in data:
                split = line.split()
                if len(split) == 0:
                    fw.write(line)
                    continue

                line_start = " ".join(line.split()[:-2])
                line_end = "\t".join(line.split()[-2:])
                # print("START:", [line_start], len(line_start))
                # print("END:", [line_end])

                clean_line_start = clean(
                    line_start,
                    to_ascii=False,
                    no_line_breaks=True,
                    no_urls=True,
                    no_emails=True,
                    no_phone_numbers=True,
                    no_digits=False,
                    no_currency_symbols=True,
                    # no_punct=True,
                    replace_with_email="email",
                    replace_with_currency_symbol="$",
                    replace_with_phone_number="number",
                    no_emoji=True,
                )
                clean_line_start = remove_emojis(clean_line_start)
                if not clean_line_start:
                    continue
                if clean_line_start[0] in ['"', "'", "&"]:
                    clean_line_start = clean_line_start[1:]
                if not clean_line_start or clean_line_start in ['"', "'", "&"]:
                    continue

                if clean_line_start.isnumeric() and len(clean_line_start) > 4:
                    clean_line_start = "0000"

                fw.write(clean_line_start + "\t" + line_end + "\n")
            fw.close()


# # after doing line.split() we should ideally get 3 strs but we get more than that too sometimes handle that
#  dont clean the last two elements of line.split() (they are labels of nouns and pronouns)
#  change any number of length more than 4 to 0000
#  URL =>
