# Libraries
import re
import unicodedata

import pandas as pd
from bs4 import BeautifulSoup


# Cleaner class
class Cleaner:
    def __init__(self):
        pass

    def put_line_breaks(self, text):
        text = text.replace("</p>", "</p>\n")
        return text

    def remove_html_tags(self, text):
        cleantext = BeautifulSoup(text, "lxml").text
        return cleantext

    def lower_case(self, text):
        return text.lower()

    def remove_accent(self, text):
        ascii_text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("utf-8")
            .strip()
            .replace("[^\w\s]", "")
        )
        return ascii_text

    def remove_special_characters(self, text):
        text = text.strip()
        PATTERN = r"[^a-zA-Z0-9]"  # only extract alpha-numeric characters
        text = re.sub(PATTERN, r" ", text)
        return text

    def clean(self, text):
        text = self.put_line_breaks(text)
        text = self.remove_html_tags(text)
        text = self.lower_case(text)
        text = self.remove_accent(text)
        text = self.remove_special_characters(text)
        return text

    def clean_dataset(self, path, text_column_name, label_column_name):
        df = pd.read_csv(path)
        df.columns = [text_column_name, label_column_name + "_original"]
        id2label = dict(enumerate(df[label_column_name + "_original"].unique()))
        num_labels = len(id2label.keys())
        label2id = {v: k for k, v in id2label.items()}

        df["text_cleaned"] = df[text_column_name].apply(self.clean)
        df.loc[:, [label_column_name]] = (
            df.copy().loc[:, [label_column_name]].replace(label2id)
        )
        return df, num_labels, id2label, label2id
