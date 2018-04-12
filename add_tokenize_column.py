import sys

if len(sys.argv) < 2:
    print("usage: {} <csv_file_path>".format(sys.argv[0]))
    exit(1)

import pandas as pd
from nltk.tokenize import sent_tokenize

FILE_PATH = sys.argv[1];

df = pd.read_csv(FILE_PATH)

def add_sen_tokens(document):
    sentences = sent_tokenize(document)
    sens_with_tokens = [" ".join(["<SOS>", s, "<EOS>"]) for s in sentences]
    new_document = " ".join(sens_with_tokens)
    return new_document

df['token_document'] = df.apply(lambda x: add_sen_tokens(x['document']), axis=1)
df.to_csv(FILE_PATH)
