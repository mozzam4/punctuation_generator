import clean_evaluate_data as ced
import pandas as pd
import torch
from torchtext import data
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.optim as optim


from torchtext import datasets

import spacy
import numpy as np

import time
import random

from torchnlp.datasets import iwslt_dataset  # doctest: +SKIP
import pandas as pd
import torch
from torchtext import data
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors
import nltk
nltk.download('punkt')

from torchnlp.datasets import iwslt_dataset  # doctest: +SKIP


def tag_sentence(model, device, sentence, text_field, tag_field):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token for token in sentence]

    if text_field.lower:
        tokens = [t.lower() for t in tokens]

    numericalized_tokens = [text_field.vocab.stoi[t] for t in tokens]

    unk_idx = text_field.vocab.stoi[text_field.unk_token]

    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]

    token_tensor = torch.LongTensor(numericalized_tokens)

    token_tensor = token_tensor.unsqueeze(-1).to(device)

    predictions = model(token_tensor)

    top_predictions = predictions.argmax(-1)

    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]

    return tokens, predicted_tags, unks


class BiLSTMPOSTagger(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]

        return predictions


TEXT = Field(lower=True, dtype=torch.float)
LABEL = Field(unk_token=None)
tv_datafields = [("Sequence", TEXT), ("Labels", LABEL)]


with open('/home/bbb/dev/mozzam/interm_files/text_without_disfluency.txt', 'r') as file:
    sentence = file.read()
list_of_sent = []
list_of_sent = ced.clean(sentence)


vectors = Vectors(name='glove.6B.100d.txt', cache='/home/bbb/dev/mozzam/punctuation_generator')

MIN_FREQ = 2

trn = TabularDataset(
           path='/home/bbb/dev/mozzam/punctuation_generator/' + 'train_data1.csv',# the file path
           format='csv',
           skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
           fields=tv_datafields)
TEXT.build_vocab(trn,
                 min_freq = MIN_FREQ,
                 vectors = vectors,
                 unk_init = torch.Tensor.normal_)
LABEL.build_vocab(trn)

BATCH_SIZE = 128
device = torch.device('cpu')

#device = torch.cuda.current_device()

INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(LABEL.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = BiLSTMPOSTagger(INPUT_DIM,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        OUTPUT_DIM,
                        N_LAYERS,
                        BIDIRECTIONAL,
                        DROPOUT,
                        PAD_IDX)

model.load_state_dict(torch.load('/home/bbb/dev/mozzam/punctuation_generator/model.pt', map_location=torch.device('cpu')))

output_text = ''

for batch in list_of_sent:
    tokens, pred_tags, unks = tag_sentence(model,
                                       device,
                                       batch,
                                       TEXT,
                                       LABEL)

    for token, pred_tag in zip(tokens, pred_tags):
        if pred_tag == '0':
            output_text = output_text + ' ' + token
        elif pred_tag == '1':
            output_text = output_text + ' ' + token + '.' + '\n'
        elif pred_tag == '2':
            output_text = output_text + ' ' + token + ',' + '\n'
        elif pred_tag == '3':
            output_text = output_text + ' ' + token + '?' + '\n' 
        else:
            output_text = output_text + ' ' + token + '!' + '\n'

with open("/home/bbb/dev/bigbluebutton_mozzam/bigbluebutton-html5/public/final_output.txt", "w") as text_file:
    text_file.write(output_text)
