from transformers import *
import pdb
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch
from bpe_encoder import get_codec
from gpt2_model import GPT2
from utils import parse_config

def bert_encoder():
    return BERTEncoder()

def gpt2_encoder():
    return GPT2Encoder()

def class_embedding(n_classes, embedding_dim):
    return nn.Embedding(n_classes, embedding_dim)


def unconditional(n_classes, embedding_dim):
    return nn.Embedding(n_classes, embedding_dim)


class Embedder(nn.Module):
    def __init__(self, embed_size):
        super(Embedder, self).__init__()
        self.embed_size = embed_size

    def forward(self, class_labels, captions):
        raise NotImplementedError


class BERTEncoder(Embedder):
    '''
    pretrained model used to embed text to a 768 dimensional vector
    '''

    def __init__(self):
        super(BERTEncoder, self).__init__(embed_size=768)
        self.pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.max_len = 50

    def tokenize(self, text_batch):
        text_token_ids = [
            torch.tensor(self.tokenizer.encode(string_, add_special_tokens=False, max_length=self.max_len)) for
            string_ in text_batch]
        padded_input = pad_sequence(text_token_ids, batch_first=True, padding_value=0)
        return padded_input

    def forward(self, class_labels, captions):
        '''
        :param class_labels : torch.LongTensor, class ids
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''

        padded_input = self.tokenize(captions)
        device = list(self.parameters())[0].device
        padded_input = padded_input.to(device)
        # takes the mean of the last hidden states computed by the pre-trained BERT encoder and return it
        return self.model(padded_input)[0].mean(dim=1)


class GPT2Encoder(Embedder):
    '''
    pretrained model used to embed text to a 768 dimensional vector
    '''
    def __init__(self):
        super(GPT2Encoder, self).__init__(embed_size=768)
        self.codec = get_codec()
        self.gpt2_config = parse_config()
        self.gpt2_model = GPT2(self.gpt2_config)

    def forward(self, class_labels, captions):
        '''
        :param class_labels: torch.LongTensor, class ids
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddungs of shape (batch_size, embed_size=768)
        '''
        count = 0
        for caption in captions:
            curembedding = self.gpt2_model(self.codec.encode(caption))
            curembedding = torch.mean(curembedding, dim=1)
            if count == 0:
                res = curembedding
                count += 1
            else:
                res = torch.cat((res, curembedding), dim=0)
        return res
