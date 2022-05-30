import torch
from pydub import AudioSegment
import model.emotion.config as config
import numpy as np
import librosa
import os
import html
import re
import json
from transformers import BertConfig, BertModel
import torch.nn as nn
from model.emotion.model import MultimodalTransformer
from torchaudio.transforms import MFCC
from model.emotion.KoBERT.tokenization import BertTokenizer

class IAI_EMOTION:
    def __init__(self):
        self.only_audio = False
        self.only_text = False
        self.n_classes = 7
        # self.logging_steps = 10
        # self.seed = 1
        # self.num_workers = 4
        # self.cuda = 'cuda:0'
        # self.attn_dropout = 0.3
        # self.relu_dropout = 0.3
        # self.emb_dropout = 0.3
        # self.res_dropout = 0.3
        # self.out_dropout = 0.3
        self.n_layers = 2
        self.d_model = 100
        self.n_heads = 2
        self.attn_mask = True
        # self.lr = 2e-05
        # self.epochs = 1
        # self.batch_size = 64
        # self.clip = 0.8
        # self.warmup_percent = 0.1
        self.max_len_audio = 400
        self.sample_rate = 48000
        self.resample_rate = 16000
        self.n_fft_size = 400
        self.n_mfcc = 40
        self.max_len_bert = 64
        self.tokenizer = BertTokenizer.from_pretrained(config.vocab_path, do_lower_case=False)
        self.vocab = self.tokenizer.vocab
        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

        self.model = MultimodalTransformer(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                n_classes=self.n_classes,
                only_audio=self.only_audio,
                only_text=self.only_text,
                d_audio_orig=self.n_mfcc,
                d_text_orig=768,  # BERT hidden size
                d_model=self.d_model,
                attn_mask=self.attn_mask
            ).cuda()
        self.model.load_state_dict(torch.load(config.model_path))
        self.model.eval()
        self.model.zero_grad()
        self.bert_config_path = os.path.join(config.bert_path, 'config.json')
        self.bert = BertModel(BertConfig(vocab_size=30797, **self.load_json(self.bert_config_path))).cuda()
        self.bert_model_path = os.path.join(config.bert_path, 'model.bin')
        self.bert.load_state_dict(self.clean_state_dict(torch.load(self.bert_model_path)), strict=False)
        self.bert.eval()
        self.bert.zero_grad()
        self.audio2mfcc = MFCC(
            sample_rate = self.resample_rate,
            n_mfcc = self.n_mfcc,
            log_mels = False,
            melkwargs = {'n_fft': self.n_fft_size}
        ).cuda()
    # noinspection PyMethodMayBeStatic
    def extract_audio_array(self, wav_file):
        audio = AudioSegment.from_wav(wav_file)
        audio = audio.set_channels(1)
        audio = audio.get_array_of_samples()

        return np.array(audio).astype(np.float32)

    # noinspection PyMethodMayBeStatic
    def _trim(self, audio):
        left, right = None, None
        for idx in range(len(audio)):
            if np.float32(0) != np.float32(audio[idx]):
                left = idx
                break
        for idx in reversed(range(len(audio))):
            if np.float32(0) != np.float32(audio[idx]):
                right = idx
                break
        return audio[left:right + 1]

    def pad_with_mfcc(self, wav_file):
        max_len = self.max_len_audio
        audio_array = torch.zeros(len([wav_file]), self.n_mfcc, max_len).fill_(float('-inf'))
        for idx, audio in enumerate([wav_file]):
            # resample and extract mfcc
            audio = librosa.core.resample(audio, self.sample_rate, self.resample_rate)
            mfcc = self.audio2mfcc(torch.tensor(self._trim(audio)).cuda())

            # normalize
            cur_mean, cur_std = mfcc.mean(dim=0), mfcc.std(dim=0)
            mfcc = (mfcc - cur_mean) / cur_std

            # save the extracted mfcc
            cur_len = min(mfcc.shape[1], max_len)
            audio_array[idx, :, :cur_len] = mfcc[:, :cur_len]

        # (batch_size, n_mfcc, seq_len) -> (batch_size, seq_len, n_mfcc)
        padded = audio_array.transpose(2, 1)

        # get key mask
        key_mask = padded[:, :, 0]
        key_mask = key_mask.masked_fill(key_mask != float('-inf'), 0)
        key_mask = key_mask.masked_fill(key_mask == float('-inf'), 1).bool()

        # -inf -> 0.0
        padded = padded.masked_fill(padded == float('-inf'), 0.)
        return padded, key_mask

    # noinspection PyMethodMayBeStatic
    def _add_special_tokens(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def pad_with_text(self, sentence, max_len):
        sentence = self._add_special_tokens(sentence)
        diff = max_len - len(sentence)
        if diff > 0:
            sentence += [self.pad_idx] * diff
        else:
            sentence = sentence[:max_len - 1] + [self.sep_idx]
        return sentence

    # noinspection PyMethodMayBeStatic
    def clean_state_dict(self, state_dict):
        new = {}
        for key, value in state_dict.items():
            if key in ['fc.weight', 'fc.bias']:
                continue
            new[key.replace('bert.', '')] = value
        return new

    # noinspection PyMethodMayBeStatic
    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    # def load_bert(self, bert_path):
    #     bert_config_path = os.path.join(config.bert_path, 'config.json')
    #     bert = BertModel(BertConfig(vocab_size=30797, **self.load_json(bert_config_path))).cuda()
    #     bert_model_path = os.path.join(bert_path, 'model.bin')
    #     bert.load_state_dict(self.clean_state_dict(torch.load(bert_model_path)), strict=False)
    #     return bert

    # noinspection PyMethodMayBeStatic
    def normalize_string(self, s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s

    # noinspection PyMethodMayBeStatic
    def tokenize(self, tokens):
        return self.tokenizer.tokenize(tokens)

    def predict(self, text, wav_file):
        audio = self.extract_audio_array(wav_file)
        with torch.no_grad():
            max_len = self.max_len_bert
            tokenize_text = []

            tokens = self.normalize_string(text)
            tokens = self.tokenize(tokens)
            tokenize_text.append(self.tokenizer.convert_tokens_to_ids(tokens))
            input_ids = torch.tensor([self.pad_with_text(sent, max_len) for sent in tokenize_text]).cuda()
            text_masks = torch.ones_like(input_ids).masked_fill(input_ids == self.pad_idx, 0).bool().cuda()
            text_emb = self.bert(input_ids, text_masks)[0]
            # text_emb = self.bert(input_ids, text_masks)['last_hidden_state']
            audio_emb, audio_mask = self.pad_with_mfcc(audio)
            audio_emb = audio_emb.cuda()
            audio_mask = audio_mask.cuda()
            logit, hidden = self.model(audio_emb, text_emb, audio_mask, torch.logical_not(text_masks))
            softmax_layer = nn.Softmax(-1)
            softmax_result = softmax_layer(logit)
            max_emotion = max(t[0] for t in softmax_result)
            y_pred = logit.max(dim=1)[1]
            emotion_pred = y_pred.detach().cpu().numpy()[0]
            emotion_prob = softmax_result.detach().cpu().numpy()
            max_emotion_prob = max_emotion.detach().cpu().numpy()

        return emotion_pred, emotion_prob, max_emotion_prob
