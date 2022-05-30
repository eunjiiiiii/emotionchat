import os

bert_path = './model/emotion/KoBERT'
vocab_path = os.path.join(bert_path, 'vocab.list')
model_path = './model/emotion/model/epoch1_3704.pt'
bert_config_path = os.path.join(bert_path, 'config.json')

# text = ['당신 옷 좀 사야겠더라.']
wav_file = './model/emotion/data/clip1001_cut0.wav'
