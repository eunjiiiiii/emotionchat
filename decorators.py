import os

import torch
from torch import Tensor, nn
import emotionchat_config as config


def backend(cls):
    for key, val in config.BASE.items():
        setattr(cls, key, val)
    return cls
# 백엔드 루트경로, 디바이스, 단어 벡터 사이즈, 미니배치 사이즈, 문장의 최대 길이(패드 시퀀싱), OS에 따른 폴더, PAD 토근값, OOV 토큰값

def data(cls):
    cls = backend(cls)
    for key, val in config.DATA.items():
        setattr(cls, key, val)
    return cls
# 학습\\검증 데이터 비율, 원본 데이터 파일 경로, out of distribution 데이터셋, 생성된 인텐트 데이터 파일 경로, 생성된 엔티티 데이터 파일 경로
# 사용자 정의 태그, NER의 BEGIN, END, INSIDE, SINGLE 태그, NER의 O태그(Outside를 의미)

def api(cls):
    cls = backend(cls)
    for key, val in config.API.items():
        setattr(cls, key, val)
    return cls

def answerer(cls):
    cls = backend(cls)
    for key, val in config.ANSWER.itmes():
        setattr(cls, key, val)

    return cls