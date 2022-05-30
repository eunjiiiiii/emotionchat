from answerer.base_answerer import BaseAnswerer
import os
import numpy as np
import torch
from model.textgeneration import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer
import emotionchat_config as config
from solution.recommend_contents import recommend_contents

class EmotionAnswerer(BaseAnswerer):

    def __init__(self):
        """
        intent==감정호소 일 때
        답변 생성 클래스
        """

        # self.root_path = 'C:/Users/R301-6/Downloads/ittp-main (2)/ittp'
        #self.root_path = '.'
        #self.data_path = f"{self.root_path}/model/textgeneration/data/wellness_dialog_for_autoregressive_train.txt"
        self.checkpoint_path = "./model/textgeneration/model"
        self.save_ckpt_path = f"{self.checkpoint_path}/kogpt2-wellnesee-auto-regressive-0504_10.pth"

        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        # 저장한 Checkpoint 불러오기
        self.checkpoint = torch.load(self.save_ckpt_path, map_location=self.device)
        self.model = DialogKoGPT2()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])

        self.model.eval()


    def generate_answer_collection(self, emotion: str, max_emotion_prob: float, text, turn_cnt: int) -> str:
        """
        답변 생성 함수
        :param text: human utterance
        :param emotion: 감정 분류 모델의 return값(6가지 감정 + None(threshold 만족 X)
        :return: chatbot response
        """
        tokenizer = get_kogpt2_tokenizer()

        output_size = 200  # 출력하고자 하는 토큰 갯수
        #fill_slot = False

        if not(max_emotion_prob < config.EMOTION['threshold']) and turn_cnt < 5:
            """
            감정도 명확히 안잡히면서 turn 수도 5회 전에 
            """

            # for i in range(5):
            sent = text  # ex) '요즘 기분이 우울한 느낌이에요'
            tokenized_indexs = tokenizer.encode(sent)

            input_ids = torch.tensor(
                [tokenizer.bos_token_id, ] + tokenized_indexs + [tokenizer.eos_token_id]).unsqueeze(0)
            # set top_k to 50
            # 답변 생성
            sample_output = self.model.generate(input_ids=input_ids)

            msg = tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:])
            msg += '[' + emotion + '(' + str(max_emotion_prob) + '%)]\n'

        elif not(max_emotion_prob < config.EMOTION['threshold']) and turn_cnt >= 5:
             # 감정의 종류 : 기쁨 분노 슬픔 놀람 불안 신뢰
            msg = '[' + emotion + '(' + str(max_emotion_prob) + '%)]\n'
            if emotion in ['기쁨', '신뢰']:
                # 긍정 메세지
                msg += config.ANSWER['default_error_end_p']

            elif emotion in ['분노', '슬픔', '불안']:
                # 부정 메세지
                msg += config.ANSWER['default_error_end_n']

            else:
                # 중립 메세지
                msg += config.ANSWER['default_error_end']

        else:
            """
            감정 명확 O turn 수 상관 X
            """
            sent = text  # ex) '요즘 기분이 우울한 느낌이에요'
            tokenized_indexs = tokenizer.encode(sent)

            input_ids = torch.tensor(
                [tokenizer.bos_token_id, ] + tokenized_indexs + [tokenizer.eos_token_id]).unsqueeze(0)
            # set top_k to 50
            # 답변 생성
            sample_output = self.model.generate(input_ids=input_ids)

            msg = '[' + emotion + '(' + str(max_emotion_prob) + '%)]\n'
            msg += tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:],
                                   skip_special_tokens=True)
            msg += recommend_contents(emotion)

        return msg

    def generate_answer_under5(self, text: str, emotion: str) -> str:
        """
        DialogKoGPT2이용한 답변 생성 함수
        감정도 명확히 안잡히면서 turn 수도 5회 전에
        :param text: human utterance
        :param emotion: 감정-주제 분류 모델의 return값(6가지 감정 + None(threshold 만족 X)
        :return: chatbot response
        """
        tokenizer = get_kogpt2_tokenizer()

        output_size = 200  # 출력하고자 하는 토큰 갯수
        # fill_slot = False

        # for i in range(5):
        sent = text  # ex) '요즘 기분이 우울한 느낌이에요'
        tokenized_indexs = tokenizer.encode(sent)

        input_ids = torch.tensor(
            [tokenizer.bos_token_id, ] + tokenized_indexs + [tokenizer.eos_token_id]).unsqueeze(0)
        # set top_k to 50
        # 답변 생성
        sample_output = self.model.generate(input_ids=input_ids)

        msg = tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:],
                               skip_special_tokens=True)

        return msg

    def generate_answer_over5(self, text: str, emotion: str) -> str:
        """
        default 감정 답변 출력 함수
        감정도 명확히 안잡히면서 turn 수 5회 초과일 때
        :param text: human utterance
        :param emotion: 감정-주제 분류 모델의 return값(6가지 감정 + None(threshold 만족 X)
        :return: chatbot response
        """

        # 감정의 종류 : 기쁨 분노 슬픔 놀람 불안 신뢰
        if emotion in ['기쁨', '신뢰']:
            # 긍정 메세지
            msg = config.ANSWER['default_error_end_p']
        elif emotion in ['분노', '슬픔', '불안']:
            # 부정 메세지
            msg = config.ANSWER['default_error_end_n']
        else:
            # 중립 메세지
            msg = config.ANSWER['default_error_end']

        return msg

    def generate_answer(self, text: str, emotion: str, max_emotion_prob) -> str:
        """
        컨텐츠 추천 답변 함수
        감정-주제 명확 O turn 수 상관 X
        :param text: human utterance
        :param emotion: 감정 분류 모델의 return값(6가지 감정 + None(threshold 만족 X)
        :return: chatbot response
        """
        tokenizer = get_kogpt2_tokenizer()

        output_size = 200  # 출력하고자 하는 토큰 갯수
        # fill_slot = False

        # for i in range(5):
        sent = text  # ex) '요즘 기분이 우울한 느낌이에요'
        tokenized_indexs = tokenizer.encode(sent)

        input_ids = torch.tensor(
            [tokenizer.bos_token_id, ] + tokenized_indexs + [tokenizer.eos_token_id]).unsqueeze(0)
        # set top_k to 50
        # 답변 생성
        sample_output = self.model.generate(input_ids=input_ids)

        msg = tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:],
                               skip_special_tokens=True)

        msg += config.ANSWER['default_error_contents']
        # 제가 기분 나아질 수 있게 컨텐츠 추천 해드려도 될까요?

        # 컨텐츠 추천 문구 추가
        msg += recommend_contents(emotion, max_emotion_prob)

        return msg