import emotionchat_config as config
from model.emotion.predict import IAI_EMOTION
from model.textgeneration.predict import DialogKoGPT2
#from scenario.default_scenario import sentimentDiscomfort
import os
from scenario.scenario import Scenario


class EmotionChat():

    def __init__(self):

        """
        emotionChat 답변생성 클래스 입니다.
        :param scenarios : 시나리오 리스트 (list 타입)
        """

        self.emotion_recognizer = IAI_EMOTION
        self.response_generator = DialogKoGPT2()
        self.scenario = Scenario()

    def run(self, text: str, wav_file, turn_cnt: int) -> dict:
        """
        인텐트 인식 후 단계 확인 후 시나리오에 적용해주는 함수
        모든 사용자 발화 text는 이 함수를 먼저 거쳐야 함.
        :param text: user utterance
        :return: dict
        """

        # 데모 후에 수정하기
        if "고마워" in text:
            return {
                'input': text,
                'state': 'SUCCESS',
                'emotion': None,
                'answer': config.ANSWER['default_error_end']
            }

        emotion_label, emotion_probs_array, max_emotion_prob_array = self.emotion_recognizer().predict(text, wav_file)
        max_emotion_prob = float(max_emotion_prob_array)
        emotion = self.label_emotion(emotion_label)

        result_scenario = self.scenario.apply_emotion(emotion=emotion, max_emotion_prob=max_emotion_prob, text=text, turn_cnt=turn_cnt)
        return result_scenario

    def label_emotion(self, emotion_label: int) -> str:
        if emotion_label == 0:
            emotion = '평온함'
        elif emotion_label == 1:
            emotion = '분노'
        elif emotion_label == 2:
            emotion = '놀람'
        elif emotion_label == 3:
            emotion = '감정없음'
        elif emotion_label == 4:
            emotion = '기쁨'
        elif emotion_label == 5:
            emotion = '불안'
        elif emotion_label == 6:
            emotion = '슬픔'

        return emotion

            # 다음 단계가 종료면 서버측에서 종료

    #def check_turn_cnt(self, turn_cnt):

