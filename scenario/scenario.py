# Copyright 2020 emotionchat. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
from collections import Callable
from copy import deepcopy
from random import randint
from decorators import data
import emotionchat_config as config
from answerer.emotion_answerer import EmotionAnswerer
from answerer.discomfort_answerer import DiscomfortAnswerer


@data
class Scenario:
    """
    시나리오 객체
    불,궁,감의 여러 상황(slot filling, slot 다 채워진 경우, turn_cnt < 5 등..)에서의
    시나리오를 구분하고, 알맞은 답변을 answer에 저장한다.
    """

    def apply_emotion(self, emotion: str, max_emotion_prob: float, text, turn_cnt: int) -> dict:

        if (max_emotion_prob < config.EMOTION['threshold']) and turn_cnt < 5:
            # 1. 감정이 명확히 분류되지 않은 경우 & turn 수 5회 미만
            return {
                'input': text,
                'state': 'REQUIRE_EMOTION_TOPIC',
                'emotion': emotion,
                'max_emotion_prob': max_emotion_prob,
                'answer': EmotionAnswerer().generate_answer_under5(text, emotion)
            }

        elif (max_emotion_prob < config.EMOTION['threshold']) and turn_cnt >= 5:
            # 2. 감정이 명확히 분류되지 않은 경우 & turn 수 5회 이상
            return {
                'input': text,
                'state': 'FAIL_EMOTION_TOPIC',
                'emotion': emotion,
                'answer': EmotionAnswerer().generate_answer_over5(text, emotion),
            }

        else:
            # 3. 감정이 명확히 분류된 경우
            return {
                'input': text,
                'state': 'SUCCESS',
                'emotion': emotion,
                'answer': EmotionAnswerer().generate_answer(text, emotion, max_emotion_prob),
            }

    def apply_emotion_under5(self, emotion: str, topic: str, text: str) -> dict:
        """
        1. 감정이 명확히 분류되지 않은 경우 & turn 수 5회 미만
        :param emotion: 감정
        :param text: user utterance
        :return: dict
        """
        return{
            'input': text,
            'state': 'REQUIRE_EMOTION_TOPIC',
            'emotion': emotion,
            'answer': EmotionAnswerer().generate_answer_under5(text, emotion)
        }

    def apply_emotion_over5(self, emotion: str, topic: str, text: str) -> dict:
        """
        2. 감정이 명확히 분류되지 않은 경우 & turn 수 5회 이상
        :param emotion: 감정
        :param text: user utterance
        :return: dict
        """
        return {
            'input': text,
            'state': 'FAIL_EMOTION_TOPIC',
            'emotion': emotion,
            'answer': EmotionAnswerer().generate_answer_over5(text, emotion)
        }

    def apply_emotion_success(self, emotion: str, topic: str, text: str) -> dict:
        """
        3. 감정이 명확히 분류된 경우
        :param emotion: 감정
        :param text: user utterance
        :return: dict
        """
        return {
            'input': text,
            'state': 'SUCCESS',
            'emotion': emotion,
            'answer': EmotionAnswerer().generate_answer(text, emotion)
        }