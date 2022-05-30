import emotionchat_config as config

class DiscomfortAnswerer:
    def fill_slot(self, entity: list) -> str:
        """
        불,궁 대화일 때 slot filling 함수
        :param entity: 필요한 엔티티 리스트
        :return: 필요한 엔티티를 묻는 챗봇의 답변(질문)
        """

        msg = ''
        for e in entity:
            msg += self.entity_question(e)

        return msg

    def entity_question(self, unique_entity: str) -> str:
        """
        엔티티 종류별로 slot filling 질문하는 함수
        :param unique_entity: 필요한 단일 엔티티
        :return: 엔티티 종류별로 slot filling 질문 텍스트
        """
        e = unique_entity
        if e == 'BODY':
            msg = '어디가 아프신가요? '
        elif e == 'SYMPTOM':
            msg = '어떻게 아프신가요? '
        elif e == 'FOOD':
            msg = '어떤 음식이 별로세요? '
        elif e == 'PLACE':
            msg = '어디로 가고싶으세요? '
        elif e == 'LOCATION':
            msg = '어느 지역을 알려드릴까요? '
        else:
            msg = config.ANSWER['default_error_uncomfort']

        return msg

    def physicalDiscomfort_check_form(self, body: str, symptom: str) -> str:
        """
        신체 불편 호소 재질의 출력 포맷
        :param body: 신체 부위
        :param symptom: 증상
        :return: 출력 메시지
        """

        if symptom != '':
            msg = '{symptom}이 있으시군요.\n'.format(symptom=symptom)
            return msg
        msg = '{body} 많이 아프신가요?\n'.format(body=body)

        return msg

    def sleepProblem_check_form(self) -> str:
        """
        수면 문제 호소 재질의 출력 포맷
        :return: 출력 메시지
        """
        msg = '수면문제는 정말 일상생활에 큰 영향을 주는 것 같아요. 그동안 많이 피곤하셨겠어요.\n'

        return msg


    def moveHelp_check_form(self, place: str) -> str:
        """
        이동 도움 요구 재질의 출력 포맷
        :param place: 장소
        :return: 출력 메시지
        """
        msg = '{place}에 가고 싶으신가요? \n'.format(place=place)

        return msg


    def changePosture_check_form(self) -> str:
        """
        자세 변경 도움 요구 재질의 출력 포맷
        :return: 출력 메시지
        """
        msg = '자세가 많이 불편하신가요?\n'

        return msg


    def higieneAct_check_form(self) -> str:
        """
        위생 활동 도움 요구 재질의 출력 포맷
        :return: 출력 메시지
        """
        msg = '위생활동을 요청하신거죠? \n'

        return msg


    def otherAct_check_form(self) -> str:
        """
        기타 활동 도움 요구 재질의 출력 포맷
        :return: 출력 메시지
        """
        msg = '제가 도와드릴까요? \n'

        return msg

    def environmentalDiscomfort_check_form(self, place: str) -> str:
        """
        환경 불편 호소 재질의 출력 포맷
        :param place: 장소
        :return: 출력 메시지
        """
        msg = '{place}의 환경이 불편하신거죠? \n'.format(place=place)

        return msg

    def expressDesire_check_form(self) -> str:
        """
        욕구 표출 재질의 출력 포맷
        :return: 출력 메시지
        """
        msg = '제가 도와드릴까요? \n'

        return msg

    def foodDiscomfort_check_form(self, food: str) -> str:
        """
        음식 불편 호소 재질의 출력 포맷
        :param food: 음식
        :return: 출력 메시지
        """
        msg = '{food}에 문제가 있는거죠? \n'.format(food=food)

        return msg


    def discomfort_sol_form(self) -> str:
        """
        불편함 호소 해결 출력(간병인 호출) 포맷
        :return: 출력 메시지
        """

        msg = '많이 힘드셨겠어요. '
        msg += self.call_caregiver # 간병인 불러드릴까요?

        return msg
