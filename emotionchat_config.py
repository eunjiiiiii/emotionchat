import os
import platform
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

root_dir = os.path.abspath(os.curdir)
# 만약 로딩이 안된다면 root_dir을 직접 적어주세요.
# 데모 기준에서 OS별 root path는 아래와 같이 적으면 됩니다.
# windows : C:Users/yourname/yourdirectory/emotionchat/demo
# linux : /home/yourname/yourdirectory/emotionchat/demo

_ = '\\' if platform.system() == 'Windows' else '/'
if root_dir[len(root_dir) - 1] != _: root_dir += _

human_name = '혜영'
bot_name = '마음결'

"""
    phase명 정리
    '/welcomemsg_chat': 인사,
    '/other_user': 넋두리,
    '/induce_ucs': 불궁감유도,
    '/recognize_uc_chat': 불궁대화인식,
    '/recognize_emotion_chat': 감정대화인식,
    '/recognize_uc': (확실한) 불궁인식,
    '/generate_emotion_chat': 생성모델을 통한 챗봇 대화,
    '/recognize_emotion': (확실한) 감정인식,
     /recognize_topic: (확실한) 주제 인식,
    '/check_ucs': 확인용 재질의,
    '/check_ucs_positive': 확인용 재질의 긍정,
    '/check_ucs_negative':  확인용 재질의 부정,
    '/recommend_contents': (감정)컨텐츠제공,
    '/call_caregiver': (불편함)해결(간병인 호출),
    '/solve':  (궁금함)해결,
    '/end_chat':  작별인사
"""

BASE = {
    'root_dir': root_dir.format(_=_),  # 백엔드 루트경로
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'vector_size': 128,  # 단어 벡터 사이즈
    'batch_size': 512,  # 미니배치 사이즈
    'max_len': 8,  # 문장의 최대 길이 (패드 시퀀싱)
    'delimeter': _,  # OS에 따른 폴더 delimeter

    'PAD': 0,  # PAD 토큰 값 (전체가 0인 벡터)
    'OOV': 1  # OOV 토큰 값 (전체가 1인 벡터)
}

DATA = {
    'data_ratio': 0.8,  # 학습\\검증 데이터 비율
    'raw_data_dir': BASE['root_dir'] + "data{_}raw{_}".format(_=_),  # 원본 데이터 파일 경로
    'ood_data_dir': BASE['root_dir'] + "data{_}ood{_}".format(_=_),  # out of distribution 데이터셋
    'intent_data_dir': BASE['root_dir'] + "data{_}intent_data.csv".format(_=_),  # 생성된 인텐트 데이터 파일 경로
    'entity_data_dir': BASE['root_dir'] + "data{_}entity_data.csv".format(_=_),  # 생성된 엔티티 데이터 파일 경로

    'NER_categories': ['DATE', 'LOCATION', 'PLACE',
                       'BODY', 'SYMPTOM', 'FOOD',
                       'EMOTION', 'MAX_EMOTION_PROB', 'TOPIC', 'MAX_TOPIC_PROB', 'TEXT', 'TURN_CNT'],  # 사용자 정의 태그
    'NER_tagging': ['B', 'E', 'I', 'S'],  # NER의 BEGIN, END, INSIDE, SINGLE 태그
    'NER_outside': 'O',  # NER의 O태그 (Outside를 의미)
}


API = {
    'request_chat_url_pattern': 'request_chat',  # request_chat 기능 url pattern
    'fill_slot_url_pattern': 'fill_slot',  # fill_slot 기능 url pattern
    'get_intent_url_pattern': 'get_intent',  # get_intent 기능 url pattern
    'get_entity_url_pattern': 'get_entity'  # get_entity 기능 url pattern
}

ANSWER = {
    # 고정된 만남 안내 메세지
    'welcomemsg_chat' : '안녕하세요. 저는 {HUMAN_NAME}님의 \n심리 상담을 도와드릴 {BOT_NAME} 입니다.\n\n상담 시작 전, 스피커 음량을 확인해 주세요.\n{BOT_NAME}이 문자와 소리 모두 제공합니다.'.format(HUMAN_NAME=human_name, BOT_NAME=bot_name),
    # 간병인 호출
    'call_caregiver': '간병인 불러드릴게요 \n',

    'fallback': "죄송해요. 제가 이해하지 못했어요.\n 다시 한 번 말씀해주실래요?\n",
    'default_error': '무슨 일 있으신가요?\n',
    'default_error_uncomfort': '다른 불편하신 점은 없으신가요?\n',
    'default_error_curious': '다른 궁금하신 점은 없으신가요?\n',
    'default_error_uc': '그러시군요. 다른 불편하신 점이나 궁금하신 점은 없으신가요?\n',
    'default_error_end_n' : '그러시군요.\n또 기분이 안 좋아지면 언제든 저에게 이야기해주세요\n',
    'default_error_end_p' : '그러시군요.\n또 기분 좋은 일 생기시면 언제든 저에게 이야기해주세요\n',
    'default_error_end': '그러시군요. 다음에 또 불러주세요\n',
    'default_error_contents': '\n{HUMAN_NAME}님의 심리 상태를 이해했습니다\n마음을 다스릴 수 있는 좋은 글과 소리를 제공해 드릴게요. 한 번 보시겠어요?\n'.format(HUMAN_NAME=human_name),
    'default_check_emotion': '그러시군요.\n혹시 더 마음 쓰이는 일은 없으셨을까요?\n'
}

SORT_INTENT = {
    'QURIOUS': ['weather', 'dust', 'restaurant', 'travel'],
    'PHISICALDISCOMFORT' : ['기타활동요구', '욕구표출', '위생활동요구', '환경불편호소', '수면문제호소', '신체불편호소', '이동도움요구', '음식불편호소', '자세변경요구'],
    'PHSICALDISCOMFORTnQURIOUS': ['기타활동요구', '욕구표출', '위생활동요구', '환경불편호소', '수면문제호소', '신체불편호소', '이동도움요구', '음식불편호소', '자세변경요구',
                    'weather', 'dust', 'restaurant', 'travel'],
    'SENTIMENTDISCOMFORT': ['마음상태호소']
}

PHASE_INTENT = {
    '/welcomemsg_chat': [''],
    '/other_user': [''],
    '/induce_ucs': [''],
    '/recognize_uc_chat': [],
    '/recognize_emotion_chat': [],
    '/recognize_uc': [],
    '/generate_emotion_chat': [],
    '/recognize_emotion': [],
    '/check_ucs': [],
    '/check_ucs_positive': [],
    '/check_ucs_negative': [],
    '/recommend_contents': [],
    '/call_caregiver': [],
    '/solve': [],
    '/end_chat': []
}

# 해당 단계의 예상 단계를 config에 미리 저장해놓는게 나을지, 아님 이전 단계의 다음 예상 단계를 저장해놓는게 나을지
# 해당 단계의 예상 단계를 config에 미리 저장해놓자!

PRED_PHASE = {
    '/welcomemsg_chat': ['other_user', '/recognize_uc_chat', '/recognize_emotion_chat',
                              '/recognize_uc'],
    '/other_user': ['/induce_ucs', '/recongnize_uc_chat', '/recongnize_emotion_chat',
                               '/recognize_uc', '/recognize_emotion', '/recognize_topic',
                               '/check_ucs'],
    '/induce_ucs': ['other_user', '/recognize_uc_chat', '/recognize_emotion_chat',
                              '/recognize_uc'],
    '/recognize_uc_chat': ['/recognize_uc', '/fill_slot'],
    '/recognize_emotion_chat': ['/generate_emotion_chat'],
    '/fill_slot': ['/fill_slot','/recognize_uc'],
    '/recognize_uc': ['/check_ucs'],
    '/generate_emotion_chat': ['/recognize_emotion','/generate_emotion_chat'],
    '/recognize_emotion': ['/check_ucs'],
    '/check_ucs': ['/check_ucs_positive', '/check_ucs_negative'],
    '/check_ucs_positive': ['/call_caregiver', '/solve', '/recommend_contents'],
    '/check_ucs_negative': ['/end_chat', '/check_ucs'],
    '/recommend_contents': ['/end_chat'],
    '/call_caregiver': ['/end_chat'],
    '/solve': ['/end_chat'],
    '/end_chat': []
}

EMOTION = {
    'threshold': 0.02
}

TOPIC = {
    'threshold': 0.0005
}