"""
0->평온함(신뢰), 1->분노, 2->놀람, 3->감정없음, 4->기쁨, 5->불안, 6->슬픔
"""

def recommend_contents(emotion: str, max_emotion_prob) -> str:
    if emotion == '평온함':
        msg = '[https://www.youtube.com/watch?v=chX9bo2OgJ0] 평온할 때 이런 클래식을 들으면 더 기분이 좋아질 거예요. 눈을 감고 편안하게 음악을 들어보세요.'
    elif emotion == '분노':
        msg = '[https://www.youtube.com/watch?v=JzPDVLdjLsg] 화가 날 때, 어떻게 해야할 지 몰라서 감당이 안될 때, 함께 명상을 하면서 감정을 다스려봐요.'
    elif emotion == '놀람':
        msg = '[https://www.youtube.com/watch?v=qkDjMJkLxIo] 많이 놀라셨겠어요. 놀랐을 때 빠르게 숨을 몰아쉬게 되는데, 얕고 짧은 호흡은 마음을 더욱 불안하게 해요. 마음 안정화를 위해 복식훈련 호흡 훈련을 같이 해볼까요?'
    elif emotion == '기쁨':
        msg = '[https://www.youtube.com/watch?v=50WTrXSs15w] 기쁠 때 신나는 음악을 들으면 더 기뻐져요. 텐션 UP 되는 노래를 들으며 기쁜 마음을 지속해봐요!'
    elif emotion == '불안':
        msg = '[https://www.youtube.com/watch?v=GiiZDs9QRJE] 불안해서 마음이 힘드셨겠어요. 이 영상을 보면서 불안감을 떨쳐내봐요.'
    elif emotion == '슬픔':
        msg = '[https://www.youtube.com/watch?v=KQnoZ2Fyrnw] 기분전환 할 수 있게 신나는 노래를 들어보면 어떨까요? 그리고 나에게 긍정적인 말을 해봐요. 과대하고 요란하며 비현실적인 긍정은 오히려 부담을 주니 >내일은 오늘보다 조금 더 좋을지도 몰라< 같은 칭찬을 해봐요.'

    msg += '[' + emotion + '(' + str(max_emotion_prob) + '%)]\n'
    return msg
