from emotionchat_engine import EmotionChat

emotionchat = EmotionChat()

if __name__ == '__main__':
    turn_cnt = 0
    conversation_history = []

    while 1:
        sent = input('User: ')
        wav_file = './exdata/' + sent + '.wav' #tnwkd
        result_dict = emotionchat.run(sent, wav_file, turn_cnt=turn_cnt)
        conversation_history.append(result_dict)

        print("Bot : " + result_dict['answer'])
        print(100 * '-')

        turn_cnt += 1


