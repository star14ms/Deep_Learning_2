import threading
import json

from modules.rnnlm_gen import BetterRnnlmGen
from modules.BotServer import BotServer
from common.util import generate_words # is_English_exist, 
import pickle

# class EnglishError(Exception):
    # def __init__(self):
    #     super().__init__('영어 감지')

def to_client(conn, addr):
    try:
        # 데이터 수신
        read = conn.recv(2048)  # 수신 데이터가 있을 때 까지 블로킹
        print('Connection from: %s' % str(addr))

        if read is None or not read:
            # 클라이언트 연결이 끊어지거나, 오류가 있는 경우
            print('클라이언트 연결 끊어짐')
            exit(0)

        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 :", recv_json_data, '\n')
        query = recv_json_data['Query']

        # 답변할 문장 생성
        try:
            # if is_English_exist(query): 
                # raise EnglishError
            
            answer = generate_words(query, model, kkma, morp_to_id, id_to_morp)
            model.reset_state()
            if answer is None: 
                raise ValueError
            print(answer)
        
        # except EnglishError:
        #     answer = "나 영알못이야 ㅋㅋ"
        #     print(answer)

        except Exception:
            answer = "미안, 무슨 말인지 모르겠어 ㅋㅋ 더 공부 할게"
            print(answer)
        
        send_json_data_str = {
            "Query" : query,
            "Answer": answer,
            "AnswerImageUrl" : None,
        }
        message = json.dumps(send_json_data_str)
        conn.send(message.encode())
        print('=' * 50)

    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    port = 5050
    listen = 10

    load_model_pkl = 'ln_25600000 lt_45h40m38s ppl_65.0 BetterRnnlm params'
    morp_to_id_pkl = 'saved_pkls/YT_cmts_morps_to_id_Kkma.pkl'

    with open(morp_to_id_pkl, 'rb') as f:
        (_, morp_to_id, id_to_morp) = pickle.load(f)
    vocab_size = len(id_to_morp)

    model = BetterRnnlmGen(vocab_size)
    model.load_params(load_model_pkl)

    # 봇 서버 동작
    bot = BotServer(port, listen)
    bot.create_sock()
    print("문장 생성봇 가동!")
    print('=' * 50)
    
    while True:
        conn, addr = bot.ready_for_client()
        to_client(conn, addr)

        # client = threading.Thread(target=to_client, args=(
        #     conn,
        #     addr,
        # ))
        # client.start()
