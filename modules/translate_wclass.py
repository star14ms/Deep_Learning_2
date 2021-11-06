
komoran_word_class_lv1 = {
    'NNG': '체언',
    'NNP': '체언',
    'NNB': '체언',
    'NP': '체언',
    'NR': '체언',
    
    'VV': '용언',
    'VA': '용언',
    'VX': '용언',
    'VCP': '용언',
    'VCN': '용언',
    
    'MM': '수식언', 
    'MAG': '수식언',
    'MAJ': '수식언',
    
    'IC': '독립언',
    
    'JKS': '관계언',
    'JKC': '관계언',
    'JKG': '관계언',
    'JKO': '관계언',
    'JKB': '관계언',
    'JKV': '관계언',
    'JKQ': '관계언',
    'JX': '관계언',
    'JC': '관계언',
    
    'EP': '의존형태',
    'EF': '의존형태',
    'EC': '의존형태',
    'ETN': '의존형태',
    'ETM': '의존형태',
    'XPN': '의존형태',
    'XSN': '의존형태',
    'XSV': '의존형태',
    'XSA': '의존형태',
    'XR': '의존형태',
    
    # 기호
    'SF': '기호',
    'SP': '기호',
    'SS': '기호',
    'SE': '기호',
    'SO': '기호',
    'SL': '기호',
    'SH': '기호',
    'SW': '기호',
    'NF': '기호',
    'NV': '기호',
    'SN': '기호',
    'NA': '기호',
}

komoran_word_class_lv2 = {

    # 체언
    'NNG': '명사', # NN 명사
    'NNP': '명사',
    'NNB': '명사',
    'NP': '대명사', # NP 대명사
    'NR': '수사', # (NR 수사)
    
    # 용언
    'VV': '동사',
    'VA': '형용사',
    'VX': '보조용언',
    'VCP': '지정사', # VC 지정사
    'VCN': '지정사',
    
    # 수식언
    'MM': '관형사', 
    'MAG': '부사', # MA 부사
    'MAJ': '부사',
    
    # 독립언
    'IC': '감탄사', 
    
    # 관계언
    'JKS': '격조사', # JK 격조사
    'JKC': '격조사',
    'JKG': '격조사',
    'JKO': '격조사',
    'JKB': '격조사',
    'JKV': '격조사',
    'JKQ': '격조사',
    'JX': '보조사',
    'JC': '접속조사',
    
    # 의존 형태
    'EP': '어미', # E 어미
    'EF': '어미',
    'EC': '어미',
    'ETN': '어미',
    'ETM': '어미',
    'XPN': '접두사', # XP 접두사
    'XSN': '접미사', # XS 접미사
    'XSV': '접미사',
    'XSA': '접미사',
    'XR': '어근',
    
    # 기호
    'SF': '마침표,물음표,느낌표',
    'SP': '쉼표,가운뎃점,콜론,빗금',
    'SS': '따옴표,괄호표,줄표',
    'SE': '줄임표',
    'SO': '붙임표(물결,숨김,빠짐)',
    'SL': '외국어',
    'SH': '한자',
    'SW': '기타기호(논리수학,화폐)',
    'NF': '명사추정범주',
    'NV': '용언추정범주',
    'SN': '숫자',
    'NA': '분석불능범주',
}

komoran_word_class_lv3 = {
    
    # 체언
    'NNG': '일반명사', # NN 명사
    'NNP': '고유명사',
    'NNB': '의존명사',
    
    'NP': '대명사', # NP 대명사
    'NR': '수사',
    
    # 용언
    'VV': '동사',
    'VA': '형용사',
    'VX': '보조용언',
    
    'VCP': '긍정지정사', # VC 지정사
    'VCN': '부정지정사',
    
    # 수식언
    'MM': '관형사', 
    
    'MAG': '일반부사', # MA 부사
    'MAJ': '접속부사',
    
    # 독립언
    'IC': '감탄사', 
    
    # 관계언
    'JKS': '주격조사', # JK 격 조사
    'JKC': '보격조사',
    'JKG': '관형격조사',
    'JKO': '목적격조사',
    'JKB': '부사격조사',
    'JKV': '호격조사',
    'JKQ': '인용격조사',
    
    'JX': '보조사',
    'JC': '접속조사',
    
    # 의존 형태
    'EP': '선어말어미', # E 어미
    'EF': '종결어미',
    'EC': '연결어미',
    'ETN': '명사형전성어미',
    'ETM': '관형형전성어미',
    
    'XPN': '체언접두사', # XP 접두사
    
    'XSN': '명사파생접미사', # XS 접미사
    'XSV': '동사파생접미사',
    'XSA': '형용사파생접미사',
    
    'XR': '어근',
    
    # 기호
    'SF': '마침표,물음표,느낌표',
    'SP': '쉼표,가운뎃점,콜론,빗금',
    'SS': '따옴표,괄호표,줄표',
    'SE': '줄임표',
    'SO': '붙임표(물결,숨김,빠짐)',
    'SL': '외국어',
    'SH': '한자',
    'SW': '기타기호(논리수학,화폐)',
    'NF': '명사추정범주',
    'NV': '용언추정범주',
    'SN': '숫자',
    'NA': '분석불능범주',
}

komoran_word_class = [komoran_word_class_lv1, komoran_word_class_lv2, komoran_word_class_lv3]


kkma_word_class_lv1 = {
    
    # N 체언
    'NNG': '체언', # NN 명사
    'NNP': '체언',
    'NNB': '체언',
    'NNM': '체언',
    
    'NR': '체언',

    'NP': '체언',

    # V 용언
    'VV': '용언',
    'VA': '용언',

    'VXV': '용언', # VX 보조 용언
    'VXA': '용언',
    
    'VCP': '용언', # VC 지정사
    'VCN': '용언',
    
    # M 수식언
    'MDT': '수식언', # MD 관형사
    'MDN': '수식언',
    
    'MAG': '수식언', # MA 부사
    'MAC': '수식언',
    
    # I 감탄사
    'IC': '감탄사', 
    
    # J 조사
    'JKS': '조사', # JK 격 조사
    'JKC': '조사',
    'JKG': '조사',
    'JKO': '조사',
    'JKM': '조사',
    'JKI': '조사',
    'JKQ': '조사',
    
    'JX': '조사',

    'JC': '조사',
    
    # E 어말 어미
    'EPH': '어미', # EP 선어말 어미
    'EPT': '어미', 
    'EPP': '어미', 

    'EFN': '어미', # EF 종결 어미
    'EFQ': '어미',
    'EFO': '어미',
    'EFA': '어미',
    'EFI': '어미',
    'EFR': '어미',

    'ECE': '어미', # EC 연결 어미
    'ECD': '어미',
    'ECS': '어미',

    'ETN': '어미', # ET 전성 어미
    'ETD': '어미',
    
    'EMO': '이모티콘', # +추가
    
    # X 어근,접사
    'XPN': '어근,접사', # XP 접두사
    'XPV': '어근,접사',
    
    'XSN': '어근,접사', # XS 접미사
    'XSV': '어근,접사',
    'XSA': '어근,접사',
    'XSM': '어근,접사',
    'XSO': '어근,접사',
    
    'XR': '어근,접사',
    
    # S 부호
    'SF': '부호',
    'SP': '부호',
    'SS': '부호',
    'SE': '부호',
    'SO': '부호',
    'SW': '부호',

    # U 분석 불능
    'UN': '분석 불능',
    'UV': '분석 불능',
    'UE': '분석 불능',

    # O 한글 이외
    'OL': '한글 이외',
    'OH': '한글 이외',
    'ON': '한글 이외',
}

kkma_word_class_lv2 = {
    
    # N 체언
    'NNG': '명사', # NN 명사
    'NNP': '명사',
    'NNB': '명사',
    'NNM': '명사',
    
    'NR': '수사',

    'NP': '대명사',

    # V 용언
    'VV': '동사',
    'VA': '형용사',

    'VXV': '보조 용언', # VX 보조 용언
    'VXA': '보조 용언',
    
    'VCP': '지정사', # VC 지정사
    'VCN': '지정사',
    
    # M 수식언
    'MDT': '관형사', # MD 관형사
    'MDN': '관형사',
    
    'MAG': '부사', # MA 부사
    'MAC': '부사',
    
    # I 감탄사
    'IC': '감탄사', 
    
    # J 조사
    'JKS': '격 조사', # JK 격 조사
    'JKC': '격 조사',
    'JKG': '격 조사',
    'JKO': '격 조사',
    'JKM': '격 조사',
    'JKI': '격 조사',
    'JKQ': '격 조사',
    
    'JX': '보조사',

    'JC': '접속 조사',
    
    # E 어말 어미
    'EPH': '선어말어미', # EP 선어말 어미
    'EPT': '선어말어미', 
    'EPP': '선어말어미', 

    'EFN': '종결어미', # EF 종결 어미
    'EFQ': '종결어미',
    'EFO': '종결어미',
    'EFA': '종결어미',
    'EFI': '종결어미',
    'EFR': '종결어미',

    'ECE': '연결어미', # EC 연결 어미
    'ECD': '연결어미',
    'ECS': '연결어미',

    'ETN': '전성어미', # ET 전성 어미
    'ETD': '전성어미',
    
    'EMO': '이모티콘', # +추가
    
    # X 어근 접사
    'XPN': '접두사', # XP 접두사
    'XPV': '접두사',
    
    'XSN': '접미사', # XS 접미사
    'XSV': '접미사',
    'XSA': '접미사',
    'XSM': '접미사',
    'XSO': '접미사',
    
    'XR': '어근',
    
    # S 부호
    'SF': '마침표,물음표,느낌표',
    'SP': '쉼표,가운뎃점,콜론,빗금',
    'SS': '따옴표,괄호표,줄표',
    'SE': '줄임표',
    'SO': '붙임표(물결,숨김,빠짐)',
    'SW': '기타기호(논리수학,화폐)',

    # U 분석 불능
    'UN': '명사추정범주',
    'UV': '용언추정범주',
    'UE': '분석불능범주',

    # O 한글 이외
    'OL': '외국어',
    'OH': '한자',
    'ON': '숫자',
}

kkma_word_class_lv3 = {
    
    # N 체언
    'NNG': '명사', # NN 명사
    'NNP': '고유 명사',
    'NNB': '의존 명사',
    'NNM': '의존 명사(단위)',
    
    'NR': '수사',

    'NP': '대명사',

    # V 용언
    'VV': '동사',
    'VA': '형용사',

    'VX': '보조 용언', # +추가
    'VXV': '보조 동사', # VX 보조 용언
    'VXA': '보조 형용사',
    
    'VCP': '긍정 지정사', # VC 지정사
    'VCN': '부정 지정사',
    
    # M 수식언
    'MDT': '관형사(일반)', # MD 관형사
    'MDN': '관형사(수)',
    
    'MAG': '부사(일반)', # MA 부사
    'MAC': '부사(접속)',
    
    # I 감탄사
    'IC': '감탄사', 
    
    # J 조사
    'JKS': '조사(주격)', # JK 격 조사
    'JKC': '조사(보격)',
    'JKG': '조사(관형격)',
    'JKO': '조사(목적격)',
    'JKM': '조사(부사격)',
    'JKI': '조사(호격)',
    'JKQ': '조사(인용격)',
    
    'JX': '보조사',

    'JC': '접속 조사',
    
    # E 어말 어미
    'EPH': '선어말어미(존칭)', # EP 선어말 어미
    'EPT': '선어말어미(시제)', 
    'EPP': '선어말어미(공손)', 

    'EFN': '종결어미(평서형)', # EF 종결 어미
    'EFQ': '종결어미(의문형)',
    'EFO': '종결어미(명령형)',
    'EFA': '종결어미(청유형)',
    'EFI': '종결어미(감탄형)',
    'EFR': '종결어미(존칭형)',

    'ECE': '연결어미(대등)', # EC 연결 어미
    'ECD': '연결어미(의존)',
    'ECS': '연결어미(보조)',

    'ETN': '전성어미(명사형)', # ET 전성 어미
    'ETD': '전성어미(관형형)',

    'EMO': '이모티콘', # +추가
    
    # X 어근 접사
    'XPN': '접두사(체언)', # XP 접두사
    'XPV': '접두사(용언)',
    
    'XSN': '접미사(명사파생)', # XS 접미사
    'XSV': '접미사(동사파생)',
    'XSA': '접미사(형용사파생)',
    'XSM': '접미사(부사파생)',
    'XSO': '접미사(기타)',
    
    'XR': '어근',
    
    # S 부호
    'SF': '마침표,물음표,느낌표',
    'SP': '쉼표,가운뎃점,콜론,빗금',
    'SS': '따옴표,괄호표,줄표',
    'SE': '줄임표',
    'SO': '붙임표(물결,숨김,빠짐)',
    'SW': '기타기호(논리수학,화폐)',

    # U 분석 불능
    'UN': '명사추정범주',
    'UV': '용언추정범주',
    'UE': '분석불능범주',

    # O 한글 이외
    'OL': '외국어',
    'OH': '한자',
    'ON': '숫자',
}

kkma_word_class = [kkma_word_class_lv1, kkma_word_class_lv2, kkma_word_class_lv3]

word_class = {'komoran': komoran_word_class, 'kkma': kkma_word_class}

for class_dict in komoran_word_class + kkma_word_class:
    class_dict['[SOS]'] = '[시작]'
    class_dict['[EOS]'] = '[끝]'


def pos_ko(pos, translate_level=3, konlpy_tag='kkma'):
    if not 1 <= translate_level <= 3:
        print('Error: translate_level의 설정 범위는 1~3이다. (기본값 2로 설정)')
        translate_level = 3

    konlpy_tag = konlpy_tag.lower()
    if konlpy_tag not in ['komoran', 'kkma']:
        print("Error: 지원하는 konlpy tag: ['komoran', 'kkma'] (기본값 kkma로 설정)")
        konlpy_tag = 'kkma'

    pos2 = []
    for idx, word in enumerate(pos):
        # print(idx, word[0]) if word[1] in ['VXV'] else ()
        ko_wclass = word_class[konlpy_tag][translate_level-1][word[1]] if word[1] not in ['EOS','SOS'] else word[1]
        pos2.append((word[0], ko_wclass))

    return pos2


def tag2morp(wclass, translate_level=3, konlpy_tag='kkma'):
    if not 1 <= translate_level <= 3:
        print('Error: translate_level의 설정 범위는 1~3이다. (기본값 2로 설정)')
        translate_level = 3

    konlpy_tag = konlpy_tag.lower()
    if konlpy_tag not in ['komoran', 'kkma']:
        print("Error: 지원하는 konlpy tag: ['komoran', 'kkma'] (기본값 kkma로 설정)")
        konlpy_tag = 'kkma'
        
    return word_class[konlpy_tag][translate_level-1][wclass]