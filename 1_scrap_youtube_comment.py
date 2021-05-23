from selenium import webdriver
import sys, os
import time as t
from common.util import time
# from selenium.webdriver.common.keys import Keys


##### 함수 정의 ###############################################################################################################


# URL에 접속하면 있는 모든 동영상의 제목과 링크 가져오기
def get_video_titles_URLs(driver, video_block, URL):
    
    # 웹 사이트 가져오기
    driver.get(URL)
    driver.implicitly_wait(10)

    # 동영상 제목과 URL 모두 가져오기
    videos = driver.find_elements_by_id(video_block)
    titles, URLs = [], []
    for video in videos:
        href = video.get_attribute("href")
        title = video.get_attribute("title")
        URLs.append(href)
        titles.append(title)

    print('\n찾은 동영상 갯수: %d개'% len(URLs))
    return URLs, titles


# 링크에 하나씩 들어가서 유튜브 댓글 수집하기
def get_youtube_video_comment(driver, URLs, titles, comment_block, cmts_txt_name, urls_txt_name=None, scroll_down_num=1000):
    print("0 | 댓글 수 | 동영상 제목")
    
    # urls 파일이 존재하면 그 안의 url들은 제외하고 댓글 수집
    cmts_already_saved_urls = None
    if urls_txt_name!=None or os.path.isfile(f'{urls_txt_name}.txt'):
        with open(f'{urls_txt_name}.txt', 'r', encoding="utf8") as f:
            cmts_already_saved_urls = [url_info.split('\t')[0] for url_info in f.read().splitlines()]
    
    saved_cmts_num = 0
    for i, url in enumerate(URLs):
        
        # 댓글 수집한 동영상 목록에 있던 url이면 건너뛰기
        if cmts_already_saved_urls!=None and url in cmts_already_saved_urls:
            print("\r{0} | 완료 | {1}".format(i+1, (titles[i] if len(titles[i])<33 else titles[i][:33]+'..')))
            continue

        # 웹 사이트 가져오기
        driver.get(url)
        driver.implicitly_wait(10)
        t.sleep(1)
        
        # 동영상 일시정지, 스크롤 내리기 반복
        video = driver.find_element_by_tag_name('video') # 'ytp-play-button'
        webdriver.ActionChains(driver).click(video).perform() # webdriver.ActionChains(driver).send_keys(Keys.SPACE)
        scroll_downs_to_load_cmts(driver, scroll_down_num)
     
        # 댓글들 가져오기
        comments = driver.find_elements_by_id(comment_block)
    
        # txt파일로 기록
        skip_num = 0
        with open(f'{cmts_txt_name}.txt', 'a', encoding="utf-8") as f: 
            # file.write("#video_title "+ titles[i]+'\n')
            for cmt in comments:
                cmt = cmt.text.replace('\n', ' ')
                if len(cmt)==0: 
                    skip_num += 1
                else:
                    f.write(cmt + ('.\n' if cmt[-1] not in '.?!' else '\n'))
        
        # 댓글 수집한 동영상 목록에 현재 url추가
        with open(f'{urls_txt_name}.txt', 'a', encoding="utf-8") as f:
            f.write(url+'\t'+titles[i]+'\n')
    
        saved_cmts_num += len(comments)-skip_num
        print(("\r{0} | {1}개 | {2}".format(i+1, len(comments)-skip_num, (titles[i] if len(titles[i])<33 else titles[i][:33]+'..'))).ljust(30))
    
    print('\n수집한 댓글 수: %d개'% saved_cmts_num)


# 스크롤 내리기 반복
def scroll_downs_to_load_cmts(driver, scroll_down_num=1000):
    for j in range(scroll_down_num):
        sys.stdout.write('\r스크롤 내리기 %d / %d (%s)' % (j, scroll_down_num, time.str_hms_delta(start_time)))
        sys.stdout.flush()
        driver.execute_script(f"window.scrollTo(0, {(j+1)*2000});")
        driver.implicitly_wait(10)
        t.sleep(0.1)


##### 변수 선언 #########################################################################################################


start_time = t.time() 
scroll_down_num = 1000

# 브라우저 원격 접속 인터페이스
driver_path = r'C:\Users\danal\Documents\programing\chromedriver.exe'
driver = webdriver.Chrome(driver_path) 

# 검색하고자 하는 유튜브 페이지
URL = 'https://www.youtube.com/feed/trending?bp=6gQJRkVleHBsb3Jl'

# 찾고자 하는 html의 특정 위치들
video_block = 'video-title'
comment_block = 'content-text'

# 저장할 txt파일 이름 (댓글, 동영상 링크 저장)
cmts_txt_name = 'dataset/YT_cmts'
urls_txt_name = 'dataset/YT_cmts_urls'


##### main #####################################################################################################################


if __name__ == '__main__':
    URLs, titles = get_video_titles_URLs(driver, video_block, URL)
    get_youtube_video_comment(driver, URLs, titles, comment_block, cmts_txt_name, urls_txt_name, scroll_down_num)
    
    print(time.str_delta(start_time))
