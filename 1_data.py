from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from common.print import add_spaces_until_endline as line
from common.util import time

import sys, os
import time as t, datetime as dt

##### 함수 선언 ###############################################################################################################


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
def get_youtube_video_comment(driver, URLs, titles, comment_block, save_txt_name, urls_txt_name=None, n_scroll_down=500):
    print("0 | 댓글 수 | 동영상 제목")
    
    # urls 파일이 존재하면 그 안의 url들은 제외하고 댓글 수집
    cmts_already_saved_urls = None
    if urls_txt_name!=None and os.path.isfile(f'{urls_txt_name}.txt'):
        with open(f'{urls_txt_name}.txt', 'r', encoding="utf-8") as f:
            cmts_already_saved_urls = [url_info.split('\t')[0] for url_info in f.read().splitlines()]
    
    saved_cmts_num = 0
    for i, url in enumerate(URLs):
        
        # 댓글 수집한 동영상 목록에 있던 url이면 건너뛰기
        if cmts_already_saved_urls!=None and url in cmts_already_saved_urls:
            print(line("\r{0} | 완료 | {1}".format(i+1, titles[i])))
            continue

        # 웹 사이트 가져오기
        driver.get(url)
        driver.implicitly_wait(10)
        t.sleep(0.2)
        
        # 동영상 일시정지, 스크롤 내리기 반복
        video = driver.find_element_by_tag_name('video') # 'ytp-play-button'
        driver.execute_script("arguments[0].pause()", video)
        # webdriver.ActionChains(driver).click(video).perform() # webdriver.ActionChains(driver).send_keys(Keys.SPACE)
        t.sleep(0.5)
        scroll_down(driver, n_scroll_down)
        
        # 댓글들 가져오기
        comments = driver.find_elements_by_id(comment_block)
    
        if len(comments) < 100:
            print(line("\r{0} | 스킵 | {1}".format(i+1, titles[i])))
            continue

        # txt파일로 기록
        skip_num = 0
        with open(f'{save_txt_name}.txt', 'a', encoding="utf-8") as f: 
            # if.write('$;{}\t{}\t{}\n\n'.format(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), url, titles[i]))
            for cmt in comments:
                cmt = cmt.text.replace('\n', ' ')
                if len(cmt)==0: 
                    skip_num += 1
                else:
                    f.write(cmt + '\n')
                    # f.write(cmt + ('.\n' if cmt[-1] not in '.?!' else '\n'))
            # f.write('\n')
        
        # 댓글 수집한 동영상 목록에 현재 url추가
        with open(f'{urls_txt_name}.txt', 'a', encoding="utf-8") as f:
            f.write(url+'\t'+titles[i]+'\n')
    
        saved_cmts_num += len(comments)-skip_num
        print(line("\r{0} | {1}개 | {2}".format(i+1, titles[i])))
    
    print('\n수집한 댓글 수: %d개'% saved_cmts_num)


# 스크롤 내리기 반복
def scroll_down(driver, n_scroll_down=500, waiting_time=0.1, break_num=80):
    body = driver.find_element_by_tag_name('body')
    height_last = driver.execute_script('return document.querySelector("#columns").offsetHeight')
    n_same_height = 0

    for n in range(1, n_scroll_down+1):
        sys.stdout.write('\r스크롤 내리기 %d / %d (창 높이: %dpx) (%s)' % (n, n_scroll_down, height_last, time.str_delta(start_time)))
        sys.stdout.flush()

        body.send_keys(Keys.END) # driver.execute_script(f"window.scrollTo(0, {(j+1)*2000});")
        driver.implicitly_wait(10)
        t.sleep(waiting_time)
        
        # 창 높이가 계속 변하지 않으면 중단
        height_now = driver.execute_script('return document.querySelector("#columns").offsetHeight')
        if height_last!=height_now:
            n_same_height = 0
            height_last = height_now
        else:
            n_same_height += 1
            if n_same_height > break_num: break
    
    sys.stdout.write('\r댓글 수집 중... %d / %d (창 높이: %dpx) (%s)' % (n, n, height_last, time.str_delta(start_time)))
    sys.stdout.flush()


##### 변수 선언 #########################################################################################################

start_time = t.time()
n_scroll_down = 500

# 브라우저 원격 접속 인터페이스
# driver_path = r'C:\Users\user\Documents\Coding\chromedriver.exe'
driver_path = r'C:\Users\danal\Documents\programing\chromedriver.exe'
driver = webdriver.Chrome(driver_path)
driver.set_window_position(0, 0)
driver.set_window_size(800, 800)

# 검색하고자 하는 유튜브 페이지
URL = 'https://www.youtube.com/feed/trending?bp=6gQJRkVleHBsb3Jl'

# 찾고자 하는 html의 특정 위치들
video_block = 'video-title'
comment_block = 'content-text'

# 저장할 txt파일 이름 (댓글, 동영상 링크 저장)
# day = dt.date.today().strftime('%y%m%d')
day = "211101~06"
save_txt_name = f'data/YT_cmts_{day}'
urls_txt_name = f'data/YT_cmts_urls_{day}'


##### main #####################################################################################################################


if __name__ == '__main__':
    URLs, titles = get_video_titles_URLs(driver, video_block, URL)
    get_youtube_video_comment(driver, URLs, titles, comment_block, save_txt_name, urls_txt_name, n_scroll_down)
    
    print(time.str_delta(start_time))
