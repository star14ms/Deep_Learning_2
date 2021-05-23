package main

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"github.com/star14ms/util"
)

const txt_file_name = "youtube_comments.txt"

const baseURL = "https://www.youtube.com"
const URL = "https://www.youtube.com/feed/trending" // youtube URL to scrap

// html cards root
const video_block_name = "body" // a#thumbnail
const comment_block_name = "body" // ytd-comment-thread-renderers
const comment_content_name = "#content"

func main() {
	txt, err := os.Create(txt_file_name)
    util.CheckErr(err)
    defer txt.Close()

	scrap_videos_comments(txt)
}

func scrap_videos_comments(txt *os.File) {
	
	// URL 접속하여 html 가져오기
	res, err := http.Get(URL)
	util.CheckErr(err)
	util.Check_Http_Response(res)
	defer res.Body.Close()
	doc, err := goquery.NewDocumentFromReader(res.Body)
	util.CheckErr(err)

	// html에서 유튜브 동영상 블럭들 가져오기
	video_blocks := doc.Find(video_block_name)
	html := video_blocks.Text()
	videos_html := strings.Split(html, `"title":{"runs":[{"text":"`)

	c1 := make(chan string)
	// video_blocks.Each(func(i int, video_block *goquery.Selection) {
	// 	go scrap_video_comments(i, video_block, txt, c1)
	// })
	// for i := 1; i < video_blocks.Length(); i++ {
	// 	fmt.Println(<-c1) // 동영상 제목 출력
	// }
    for i := 1; i < len(videos_html); i++ {
		go scrap_video_comments(i, videos_html[i], txt, c1)
	}
	for i := 1; i < len(videos_html); i++ {
		// fmt.Println(<-c1) // 동영상 제목 출력
		<-c1
	}
}

// func scrap_video_comments(i int, video_block *goquery.Selection, txt *os.File, c1 chan<- string) {
func scrap_video_comments(i int, video_block string, txt *os.File, c1 chan<- string) {
    if i != 1 {
		c1 <- ""
		return 
	}
	// 비디오 블럭에서 링크와 동영상 제목 가져오기
	// video_URL, video_title := get_video_URL_title(video_block)
	video_title := strings.Split(video_block, `"`)[0]
	video_block2 := strings.Split(video_block, `,"watchEndpoint":{"videoId":"`)[1]
	id := strings.Split(video_block2, `"`)[0]
	video_URL := baseURL + "/watch?v=" + id
	fmt.Println(video_title, video_URL)

	// URL 접속하여 html 가져오기
	res, err := http.Get(video_URL)
	util.CheckErr(err)
	util.Check_Http_Response(res)
	defer res.Body.Close()
	doc, err := goquery.NewDocumentFromReader(res.Body)
	util.CheckErr(err)

    // html에서 유튜브 동영상 블럭들 가져오기
	video_html := doc.Text()
	fmt.Fprintf(txt, video_html)
    
	c2 := make(chan bool)
	// video_html.Each(func(i int, comment_black *goquery.Selection) {
	// 	go scrap_video_comment(comment_black, txt, c2)
	// })
	// for i := 0; i < video_html.Length(); i++ { /// length() X
	// 	<-c2
	// }
	// for i := 1; i < len(video_html); i++ {
	// 	go scrap_video_comment(video_html[i], txt, c2)
	// }
	for i := 1; i < len(video_html); i++ {
		// fmt.Println(<-c1) // 동영상 제목 출력
		<-c2
	}
	c1 <- video_title
}

func scrap_video_comment(comment_black *goquery.Selection, txt *os.File, c2 chan<- bool) {
    comment := comment_black.Find(comment_content_name).Text()
	fmt.Println(comment)
	fmt.Fprintf(txt, string(comment))
	c2 <- true
}

func get_video_URL_title(video_block *goquery.Selection) (string, string) {
    href, exists := video_block.Attr("href")
	util.CheckExists(exists)
	video_title, exists := video_block.Attr("title")
	util.CheckExists(exists)
	video_URL := baseURL + href
	return video_URL, video_title
}