import gradio as gr
from natural_language_youtube_search import VideoSearch

# Gradio에 입력과 출력을 정의하여 웹 애플리케이션을 생성
input_video_url = gr.inputs.Textbox(label="Enter the YouTube video URL")
input_text = gr.inputs.Textbox(label="Enter your search query")

def search_video(video_url, query):
    # 비디오 URL을 입력받아 VideoSearch 객체 생성
    video_search = VideoSearch(video_url)
    # 입력된 검색어로 비디오 검색 수행
    result_images = video_search.search_video(query)
    return result_images

def handle_submit(video_url, query):
    result_images = search_video(video_url, query)
    # 이미지들을 HTML로 묶어서 스크롤 가능하도록 만듭니다.
    images_html = "<br>".join([f"<img src='{img}' style='max-width: 100%; max-height: 300px; margin: 10px;'/>" for img in result_images])
    return images_html

# Gradio 인터페이스를 정의하고 실행
interface = gr.Interface(fn=handle_submit, inputs=[input_video_url, input_text], outputs="html", live=False, capture_session=True, title="YouTube Video Search")
interface.launch()
