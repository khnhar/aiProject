from pytube import YouTube
import cv2
from PIL import Image
import clip
import torch
import math
import numpy as np
import plotly.express as px
import datetime
from IPython.core.display import HTML


class VideoSearch:
    def __init__(self, video_url):
        self.video_url = video_url
        self.N = 120
        self.download_video()
        self.extract_frames()
        self.load_clip_model()
        self.encode_frames()

    def download_video(self):
        streams = YouTube(self.video_url).streams.filter(adaptive=True, subtype="mp4", resolution="360p",
                                                         only_video=True)
        if len(streams) == 0:
            raise ValueError("No suitable stream found for this YouTube video!")
        print("Downloading...")
        streams[0].download(filename="video.mp4")
        print("Download completed.")

    def extract_frames(self):
        self.video_frames = []
        capture = cv2.VideoCapture('video.mp4')
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        current_frame = 0
        while capture.isOpened():
            ret, frame = capture.read()
            if ret == True:
                self.video_frames.append(Image.fromarray(frame[:, :, ::-1]))
            else:
                break
            current_frame += self.N
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        print(f"Frames extracted: {len(self.video_frames)}")

    def load_clip_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def encode_frames(self):
        batch_size = 256
        self.batches = math.ceil(len(self.video_frames) / batch_size)
        self.video_features = torch.empty([0, 512], dtype=torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i in range(self.batches):
            batch_frames = self.video_frames[i * batch_size: (i + 1) * batch_size]
            batch_preprocessed = torch.stack([self.preprocess(frame) for frame in batch_frames]).to(device)
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_preprocessed)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
            self.video_features = torch.cat((self.video_features, batch_features))
        print(f"Features: {self.video_features.shape}")

    def search_video(self, search_query, display_heatmap=True, display_results_count=3):
        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize(search_query))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * self.video_features @ text_features.T)
        values, best_photo_idx = similarities.topk(display_results_count, dim=0)
        if display_heatmap:
            print("Search query heatmap over the frames of the video:")
            fig = px.imshow(similarities.T.cpu().numpy(), height=50, aspect='auto', color_continuous_scale='viridis',
                            binary_string=True)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            fig.show()
            print()

        result_images = []
        for frame_id in best_photo_idx:
            frame_id = frame_id.item()  # Convert to integer
            result_images.append(self.video_frames[frame_id])
            seconds = round(frame_id * self.N / self.fps)
            print(f"Found at {str(datetime.timedelta(seconds=seconds))} (Link: {self.video_url}&t={seconds})")

        return result_images

    @staticmethod
    def display_image(image):
        image.show()
