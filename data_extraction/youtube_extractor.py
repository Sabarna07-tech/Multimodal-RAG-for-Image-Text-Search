# multimodal_rag/data_extraction/youtube_extractor.py

import os
import cv2
import yt_dlp # <-- Import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

def extract_youtube_data(url, output_dir="output/youtube_data"):
    """
    Downloads a YouTube video, extracts its transcript with timestamps,
    and saves video frames using the robust yt-dlp library.

    Args:
        url (str): The URL of the YouTube video.
        output_dir (str): The directory to save the video, transcript, and frames.

    Returns:
        tuple: A tuple containing the path to the video, the transcript,
               and a list of paths to the extracted frames.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- Use yt-dlp for downloading and metadata ---
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_id = info_dict.get('id', None)
            video_path = ydl.prepare_filename(info_dict)

        if not video_id:
            raise ValueError("Could not extract video ID from the URL.")
            
        print(f"Video downloaded to: {video_path}")

        # --- Get transcript (youtube_transcript_api is still reliable) ---
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # --- Extract frames (same logic as before) ---
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_seconds = 5
        frame_paths = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_sec = frame_count / fps
            
            # Save a frame every 'frame_interval_seconds'
            if frame_count % (int(fps) * frame_interval_seconds) == 0:
                frame_filename = os.path.join(output_dir, f"frame_{int(current_time_sec)}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename)
                print(f"Saved frame at {int(current_time_sec)}s")
            
            frame_count += 1
            
        cap.release()
        return video_path, transcript, frame_paths

    except Exception as e:
        # Pass a more descriptive error up
        raise type(e)(f"An error occurred in YouTube extraction: {e}")