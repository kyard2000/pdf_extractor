#This class was mostly coded by AI.
# Example usage:
#from VideoProcessorClass import VideoProcessor
#import os
#processor = VideoProcessor(azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
#        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#        azure_speech_key='6acf860ab894490b95b247537aeee4c2',
#        azure_service_region='eastus',
#        base_path=r"C:\Users\mrmikolyski\OneDrive - Church of Jesus Christ\Documents\WorkingDocs\Python\AI-POCs\AI-POCs\NonTextRAGSolutionsPOC\\",
#        apiversion = "2024-02-01",
#        gptengine = "gpt4o-databricks")
#processor.process_video("tithing.mp4")#this part takes a while.

import os
import re
from openai import AzureOpenAI
from base64 import b64encode
from moviepy.editor import VideoFileClip
import noisereduce as nr
import soundfile as sf
import numpy as np
import azure.cognitiveservices.speech as speechsdk
import time

class VideoProcessor:
    def __init__(self, azure_openai_key, azure_openai_endpoint, azure_speech_key, azure_service_region, base_path, apiversion, gptengine):
        self.openai_client = AzureOpenAI(
            api_key=azure_openai_key,
            api_version=apiversion,
            azure_endpoint=azure_openai_endpoint
        )
        self.azure_speech_key = azure_speech_key
        self.azure_service_region = azure_service_region
        self.base_path = base_path
        self.engine = gptengine

    def set_paths(self, video_filename):
        self.source_mp4 = os.path.join(self.base_path, video_filename)
        self.wav_path = os.path.join(self.base_path, "movieaudio.wav")
        self.clean_wav_path = os.path.join(self.base_path, "clean_movieaudio.wav")
        self.output_text_path = os.path.join(self.base_path, "audiotranscript.txt")
        self.frames_folder = os.path.join(self.base_path, "frames")
        self.screenplay_path = f"{self.source_mp4}.screenplay.text"

    def extract_frames(self, interval=5):
        print(f"Extracting frames from {self.source_mp4} to {self.frames_folder} every {interval} seconds.")
        os.makedirs(self.frames_folder, exist_ok=True)
        video_clip = VideoFileClip(self.source_mp4)
        duration = video_clip.duration
        for t in range(0, int(duration), interval):
            frame_path = os.path.join(self.frames_folder, f"frame_{t}.jpg")
            video_clip.save_frame(frame_path, t)
        print("Frame extraction complete.")
        video_clip.close()

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path, prompt):
        base64_image = self.encode_image(image_path)
        try:
            response = self.openai_client.chat.completions.create(
                model=self.engine,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while processing {image_path}: {str(e)}")
            return None

    @staticmethod
    def natural_sort_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

    def process_image_folder(self, prompt):
        results = {}
        image_files = [f for f in os.listdir(self.frames_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
        image_files.sort(key=self.natural_sort_key)
        for filename in image_files:
            image_path = os.path.join(self.frames_folder, filename)
            result = self.analyze_image(image_path, prompt)
            if result:
                results[filename] = result
        return results

    def extract_audio(self):
        print(f"Extracting audio from {self.source_mp4} to {self.wav_path}")
        video_clip = VideoFileClip(self.source_mp4)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(self.wav_path, codec='pcm_s16le')
        print("Extraction complete")
        video_clip.close()

    @staticmethod
    def reduce_noise(y, sr):
        print("Reducing noise from the audio")
        y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        return y_clean

    def process_audio(self):
        print(f"Processing audio for noise reduction: {self.wav_path}")
        y, sr = sf.read(self.wav_path)
        if len(y.shape) == 2 and y.shape[1] == 2:
            print("Converting to mono")
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        y_clean = self.reduce_noise(y, sr)
        sf.write(self.clean_wav_path, y_clean, sr)
        print("Audio processing complete")

    def transcribe_audio_azure(self):
        print(f"Transcribing {self.clean_wav_path} using Azure Speech-to-Text")
        speech_config = speechsdk.SpeechConfig(subscription=self.azure_speech_key, region=self.azure_service_region)
        audio_input = speechsdk.AudioConfig(filename=self.clean_wav_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

        all_results = []
        done = False
        audio_length = sf.info(self.clean_wav_path).duration

        def handle_final_result(evt):
            all_results.append(evt.result.text)

        def stop_recognition(evt):
            nonlocal done
            speech_recognizer.stop_continuous_recognition()
            done = True

        speech_recognizer.recognized.connect(handle_final_result)
        speech_recognizer.session_stopped.connect(stop_recognition)
        speech_recognizer.canceled.connect(stop_recognition)

        speech_recognizer.start_continuous_recognition()
        print("Processing audio for transcription...")

        start_time = time.time()
        while not done:
            time_elapsed = time.time() - start_time
            percent_complete = min((time_elapsed / audio_length) * 100, 100)
            print(f"Transcription progress: {percent_complete:.2f}% complete", end='\r')
            time.sleep(0.5)

        print("\nTranscription complete.")
        return " ".join(all_results)

    def generate_video_summary(self, image_descriptions, transcription):
        prompt = "The following are descriptions of individual frames from a video, followed by the video's transcription. Please provide a screenplay for the video based on both the visual descriptions and the audio transcription:\n\n"
        for filename, description in image_descriptions.items():
            prompt += f"Frame {filename}: {description}\n\n"
        prompt += f"Audio Transcription: {transcription}\n\n"
        prompt += "Video Summary:"

        try:
            response = self.openai_client.chat.completions.create(
                model=self.engine,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while generating the video summary: {str(e)}")
            return None

    def process_video(self, video_filename):
        self.set_paths(video_filename)
        self.extract_frames()
        
        image_prompt = "Describe this frame from a video in detail."
        all_results = self.process_image_folder(image_prompt)
        
        self.extract_audio()
        if os.path.exists(self.wav_path):
            self.process_audio()
        
        azure_text = ""
        if os.path.exists(self.clean_wav_path):
            azure_text = self.transcribe_audio_azure()
            print(f"Audio Transcription: {azure_text}")
            
            with open(self.output_text_path, 'w', encoding='utf-8') as f:
                f.write("Audio Transcription:\n")
                f.write(azure_text)
        
        video_summary = self.generate_video_summary(all_results, azure_text)
        if video_summary:
            with open(self.screenplay_path, 'w', encoding='utf-8') as f:
                f.write(video_summary)
        else:
            print("\nFailed to generate screenplay.")
        return f"{self.screenplay_path}-Created."

    def process_image(self, filepath):
        self.set_paths(filepath)
        self.extract_frames()
        
        image_prompt = "Describe this image in detail."
        all_results = self.process_image_folder(image_prompt)
        return all_results
        