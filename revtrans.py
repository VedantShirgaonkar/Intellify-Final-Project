# Convert text to gloss for video extraction and concatenation
import os
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY (e.g., in your shell or a .env file)."
client = OpenAI()  


# API Call
def text_to_gloss(sentence: str):
    prompt = f"""
    You are a sign language gloss generator.
    Convert the following English sentence into simplified ASL gloss (UPPERCASE keywords only, drop articles like 'the', 'is'):
    
    Sentence: "{sentence}"
    Output gloss:
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.2
    )
    
    gloss = response.choices[0].message.content.strip()
    return gloss


import os
import subprocess

# folder where your videos are stored
VIDEO_DIR = "videos"

# mapping from gloss word to actual file name
GLOSS_TO_VIDEO = {
    "college": "college.mp4",
    "good": "good.mp4",
    "teacher": "teacher.mp4",
    "work": "work.mp4",
    "day": "day.mp4",
    "i": "i.mp4",
    "time": "time.mp4",
    "you": "you.mp4",
    "exam": "exam.mp4",
    "student": "student.mp4",
    "we": "we.mp4",
}



def gloss_to_video_list(gloss_text):
    words = gloss_text.lower().split()
    video_files = []

    for w in words:
        if w in GLOSS_TO_VIDEO:
            video_files.append(os.path.join(VIDEO_DIR, GLOSS_TO_VIDEO[w]))
    return video_files


def create_concat_file(video_files, list_file="videos_to_concat.txt"):
    with open(list_file, "w") as f:
        for vf in video_files:
            f.write(f"file '{vf}'\n")
    return list_file


def concat_videos(video_files, output_file="output.mp4"):
    if not video_files:
        print("⚠️ No videos found for given gloss.")
        return
    
    list_file = create_concat_file(video_files)
    
    # run ffmpeg concat
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_file
    ])
    
    print(f"✅ Concatenated video saved as {output_file}")



def concat_videos_speed(video_files, output_file, speed=1.0):
    list_file = create_concat_file(video_files)

    # ffmpeg filter for speed
    speed_filter = f"setpts={1/speed}*PTS"

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-filter:v", speed_filter,
        "-an",  # remove audio, or handle separately
        output_file
    ])
    print(f"✅ Concatenated video saved as {output_file} (speed ×{speed})")



# Function to convert sentence to gloss tokens
def sentence_to_gloss_tokens(sentence, available_tokens=None):
    """Convert sentence to gloss and return tokens list"""
    gloss = text_to_gloss(sentence)
    tokens = gloss.lower().split()
    
    # Filter by available tokens if provided
    if available_tokens:
        tokens = [token for token in tokens if token in available_tokens]
    
    return tokens


# Standalone execution (only when run directly)
if __name__ == "__main__":
    print('Enter sentence to convert to gloss:')
    sentence = input().strip()
    gloss = text_to_gloss(sentence)
    print(f"Generated Gloss: {gloss}")
    video_files = gloss_to_video_list(gloss)
    concat_videos(video_files, "output.mp4")