import json
import os

# --- Configuration ---
NUM_CLASSES_TO_DOWNLOAD = 10
JSON_PATH = 'WLASL-master/start_kit/WLASL_v0.3.json'
VIDEO_FOLDER = 'videos'

# --- Setup ---
# We still create the main 'videos' folder
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# --- Load the JSON file ---
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# --- Main Download Loop ---
successful_downloads = 0
failed_downloads = 0

print(f"--- Starting download for the first {NUM_CLASSES_TO_DOWNLOAD} classes ---")

for i in range(NUM_CLASSES_TO_DOWNLOAD):
    entry = data[i]
    gloss = entry['gloss']
    print(f"\nProcessing Class: {gloss}")

    # --- NEW: Create a subfolder for the current word ---
    class_folder_path = os.path.join(VIDEO_FOLDER, gloss)
    os.makedirs(class_folder_path, exist_ok=True)

    for instance in entry['instances']:
        video_id = instance['video_id']
        video_url = instance['url']
        
        # --- UPDATED: The output path now points to the new subfolder ---
        output_path = os.path.join(class_folder_path, f"{video_id}.mp4")

        # Check if the file already exists to avoid re-downloading
        if os.path.exists(output_path):
            print(f"  - Video {video_id} already exists. Skipping.")
            continue

        print(f"  - Attempting to download video {video_id} from: {video_url}")
        
        command = (
            f'yt-dlp --quiet --no-warnings '
            f'-o "{output_path}" '
            f'--merge-output-format mp4 "{video_url}"'
        )
        
        status_code = os.system(command)
        if status_code == 0:
            print(f"    SUCCESS: Video {video_id} downloaded.")
            successful_downloads += 1
        else:
            print(f"    FAILURE: Could not download video {video_id}.")
            failed_downloads += 1

print("\n--- Download complete! ---")
print(f"Successfully downloaded: {successful_downloads} videos")
print(f"Failed to download: {failed_downloads} videos")