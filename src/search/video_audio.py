import cv2
import numpy as np
import os
import sys
import time
import threading
import queue
import sounddevice as sd
from insightface.app import FaceAnalysis
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

##############################
# Global Variables and Setup
##############################

# Video globals
known_faces = []  # Each dict: {"name": "person_X", "embedding": np.array, "folder": folder_path}
current_active_person = None
video_lock = threading.Lock()  # For safely updating current_active_person
base_dir = "persons"
os.makedirs(base_dir, exist_ok=True)
MAX_FRAMES_PER_PERSON = 10

# Audio globals
SAMPLERATE = 16000
CHUNK_DURATION = 5  # seconds per audio chunk
CHUNK_SAMPLES = int(SAMPLERATE * CHUNK_DURATION)
# Instead of a simple audio_queue, we now queue tuples: (person, chunk)
audio_queue = queue.Queue()
audio_buffer = np.zeros((0,), dtype=np.float32)
# Dictionary to hold transcripts per person
transcripts = {}   # e.g., {"person_1": ["line1", "line2"], "person_2": [...] }
transcript_lock = threading.Lock()

# Control flags (shared by audio and video)
stop_event = threading.Event()       # Signals when to stop everything
recording_active = False             # True if a face has been detected and session is active

##############################
# Helper Functions: Video
##############################

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def add_new_person(embedding):
    """Creates a new person entry and folder."""
    person_id = len(known_faces) + 1
    person_name = f"person_{person_id}"
    folder = os.path.join(base_dir, person_name)
    os.makedirs(folder, exist_ok=True)
    known_faces.append({"name": person_name, "embedding": embedding, "folder": folder})
    # Also initialize transcript for that person
    with transcript_lock:
        transcripts[person_name] = []
    print(f"[Video] New person detected: {person_name}")
    return person_name

def identify_person(embedding, threshold=0.5):
    """Compare embedding with known faces using cosine similarity."""
    for person in known_faces:
        sim = cosine_similarity(embedding, person["embedding"])
        if sim > threshold:
            return person["name"]
    return None

##############################
# Helper Functions: Audio
##############################

# Load the Speech2Text processor and model
MODEL_NAME = "facebook/s2t-large-librispeech-asr"
processor = Speech2TextProcessor.from_pretrained(MODEL_NAME)
model = Speech2TextForConditionalGeneration.from_pretrained(MODEL_NAME)
SAVE_INTERVAL = 3  # seconds, for saving transcript

def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block. Queues audio only if a person is active.
       We now queue a tuple of (active_person, audio_chunk)."""
    global audio_buffer
    # Use video_lock to read the current active person safely
    with video_lock:
        active_person = current_active_person

    if not recording_active or stop_event.is_set() or active_person is None:
        return  # Only process audio if recording is active and a person is detected

    # Use only one channel (mono)
    audio_chunk = indata[:, 0]
    audio_buffer = np.concatenate([audio_buffer, audio_chunk])
    if len(audio_buffer) >= CHUNK_SAMPLES:
        chunk = audio_buffer[:CHUNK_SAMPLES]
        audio_buffer = audio_buffer[CHUNK_SAMPLES:]
        # Queue the tuple (active_person, chunk)
        audio_queue.put((active_person, chunk))

def transcribe_loop():
    """Continuously transcribes audio chunks from the queue and routes transcription to the correct person."""
    global recording_active
    while not stop_event.is_set():
        try:
            person, chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        print(f"\n[Audio] Transcribing a new chunk for {person}...")
        inputs = processor(chunk, sampling_rate=SAMPLERATE, return_tensors="pt")
        generated_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"[Audio] Transcription for {person}:", transcription)
        with transcript_lock:
            # Append the transcript line to the corresponding person
            if person in transcripts:
                transcripts[person].append(transcription)
            else:
                transcripts[person] = [transcription]
        # If "stop recording" is found, trigger stop
        if "stop recording" in transcription.lower():
            print("[Audio] Detected 'stop recording'. Triggering stop event.")
            stop_event.set()
            recording_active = False

def save_transcript_loop():
    """Periodically saves the transcript for each person to separate files."""
    while not stop_event.is_set():
        time.sleep(SAVE_INTERVAL)
        with transcript_lock:
            for person, lines in transcripts.items():
                filename = f"transcript_{person}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print(f"[Audio] Transcript saved to {filename}.")

##############################
# Main: Video Capture Loop
##############################

def video_loop():
    global current_active_person, recording_active
    # Initialize InsightFace
    face_app = FaceAnalysis()
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Video] Error: Could not open video stream.")
        stop_event.set()
        return

    print("[Video] Press 'q' in the video window to quit or say 'stop recording' via audio.")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[Video] Failed to read frame from camera.")
                break

            faces = face_app.get(frame)
            if len(faces) == 1:
                embedding = faces[0].embedding
                person_name = identify_person(embedding)
                if person_name is None:
                    person_name = add_new_person(embedding)
                # Mark the person as active and signal audio recording should start
                with video_lock:
                    current_active_person = person_name
                    if not recording_active:
                        print(f"[Video] Starting recording for {person_name}.")
                        recording_active = True

                # Save up to MAX_FRAMES_PER_PERSON images for this person
                person_folder = os.path.join(base_dir, person_name)
                person_frame_files = [f for f in os.listdir(person_folder) if f.startswith("face_") and f.endswith(".jpg")]
                if len(person_frame_files) < MAX_FRAMES_PER_PERSON:
                    timestamp = int(time.time())
                    bbox = faces[0].bbox.astype(int)
                    x1 = max(bbox[0], 0)
                    y1 = max(bbox[1], 0)
                    x2 = min(bbox[2], frame.shape[1])
                    y2 = min(bbox[3], frame.shape[0])
                    if x2 > x1 and y2 > y1:
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size != 0:
                            img_filename = os.path.join(person_folder, f"face_{timestamp}.jpg")
                            cv2.imwrite(img_filename, face_img)
                            print(f"[Video] Saved frame {len(person_frame_files)+1}/{MAX_FRAMES_PER_PERSON} for {person_name}")
                        else:
                            print("[Video] Warning: Empty face image; skipping save.")
                    else:
                        print("[Video] Warning: Invalid bounding box; skipping save.")
            else:
                # If no face or multiple faces detected, reset active flag
                with video_lock:
                    current_active_person = None

            # Draw bounding boxes and labels on the frame for visual feedback
            for face in faces:
                bbox = face.bbox.astype(int)
                detected_name = identify_person(face.embedding)
                label = detected_name if detected_name else "Unknown"
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Face & Audio Capture", frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Video] 'q' pressed. Triggering stop event.")
                stop_event.set()
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()  # Explicitly destroy all windows
        print("[Video] Video session ended and camera released.")

##############################
# Main Function to Start Threads
##############################

def main():
    # Start audio transcription and transcript saving threads
    transcribe_thread = threading.Thread(target=transcribe_loop, daemon=True)
    save_thread = threading.Thread(target=save_transcript_loop, daemon=True)
    transcribe_thread.start()
    save_thread.start()

    # Start the audio input stream in its own thread
    def audio_stream():
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLERATE, dtype="float32"):
            while not stop_event.is_set():
                time.sleep(0.1)
    audio_thread = threading.Thread(target=audio_stream, daemon=True)
    audio_thread.start()

    # Run the video capture loop in the main thread
    video_loop()

    # Wait for audio-related threads to finish
    transcribe_thread.join(timeout=2)
    save_thread.join(timeout=2)
    audio_thread.join(timeout=2)
    
    # Final cleanup: ensure OpenCV windows are destroyed
    cv2.destroyAllWindows()
    print("[Main] Final transcripts saved. Exiting now.")
    sys.exit(0)

if __name__ == "__main__":
    main()