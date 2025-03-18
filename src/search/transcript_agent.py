import re
import json
import time
import os
import requests
from textblob import TextBlob

# --- DeepSeek Configuration ---
DEEPSEEK_API_KEY = "sk-a342026d7e6f4208865a292559b3f8db"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Accept": "application/json"
}

# --- File Monitoring Config ---
TRANSCRIPT_FILE = "/Users/vinaychaudhari/Documents/Face_Detection/SearchEngine_Face_Detection/name_city_data/transcript.txt"
OUTPUT_JSON = "people_data.json"
POLL_INTERVAL = 9  # Seconds between checks

# --- State Management ---
class FileWatcher:
    def __init__(self, filename):
        self.filename = filename
        self.last_size = 0
        self.last_modified = 0

    def check_changes(self):
        try:
            stat = os.stat(self.filename)
            if stat.st_mtime > self.last_modified or stat.st_size != self.last_size:
                self.last_modified = stat.st_mtime
                self.last_size = stat.st_size
                return True
        except FileNotFoundError:
            print(f"Waiting for file: {self.filename}")
        return False

# --- Processing Core ---
class LiveProcessor:
    def __init__(self):
        self.watcher = FileWatcher(TRANSCRIPT_FILE)
        self.file_position = 0
        self.ensure_json_file()

    def ensure_json_file(self):
        if not os.path.exists(OUTPUT_JSON):
            with open(OUTPUT_JSON, 'w') as f:
                json.dump([], f)

    def process_new_content(self):
        try:
            with open(TRANSCRIPT_FILE, 'r', encoding='utf-8') as f:
                f.seek(self.file_position)
                new_content = f.read()
                self.file_position = f.tell()

            if new_content.strip():
                segments = self.preprocess_text(new_content)
                new_records = []
                for seg in segments:
                    records = self.process_segment(seg)
                    if records:
                        new_records.extend(records)

                if new_records:
                    self.update_json_output(new_records)
                    return True
        except Exception as e:
            print(f"Processing error: {str(e)}")
        return False

    def preprocess_text(self, text):
        # Normalize whitespace and trim text
        normalized = re.sub(r'\s+', ' ', text).strip()
        # Split text into segments using punctuation as delimiter
        segments = re.split(r'(?<=[.!?])\s+', normalized)
        return [seg.strip() for seg in segments if seg.strip()]

    def process_segment(self, segment):
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=API_HEADERS,
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": self.create_prompt(segment),
                    "temperature": 1.0,
                },
                timeout=120
            )
            response.raise_for_status()

            content = response.json()['choices'][0]['message']['content']
            # Remove backticks or code formatting and ensure valid JSON
            cleaned_content = content.strip('`').replace('json\n', '')
            api_data = json.loads(cleaned_content)

            # Expected API response structure:
            # {
            #   "persons": [{ "name": "John Doe", "additional_details": { ... } }, ...],
            #   "locations": [{ "name": "New York", "additional_details": { ... } }, ...],
            #   "additional_details": { ... }  // optional extra info for the segment
            # }
            persons = api_data.get('persons', [])
            locations = api_data.get('locations', [])
            extra_details = api_data.get('additional_details', {})

            records = []
            # For each person entry, create a record and pair with location if available.
            for idx, person_data in enumerate(persons):
                # Person data can be a dict or a simple string.
                if isinstance(person_data, dict):
                    person_name = person_data.get("name")
                    person_extra = person_data.get("additional_details", {})
                else:
                    person_name = person_data
                    person_extra = {}

                # Check if we have a corresponding location
                if idx < len(locations):
                    loc_data = locations[idx]
                    if isinstance(loc_data, dict):
                        loc_name = loc_data.get("name")
                        loc_extra = loc_data.get("additional_details", {})
                    else:
                        loc_name = loc_data
                        loc_extra = {}
                    location_record = {"name": loc_name, "additional_details": loc_extra} if loc_name else None
                else:
                    location_record = None

                # Combine extra details: both segment-level and person-specific.
                combined_details = extra_details.copy()
                combined_details.update(person_extra)

                records.append({
                    "person": {"name": person_name, "additional_details": person_extra} if isinstance(person_data, dict) else person_name,
                    "location": location_record,
                    "details": {"text_summary": combined_details.get("text_summary", ""), **combined_details},
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "context": segment
                })
            return records

        except Exception as e:
            print(f"API Error for segment '{segment}': {str(e)}")
            return []

    def create_prompt(self, text):
        # The prompt instructs DeepSeek to extract persons, locations, and extra details in JSON format.
        return [
            {
                "role": "system",
                "content": (
                    "Extract persons and locations in JSON format with array fields 'persons' and 'locations'. "
                    "For each person, return their name and any additional details in an 'additional_details' field. "
                    "For each location, return the name and any extra information in an 'additional_details' field. "
                    "Also include any extra summary information in an 'additional_details' object if available."
                )
            },
            {
                "role": "user",
                "content": f"Text: {text}\n\nJSON:"
            }
        ]

    def update_json_output(self, new_records):
        try:
            try:
                with open(OUTPUT_JSON, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []

            # Append new records
            updated_data = existing_data + new_records

            # Write updated data back to file
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(updated_data, f, indent=2)
            print(f"Updated {OUTPUT_JSON} with {len(new_records)} new record(s).")
        except Exception as e:
            print(f"JSON Update Error: {str(e)}")

# --- Main Execution ---
def monitor_and_process():
    processor = LiveProcessor()
    print(f"Monitoring {TRANSCRIPT_FILE} for changes...")
    try:
        while True:
            if processor.watcher.check_changes():
                if processor.process_new_content():
                    print("Processed new content.")
                else:
                    print("No new content processed.")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_and_process()