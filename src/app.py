import os
import subprocess
from utils.file_utils import get_latest_json
from search.browser_collector import run_browser_collection
from summarization.model_summarizer import run_summarizer

if __name__ == '__main__':
    # Run the browser collector to fetch images and data
    run_browser_collection()

    # Ask the user if they want to summarize the collected data
    user_input = input('Do you want to run the summarizer? (yes/no): ').strip().lower()

    if user_input == 'yes':
        json_path = get_latest_json('data/json/')
        if json_path:
            os.environ['JSON_FILE_PATH'] = json_path
            print(f"Using JSON file: {json_path}")
            run_summarizer(json_path)
        else:
            print("No JSON file found in the directory.")
    else:
        print("Summarizer execution skipped.")
