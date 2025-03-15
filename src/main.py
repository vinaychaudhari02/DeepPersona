import subprocess
import os
import glob

# Function to run a Python script
def run_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

# Function to get the most recent JSON file from a directory
def get_latest_json(directory):
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if not json_files:
        return None
    # Get the file with the latest modification time
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file

if __name__ == '__main__':
    # Paths of the scripts
    script_dir = os.path.dirname(__file__)
    browser_script = os.path.join(script_dir, 'browser_collector.py')
    summarizer_script = os.path.join(script_dir, 'model_summarizer.py')

    # Run the browser script
    run_script(browser_script)

    # Ask the user if they want to run the summarizer
    user_input = input('Do you want to run the summarizer to extract details? (yes/no): ').strip().lower()

    if user_input == 'yes':
        # Define the directory where JSON files are stored relative to the project root
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        json_directory = os.path.join(project_root, 'data', 'json')

        # Get the latest JSON file
        latest_json = get_latest_json(json_directory)
        if latest_json:
            os.environ['JSON_FILE_PATH'] = latest_json
            print(f"Using JSON file: {latest_json}")
            run_script(summarizer_script)
        else:
            print("No JSON file found in the directory.")
    else:
        print("Summarizer execution skipped.")