import os
import json
import subprocess
import glob
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF

# ---------------------------
# Utility Functions
# ---------------------------

def extract_person_name(filename):
    """
    Extracts the person's name from the filename.
    The filename is expected to be in the format "firstname_lastname_search.json".
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return " ".join(parts[:2])
    else:
        return base

def clean_json(data, person_name):
    """
    Recursively cleans JSON data using keyword-based filtering.
    Keeps dictionary keys or string values that contain the person's name,
    as well as numeric or boolean values.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            key_has_name = person_name.lower() in key.lower()
            cleaned_value = clean_json(value, person_name)
            if isinstance(value, str):
                value_has_name = person_name.lower() in value.lower()
            else:
                value_has_name = False

            if key_has_name or value_has_name or isinstance(value, (int, float, bool)):
                if cleaned_value not in [None, {}, []]:
                    new_dict[key] = cleaned_value
            elif isinstance(cleaned_value, dict) and cleaned_value:
                new_dict[key] = cleaned_value
            elif isinstance(cleaned_value, list) and cleaned_value:
                new_dict[key] = cleaned_value
        return new_dict
    elif isinstance(data, list):
        new_list = []
        for item in data:
            cleaned_item = clean_json(item, person_name)
            if cleaned_item not in [None, {}, []]:
                new_list.append(cleaned_item)
        return new_list
    else:
        return data

# ---------------------------
# Advanced Extraction Prompt
# ---------------------------

EXPERT_PROMPT = (
    "You are an expert information extractor and summarizer. Your task is to analyze the provided JSON data, "
    "which contains web search results for a person, and extract every piece of available information related to that person. "
    "This task applies to individuals from any profession on the planet, whether they are in research, business, arts, sports, or any other field. "
    "You must extract both personal and professional details in full, without omitting or excessively summarizing any information.\n\n"
    "Instructions:\n"
    "1. Thoroughly analyze the JSON structure, including any 'pages' arrays from various search engines (such as Bing, Google, Yahoo), "
    "and identify all fields containing text, links, and any other data.\n"
    "2. Extract and organize the information into a structured format with clear headings and subheadings. Include, but do not limit yourself to, the following sections:\n"
    "   - **Overall Summary:** A brief overview of the main findings.\n"
    "   - **Personal Profile:** Name, basic personal information (if available), and contact details (e.g., email, phone number, office or mailing address).\n"
    "   - **Professional Profile:** Current position, roles, affiliations, and work history.\n"
    "   - **Educational Background:** Degrees, institutions attended, and graduation years.\n"
    "   - **Employment and Work History:** Details of previous work, collaborations, and relevant professional experiences.\n"
    "   - **Projects, Initiatives, and Interests:** Information about projects, initiatives, or areas of interest relevant to the person’s profession.\n"
    "   - **Achievements and Awards:** Any honors, awards, or recognitions received.\n"
    "   - **Publications and Media:** A list of all publications, articles, or media appearances, including details like titles, co-authors, publication venues, dates, DOIs, and links if available.\n"
    "   - **External Links and Profiles:** All available URLs and links to personal websites, social media accounts, and professional profiles (e.g., LinkedIn, ORCID, ResearchGate, Rate My Professors, Google Scholar, etc.).\n"
    "   - **Additional Details:** Any other relevant information, such as news articles, interviews, events, professional service, or miscellaneous details.\n\n"
    "3. Include every available link and URL, categorizing them under the appropriate sections.\n"
    "4. If any expected information is missing from the JSON data, explicitly indicate it as 'Not available'.\n"
    "5. Your output should list all extracted details in a clear, structured, and organized format without discarding any information.\n\n"
    "The JSON data is provided below. Process the data and output the complete information in a structured and readable format."
)

# ---------------------------
# Summarization/Extraction Functions
# ---------------------------

def ollama_summarize(text, model="gemma3:4b", extra_prompt=""):
    """
    Calls the external summarization engine (e.g., via Ollama) using a custom prompt.
    """
    person_name = extract_person_name(json_path)
    base_prompt = (
        f"Extract all available details about {person_name} from the following JSON data. "
        "Always focus on this person's personal and professional information (e.g., name, contact details, education, employment, projects, publications, awards, external links, etc.) "
        "and remove any data that is not related to this person."
    )
    if extra_prompt:
        base_prompt += f" Also, include these additional instructions: {extra_prompt}."
    prompt = f"{base_prompt}\n\n{text}"
    
    command = ["ollama", "run", model]
    result = subprocess.run(command, input=prompt, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error calling Ollama:", result.stderr)
        return ""
    
    return result.stdout.strip()

def generate_expert_summary(json_text):
    """
    Uses the EXPERT_PROMPT combined with the JSON data to generate a complete extraction.
    """
    prompt = f"{EXPERT_PROMPT}\n\n{json_text}"
    print("Generating expert-level extraction...")
    summary = ollama_summarize(prompt, model="gemma3:4b")
    return summary

def chunk_text(text, max_words=200):
    """
    Splits text into chunks of approximately max_words words.
    """
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

def generate_summary(data_str, extra_prompt=""):
    """
    Optionally generate a summary from the cleaned data.
    """
    chunks = chunk_text(data_str, max_words=200)
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = ollama_summarize(chunk, model="gemma3:4b", extra_prompt=extra_prompt)
        print(f"Summary for chunk {i+1}: {summary}\n")
        summaries.append(summary)
    final_summary = "\n".join(summaries)
    return final_summary

def filter_details(extra_details, json_text):
    """
    Given a comma-separated string of details, checks the JSON text and returns
    a string of details found.
    """
    details_to_check = [d.strip() for d in extra_details.split(",") if d.strip()]
    found_details = []
    json_lower = json_text.lower()
    for detail in details_to_check:
        if detail.lower() in json_lower:
            found_details.append(detail)
    return ", ".join(found_details)

# ---------------------------
# Advanced Technique: Semantic Clustering
# ---------------------------

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_extracted_details(text, threshold=0.7):
    """
    Splits the text into sentences, computes embeddings, and clusters similar sentences.
    """
    sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    if not sentences:
        return {}
    
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    clusters = {}
    used_indices = set()
    
    for i, emb in enumerate(embeddings):
        if i in used_indices:
            continue
        cluster = [sentences[i]]
        used_indices.add(i)
        for j in range(i+1, len(embeddings)):
            if j in used_indices:
                continue
            similarity = util.cos_sim(emb, embeddings[j]).item()
            if similarity >= threshold:
                cluster.append(sentences[j])
                used_indices.add(j)
        clusters[f"Cluster {i+1}"] = cluster
    return clusters

def print_clusters(clusters):
    """
    Prints the clustered details in a readable format.
    """
    print("\n--- Clustered Extracted Details ---")
    for cluster, items in clusters.items():
        print(f"{cluster}:")
        for item in items:
            print(f"  - {item}")
    print("-----------------------------------\n")

# ---------------------------
# PDF Generation Function
# ---------------------------
def save_pdf(text, filename):
    """
    Saves the provided text into a PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)
    print(f"PDF saved as {filename}")

# ---------------------------
# Helper Function to Get JSON File
# ---------------------------
def get_latest_json(directory):
    """
    Returns the most recently modified JSON file in the specified directory.
    """
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if not json_files:
        return None
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file

def select_json_file(directory):
    """
    Prompts the user to input a search term (or leave blank) to select a JSON file.
    If a term is provided, search for files containing that term.
    If multiple files match, ask the user to choose one.
    Otherwise, return the latest JSON file.
    """
    search_term = input("Enter part of the person's name to select a specific JSON file, or press Enter to use the latest file: ").strip().lower()
    json_files = glob.glob(os.path.join(directory, '*.json'))
    
    if search_term:
        matching_files = [f for f in json_files if search_term in os.path.basename(f).lower()]
        if not matching_files:
            print(f"No files found matching '{search_term}'. Using the latest JSON file instead.")
            return get_latest_json(directory)
        elif len(matching_files) == 1:
            print(f"One file found: {matching_files[0]}")
            return matching_files[0]
        else:
            print("Multiple files found:")
            for idx, file in enumerate(matching_files, start=1):
                print(f"{idx}: {file}")
            try:
                choice = int(input("Enter the number corresponding to the file you want to use: ").strip())
                if 1 <= choice <= len(matching_files):
                    return matching_files[choice - 1]
                else:
                    print("Invalid choice. Using the latest JSON file.")
                    return get_latest_json(directory)
            except ValueError:
                print("Invalid input. Using the latest JSON file.")
                return get_latest_json(directory)
    else:
        return get_latest_json(directory)

# ---------------------------
# Main Execution
# ---------------------------

# Define the directory where browser_collector.py stores JSON files.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
json_directory = os.path.join(project_root, 'data', 'json')

# Ask the user if they want to run the summarizer
user_input = input('Do you want to run the summarizer to extract details? (yes/no): ').strip().lower()

if user_input in ['yes', 'y']:
    # Ask for a specific JSON file or use the latest if no input is provided.
    json_path = select_json_file(json_directory)
    if not json_path:
        print("No JSON file found in the directory:", json_directory)
        exit(1)
    
    print(f"Using JSON file: {json_path}")

    # 1. Load the JSON file.
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Extract the target person's name.
    person_name = extract_person_name(json_path)
    print(f"Extracted person name: {person_name}")

    # 2. Clean the JSON data.
    cleaned_data = clean_json(json_data, person_name)
    cleaned_data_str = json.dumps(cleaned_data, indent=2)

    # 3. Generate an initial summary.
    print("\nGenerating initial summary from cleaned data...\n")
    final_summary = generate_summary(cleaned_data_str)
    print("Final Combined Summary:")
    print(final_summary)

    # 4. Generate an expert-level extraction.
    print("\nGenerating expert-level extraction using the provided instructions...\n")
    expert_summary = generate_expert_summary(cleaned_data_str)
    print("Expert-Level Extraction:")
    print(expert_summary)

    # 5. Cluster the expert summary.
    clusters = cluster_extracted_details(expert_summary, threshold=0.75)
    print_clusters(clusters)

    # 6. Verification Loop.
    while True:
        response = input("\nAre the key personal and professional details in the extractions correct and focused only on the required person? (Yes/No/Don't know): ").strip().lower()
        if response in ['yes', 'y']:
            print("Great! The details are verified.")
            break
        elif response in ['no', "don't know", "dont know"]:
            extra_details = input("Please list the details (comma-separated) you want to verify or remove (e.g., name, city, age, etc.): ").strip()
            details_found = filter_details(extra_details, cleaned_data_str)
            if details_found:
                print(f"Details found in cleaned JSON that need to be rechecked/removed: {details_found}")
            else:
                print("None of the provided details were found in the cleaned JSON; ignoring them.")
            print("\nRe-generating extraction with extra instructions based on the verified details...\n")
            final_summary = generate_summary(cleaned_data_str, extra_prompt=details_found)
            print("Revised Final Combined Summary:")
            print(final_summary)
        else:
            print("Please respond with 'Yes', 'No', or 'Don't know'.")

    # 7. Ask user if they want to generate a PDF.
    generate_pdf_choice = input("\nDo you want to generate a PDF for the final summary? (Yes/No): ").strip().lower()
    if generate_pdf_choice in ['yes', 'y']:
        pdf_filename = os.path.splitext(os.path.basename(json_path))[0] + ".pdf"
        save_pdf(final_summary, pdf_filename)
        print("PDF generated!")
    else:
        print("PDF generation skipped.")
else:
    print("Summarizer execution skipped.")