import os
import json
import glob
import logging
import requests
from langchain.llms.base import LLM
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Preformatted

# DeepSeek API Configuration
DEEPSEEK_API_KEY = "sk-04ece6a417ad43b9aa1b4873b2d20873"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ---------------------------
# Utility Functions
# ---------------------------
def extract_person_name(filename):
    """Extracts a person's name from the filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    name = " ".join(parts[:2]) if len(parts) >= 2 else base
    logger.info(f"Extracted person name: {name}")
    return name

def clean_json(data):
    """Recursively cleans JSON data by removing keys like 'images'."""
    if isinstance(data, dict):
        return {k: clean_json(v) for k, v in data.items() if k.lower() != "images"}
    elif isinstance(data, list):
        return [clean_json(item) for item in data]
    return data

def select_json_file(directory):
    """Allows the user to select a JSON file from the directory."""
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if not json_files:
        logger.info("No JSON files found in the directory.")
        return None
    search_term = input("Enter name part or press Enter for latest: ").strip()
    if search_term:
        matches = [f for f in json_files if search_term.lower() in f.lower()]
        if matches:
            return matches[0]
        else:
            logger.info("No matches found. Using latest file.")
            return max(json_files, key=os.path.getmtime)
    else:
        return max(json_files, key=os.path.getmtime)

def save_pdf_reportlab(text, filename):
    """Generates a PDF report using ReportLab."""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    preformatted_text = Preformatted(text, style)
    try:
        doc.build([preformatted_text])
        logger.info(f"PDF saved as {filename} using ReportLab.")
    except Exception as e:
        logger.error(f"Error saving PDF with ReportLab: {e}")

# ---------------------------
# Custom LangChain LLM Using DeepSeek API
# ---------------------------
class DeepSeekLLM(LLM):
    @property
    def _llm_type(self):
        return "deepseek"
    
    def _call(self, prompt: str, stop=None):
        messages = [{
            "role": "user",
            "content": prompt
        }]
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": messages,
                    "temperature": 0.5,
                    "max_tokens": 3000,
                    "top_p": 0.9
                },
                timeout=120
            )
            response.raise_for_status()
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                result = response_data["choices"][0].get("message", {}).get("content", "").strip()
                return result
            else:
                logger.error("DeepSeek API returned no choices.")
                return ""
        except Exception as e:
            logger.error(f"DeepSeek API Error in DeepSeekLLM: {e}")
            return ""

# ---------------------------
# Chat Manager Class
# ---------------------------
class ChatManager:
    def __init__(self):
        self.llm = DeepSeekLLM()
    
    def generate_expert_details_extraction(self, json_text, json_path):
        """
        Uses DeepSeekLLM to generate an expert extraction from the JSON data.
        """
        person_name = extract_person_name(json_path)
        EXPERT_PROMPT = (
            f"Act as a senior intelligence analyst. Comprehensively analyze the JSON data about {person_name} and:\n"
            "1. Extract ALL personal/professional details\n"
            "2. Maintain strict section structure\n"
            "3. Include all URLs verbatim\n"
            "4. Highlight connections between entities\n"
            "5. Identify potential data gaps\n"
            "6. Rate data reliability 1-10\n\n"
            "Format: Markdown sections with tables where appropriate."
        )
        combined_text = f"{EXPERT_PROMPT}\n\n{json_text}"
        expert_output = self.llm(combined_text)
        return expert_output
    
    def generate_chain_of_thought_extraction(self, json_text, json_path):
        """
        Uses DeepSeekLLM to generate a chain-of-thought reasoning extraction.
        """
        COT_PROMPT = (
            "Analyze step-by-step: 1. Identify key entities 2. Map relationships 3. Verify sources 4. Assess credibility 5. Structure findings"
        )
        combined_text = f"{COT_PROMPT}\n\n{json_text}"
        cot_output = self.llm(combined_text)
        return cot_output

    def agent_reinforcement_learning(self, chain_output, json_text, json_path):
        """
        Uses DeepSeekLLM to evaluate the chain-of-thought extraction.
        Returns a score, missing elements, improvements, and flags any speculative content.
        """
        RL_PROMPT = (
            "As a senior data quality auditor, evaluate this extraction:\n"
            "1. Assign a score between 0 and 1 for completeness/accuracy\n"
            "2. List critical missing elements\n"
            "3. Provide an improved version of the extraction\n"
            "4. Flag any speculative content\n\n"
            "Use format:\nScore: X/1\nMissing Elements: [list]\nImprovements: [text]\nSpeculative Content: [list]"
        )
        combined_text = f"Extraction:\n{chain_output}\n\nSource Data:\n{json_text}"
        rl_output = self.llm(f"{RL_PROMPT}\n\n{combined_text}")
        return rl_output

    def agent_critic(self, expert_extraction, json_text, json_path):
        """
        Uses DeepSeekLLM to verify the expert extraction against the JSON data.
        """
        critic_prompt = (
            "You are a critic tasked with verifying the comprehensiveness of the following data extraction from JSON data.\n"
            "Review the extracted details and the cleaned JSON data below, and check if the extraction contains all key personal and "
            "professional details. If any important detail is missing, list them; if all details are present, respond with 'All details are present.'"
        )
        combined_text = f"Extracted Details:\n{expert_extraction}\n\nCleaned JSON Data:\n{json_text}"
        critic_output = self.llm(f"{critic_prompt}\n\n{combined_text}")
        return critic_output

    def run_agents(self, json_text, json_path):
        """
        Runs all four agents sequentially and returns the final combined output.
        """
        expert_output = self.generate_expert_details_extraction(json_text, json_path)
        cot_output = self.generate_chain_of_thought_extraction(json_text, json_path)
        rl_output = self.agent_reinforcement_learning(cot_output, json_text, json_path)
        critic_output = self.agent_critic(expert_output, json_text, json_path)
        
        final_output = (
            f"=== Expert Extraction ===\n{expert_output}\n\n"
            f"=== Chain-of-Thought Extraction ===\n{cot_output}\n\n"
            f"=== Reinforcement Learning Analysis ===\n{rl_output}\n\n"
            f"=== Critic Evaluation ===\n{critic_output}"
        )
        return final_output

# ---------------------------
# Main Process
# ---------------------------
if __name__ == "__main__":
    # Update this path to point to your JSON files directory.
    json_directory = "/Users/vinaychaudhari/Documents/Face_Detection/SearchEngine_Face_Detection/SearchEngine_Face_Detection/data/json/"
    json_file = select_json_file(json_directory)
    if not json_file:
        logger.error("No valid JSON file selected.")
        exit(1)
    
    with open(json_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Clean the JSON data
    cleaned_data = clean_json(raw_data)
    cleaned_str = json.dumps(cleaned_data, indent=2)
    
    # Initialize ChatManager and run all agents
    chat_manager = ChatManager()
    final_data = chat_manager.run_agents(cleaned_str, json_file)
    
    print("Final Combined Output:")
    print(final_data)
    
    # Ask user if they want to save the final output as PDF using ReportLab
    save_choice = input("Would you like to save the final combined output as a PDF? (y/n): ").strip().lower()
    if save_choice == "y":
        pdf_filename = input("Enter the filename for the PDF (e.g., final_report.pdf): ").strip()
        if not pdf_filename:
            pdf_filename = "final_report.pdf"
        # Ensure the filename ends with ".pdf"
        if not pdf_filename.lower().endswith(".pdf"):
            pdf_filename += ".pdf"
        save_pdf_reportlab(final_data, pdf_filename)
        print(f"Final output saved as PDF: {pdf_filename}")
    else:
        print("Final output was not saved as a PDF.")
