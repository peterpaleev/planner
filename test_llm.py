import json
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import google.generativeai as genai

from datetime import datetime

def parse_event_with_llm(user_input):
    model = genai.GenerativeModel("gemini-1.5-flash")
    current_datetime = datetime.now().isoformat()
    prompt = (
        f"Current date and time: {current_datetime}\n\n"
        "Extract the event details from the following user input:\n\n"
        f"\"{user_input}\"\n\n"
        "Provide the event name, date, and time in JSON format with keys 'name', 'date', and 'time'. "
        "Date and time should be in ISO 8601 format."
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=200,
            temperature=0.0,
        ),
    )
    content = response.text
    logger.info(f"Response from LLM: {content}")

    try:
        event_data = json.loads(content)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            event_json = json_match.group(0)
            event_data = json.loads(event_json)
        else:
            raise ValueError("Failed to parse JSON from LLM response.")
    return event_data

# Test function with example input
if __name__ == "__main__":
    user_input = "Reminder: Schedule a meeting with the team on September 25th at 2 PM."
    try:
        event_details = parse_event_with_llm(user_input)
        print("Parsed Event Details:", event_details)
    except Exception as e:
        print("Error:", str(e))
