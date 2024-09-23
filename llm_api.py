import os
import google.generativeai as genai
from telegram import InlineKeyboardButton

# 6841024613:AAHbvsT_g93YRRQZKUsDOVUje6QaXwO2bsU

# Configure the API key
os.environ["API_KEY"] = "<YOUR_API_KEY>"  # Set this to your actual API key or better, load it from environment variables
genai.configure(api_key=os.environ["API_KEY"])

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_buttons(input_text):
    """
    Generates button labels using the Gemini API based on the input text from the user.
    """
    try:
        # Generate responses for buttons
        date_response = model.generate_content(f"Suggest a date for: {input_text}")
        time_response = model.generate_content(f"Suggest a time for: {input_text}")
        details_response = model.generate_content("Suggest what details might be needed for an event.")
        draft_response = "Save as Draft 📁"

        # Construct buttons
        buttons = [
            [InlineKeyboardButton(date_response.text, callback_data='set_date')],
            [InlineKeyboardButton(time_response.text, callback_data='set_time')],
            [InlineKeyboardButton(details_response.text, callback_data='add_details')],
            [InlineKeyboardButton(draft_response, callback_data='save_draft')]
        ]
        return buttons
    except Exception as e:
        print(f"Error generating buttons: {e}")
        # Fallback buttons in case of failure
        return [
            [InlineKeyboardButton("Set Date 📅", callback_data='set_date')],
            [InlineKeyboardButton("Set Time ⏰", callback_data='set_time')],
            [InlineKeyboardButton("Add Details ✏️", callback_data='add_details')],
            [InlineKeyboardButton("Save as Draft 📁", callback_data='save_draft')]
        ]

if __name__ == "__main__":
    # Test function
    test_input = "Meet with Sonya about the new project"
    print(generate_buttons(test_input))