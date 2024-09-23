from dotenv import load_dotenv

load_dotenv()

import logging
import os
import io
import json
import google.generativeai as genai
from telegram import (
    Update,
    InputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters,
)
from dateutil import parser as date_parser
from ics import Calendar, Event
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz

from flask import Flask

# Initialize Flask app
app = Flask(__name__)

# Define a simple route to keep Heroku happy
@app.route('/')
def index():
    return "Hello, I'm alive!"


# Define conversation states
ASKING_NAME, ASKING_DATE, ASKING_TIME, ASKING_LOCATION = range(4)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='bot_activity.log',  # Log to a file
    filemode='a',
)

logger = logging.getLogger(__name__)

# Configure Gemini API
genai_api_key = os.environ.get('GEMINI_API_KEY')
if not genai_api_key:
    logger.error("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable.")
else:
    genai.configure(api_key=genai_api_key)

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_message = (
        f"Hello, {user.first_name}! I'm Paraplan Bot 🤖\n\n"
        "Send me event details like 'Meeting with Alex on September 25th at 3 PM', "
        "and I'll create a calendar event for you!"
    )

    # Check if timezone is set
    if 'timezone' not in context.user_data:
        # Add "Set Timezone" button
        keyboard = [[InlineKeyboardButton("Set Timezone", callback_data='set_timezone')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
    else:
        await update.message.reply_text(welcome_message)

    # Log the interaction
    logger.info(f"/start command received from user {user.id} ({user.username})")

# Handler for 'Set Timezone' button
async def set_timezone_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Send a message asking for location, with a keyboard that includes a button to share location
    keyboard = [[KeyboardButton("Share Location", request_location=True)]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    await query.edit_message_text("Please share your location to set your timezone.")
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Click the button below to share your location.",
        reply_markup=reply_markup,
    )

    return ASKING_LOCATION

# Handler to receive location and set timezone
async def receive_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_location = update.message.location
    if user_location:
        latitude = user_location.latitude
        longitude = user_location.longitude

        # Use TimezoneFinder to get the timezone
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=longitude, lat=latitude)

        if timezone_str:
            context.user_data['timezone'] = timezone_str
            await update.message.reply_text(
                f"Your timezone has been set to {timezone_str}.",
                reply_markup=ReplyKeyboardRemove(),
            )
            await update.message.reply_text("You can now send me event details.")
        else:
            await update.message.reply_text(
                "Sorry, I couldn't determine your timezone. Please try again.",
                reply_markup=ReplyKeyboardRemove(),
            )

        return ConversationHandler.END
    else:
        await update.message.reply_text("Please share your location to set your timezone.")
        return ASKING_LOCATION

# Function to parse event using Gemini API
def parse_event_with_llm(user_input, context):
    # Initialize the model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Get the user's timezone
    timezone_str = context.user_data.get('timezone', 'UTC')
    current_datetime = datetime.now(pytz.timezone(timezone_str)).isoformat()

    # Construct the prompt
    prompt = (
        "Extract the event details from the following user input:\n\n"
        f"\"{user_input}\"\n\n"
        "Provide the event name, date, and time in JSON format with keys 'name', 'date', and 'time'. "
        "Date and time should be in ISO 8601 format. Time in format: HH:MM:SS "
        f"Current date and time is {current_datetime} in timezone {timezone_str}."
    )

    # Generate the response with configuration
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=200,
            temperature=0.0,
        ),
    )

    content = response.text

    # Log the response from LLM
    logger.info(f"Response from LLM: {content}")

    # Extract JSON from the response
    try:
        event_data = json.loads(content)
    except json.JSONDecodeError:
        # Attempt to extract JSON from text
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            event_json = json_match.group(0)
            event_data = json.loads(event_json)
        else:
            raise ValueError("Failed to parse JSON from LLM response.")
    return event_data

# Date time parser
def llm_date_time_parser(user_input, context):
    model = genai.GenerativeModel("gemini-1.5-flash")
    timezone_str = context.user_data.get('timezone', 'UTC')
    prompt = (
        "Extract the date and time from the following user input:\n\n"
        f"\"{user_input}\"\n\n"
        "Provide the event date and time in one string. "
        "Date and time should be in ISO 8601 format. "
        "Current date and time is " + datetime.now(pytz.timezone(timezone_str)).isoformat() + ". "
        "Context: "
        "ANSWER ONLY THE DATE AND TIME IN ISO FORMAT."
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=50,
            temperature=0.0,
        ),
    )
    content = response.text

    return content.strip()

# Message handler for event creation
async def create_event(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    chat_id = update.effective_chat.id
    user = update.effective_user

    # Log the user's message
    logger.info(f"Message from user {user.id} ({user.username}): {user_input}")

    try:
        # Use LLM to parse the event details
        event_data = parse_event_with_llm(user_input, context)

        # Log the parsed event data
        logger.info(f"Parsed event data: {event_data}")

        event_name = event_data.get('name')
        event_date = event_data.get('date')
        event_time = event_data.get('time')

        # Check which fields are missing
        missing_fields = []
        if not event_name:
            missing_fields.append('name')
        if not event_date:
            missing_fields.append('date')
        if not event_time:
            missing_fields.append('time')

        if missing_fields:
            # Store the event data in context
            context.user_data['event_data'] = event_data

            # Ask for the missing fields
            if 'name' in missing_fields:
                await update.message.reply_text("What is the name of the event?")
                return ASKING_NAME
            elif 'date' in missing_fields:
                await update.message.reply_text("When?")
                return ASKING_DATE
            elif 'time' in missing_fields:
                await update.message.reply_text("What is the time of the event? Please provide in HH:MM format.")
                return ASKING_TIME
        else:
            # All fields are present, proceed to create the event
            # Combine date and time into a datetime object
            # Get the user's timezone
            timezone_str = context.user_data.get('timezone', 'UTC')
            user_timezone = pytz.timezone(timezone_str)

            event_datetime_str = f"{event_date}T{event_time}"
            naive_datetime = datetime.fromisoformat(event_datetime_str)
            event_datetime = user_timezone.localize(naive_datetime)

            # Create the calendar event
            c = Calendar()
            e = Event()
            e.name = event_name
            e.begin = event_datetime
            c.events.add(e)

            # Save the .ics file to a BytesIO object
            ics_file = io.BytesIO(str(c).encode('utf-8'))
            ics_file.name = f"{e.name.replace(' ', '_')}.ics"

            # Send a summary and the .ics file to the user
            summary_message = (
                f"**Event:** {e.name}\n"
                f"**Date and Time:** {e.begin.format('YYYY-MM-DD HH:mm')}"
            )
            await update.message.reply_text(summary_message, parse_mode='Markdown')
            await context.bot.send_document(chat_id=chat_id, document=InputFile(ics_file))

            # Log the successful event creation
            logger.info(f"Event created for user {user.id} ({user.username}): {event_name} at {event_datetime}")

            return ConversationHandler.END

    except Exception as ex:
        # Log detailed error information
        logger.error(f"Error processing event for user {user.id} ({user.username}): {ex}", exc_info=True)
        await update.message.reply_text(
            "Sorry, I couldn't understand that. Please provide the event details again."
        )
        return ConversationHandler.END

# Handler to ask for the event name
async def ask_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_data = context.user_data.get('event_data', {})
    event_data['name'] = update.message.text
    context.user_data['event_data'] = event_data

    # Check if other fields are missing
    if not event_data.get('date'):
        await update.message.reply_text("When?")
        return ASKING_DATE
    elif not event_data.get('time'):
        await update.message.reply_text("What is the time of the event? Please provide in HH:MM format.")
        return ASKING_TIME
    else:
        # All fields are present, proceed to create the event
        return await finalize_event_creation(update, context)

# Handler to ask for the event date
async def ask_date(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_data = context.user_data.get('event_data', {})
    event_date_str = update.message.text
    try:
        # Use date_time_parser to parse the date and time
        date_time_str = llm_date_time_parser(event_date_str, context)

        # Log the response from the LLM
        logger.info(f"Response from LLM for date parsing: {date_time_str}")

        if date_time_str == "NOT_CLEAR":
            await update.message.reply_text("I'm sorry, I couldn't understand the date and time. Please provide again.")
            return ASKING_DATE

        # Extract the actual date-time string from the response
        parsed_datetime = datetime.fromisoformat(date_time_str)
        event_data['date'] = parsed_datetime.date().isoformat()
        event_data['time'] = parsed_datetime.time().isoformat()
        context.user_data['event_data'] = event_data
    except Exception as ex:
        # Log the error and ask for the date again
        logger.error(f"Error parsing date and time: {ex}", exc_info=True)
        await update.message.reply_text("Invalid date and time format. Please provide the details again.")
        return ASKING_DATE

    if not event_data.get('name'):
        await update.message.reply_text("What is the name of the event?")
        return ASKING_NAME
    else:
        # All fields are present, proceed to create the event
        return await finalize_event_creation(update, context)

# Handler to ask for the event time
async def ask_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_data = context.user_data.get('event_data', {})
    event_time_str = update.message.text
    try:
        # Try to parse the time
        parsed_time = date_parser.parse(event_time_str).time()
        event_data['time'] = parsed_time.isoformat()
        context.user_data['event_data'] = event_data
    except Exception as ex:
        await update.message.reply_text("Invalid time format. Please provide the time in HH:MM format.")
        return ASKING_TIME

    if not event_data.get('name'):
        await update.message.reply_text("What is the name of the event?")
        return ASKING_NAME
    elif not event_data.get('date'):
        await update.message.reply_text("When?")
        return ASKING_DATE
    else:
        # All fields are present, proceed to create the event
        return await finalize_event_creation(update, context)

# Function to finalize event creation
async def finalize_event_creation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_data = context.user_data.get('event_data', {})
    event_name = event_data.get('name')
    event_date = event_data.get('date')
    event_time = event_data.get('time')

    chat_id = update.effective_chat.id
    user = update.effective_user

    # Combine date and time into a datetime object
    event_datetime_str = f"{event_date}T{event_time}"
    naive_datetime = datetime.fromisoformat(event_datetime_str)
    timezone_str = context.user_data.get('timezone', 'UTC')
    user_timezone = pytz.timezone(timezone_str)
    event_datetime = user_timezone.localize(naive_datetime)

    # Create the calendar event
    c = Calendar()
    e = Event()
    e.name = event_name
    e.begin = event_datetime
    c.events.add(e)

    # Save the .ics file to a BytesIO object
    ics_file = io.BytesIO(str(c).encode('utf-8'))
    ics_file.name = f"{e.name.replace(' ', '_')}.ics"

    # Send a summary and the .ics file to the user
    summary_message = (
        f"**Event:** {e.name}\n"
        f"**Date and Time:** {e.begin.format('YYYY-MM-DD HH:mm')}"
    )
    await update.message.reply_text(summary_message, parse_mode='Markdown')
    await context.bot.send_document(chat_id=chat_id, document=InputFile(ics_file))

    # Log the successful event creation
    logger.info(f"Event created for user {user.id} ({user.username}): {event_name} at {event_datetime}")

    # Clear user data
    context.user_data.clear()

    return ConversationHandler.END

# Main function to start the bot
def main():
    TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

    if not TOKEN:
        logger.error("Telegram Bot Token not found. Please set the TELEGRAM_BOT_TOKEN environment variable.")
        return

    if not genai_api_key:
        logger.error("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable.")
        return

    application = ApplicationBuilder().token(TOKEN).build()

    # log token to console
    print(f"Token: {TOKEN}")

    application.add_handler(CommandHandler('start', start))

    # Conversation handler for timezone setting
    timezone_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(set_timezone_button, pattern='^set_timezone$')],
        states={
            ASKING_LOCATION: [MessageHandler(filters.LOCATION, receive_location)],
        },
        fallbacks=[],
    )
    application.add_handler(timezone_handler)

    # Conversation handler for event creation
    conversation_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, create_event)],
        states={
            ASKING_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_name)],
            ASKING_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_date)],
            ASKING_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_time)],
        },
        fallbacks=[],
    )
    application.add_handler(conversation_handler)

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    # Start the bot in a separate thread
    main()
    # Start the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)