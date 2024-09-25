# main.py

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
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
import pytz
from flask import Flask
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.background import BackgroundScheduler
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import pickle

# Initialize Flask app
app = Flask(__name__)

# Define a simple route to keep Heroku happy
@app.route('/')
def index():
    return "Hello, I'm alive!"

# Define conversation states
ASKING_NAME, ASKING_DATE, ASKING_TIME, ASKING_LOCATION, ASKING_DURATION = range(5)

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

# Configure Google OAuth 2.0
SCOPES = ['https://www.googleapis.com/auth/calendar']
GOOGLE_CLIENT_SECRETS_FILE = 'YOUR_GOOGLE_CLIENT_SECRETS_FILE.json'  # Replace with your client secrets file

# Database setup
DATABASE_URL = os.environ.get('DATABASE_URL')  # Heroku sets this environment variable
if not DATABASE_URL:
    logger.error("Database URL not found. Please set the DATABASE_URL environment variable.")
    # For local testing, you can set DATABASE_URL to something like 'sqlite:///events.db'

engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()

# Define the Event model
class EventModel(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    event_name = Column(String)
    event_description = Column(Text)
    event_start = Column(DateTime)
    event_end = Column(DateTime)
    event_duration = Column(Float)  # Duration in minutes
    event_timezone = Column(String)
    user_id = Column(Integer)
    user_username = Column(String)

# Create tables
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.start()

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_message = (
        f"Hello, {user.first_name}! I'm Paraplan Bot 🤖\n\n"
        "Send me event details like 'Meeting with Alex on September 25th at 3 PM for 2 hours', "
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
        "Extract the event details or query intent from the following user input:\n\n"
        f"\"{user_input}\"\n\n"
        "If the user is trying to create an event, provide the event name, date, time, duration, and description in JSON format with keys 'name', 'date', 'time', 'duration', 'description'. "
        "Duration should be in minutes. If duration is not specified, return null for duration. "
        "Date and time should be in ISO 8601 format. Time in format: HH:MM:SS. "
        "If the user is requesting information about their events (e.g., 'What do I have for today?'), set 'intent' to 'retrieve' and 'period' to 'today', 'week', or 'upcoming'. "
        f"Current date and time is {current_datetime} in timezone {timezone_str}. "
        "Understand natural language dates like 'next Monday', 'tomorrow', 'in 2 hours', etc."
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
        "If the date and time are not clear, respond with 'NOT_CLEAR'. "
        f"Current date and time is {datetime.now(pytz.timezone(timezone_str)).isoformat()}. "
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

        # Handle retrieval intent
        if event_data.get('intent') == 'retrieve':
            period = event_data.get('period', 'today')
            await show_events(update, context, period=period)
            return ConversationHandler.END

        event_name = event_data.get('name')
        event_date = event_data.get('date')
        event_time = event_data.get('time')
        event_duration = event_data.get('duration')
        event_description = event_data.get('description', '')

        # Check which fields are missing
        missing_fields = []
        if not event_name:
            missing_fields.append('name')
        if not event_date:
            missing_fields.append('date')
        if not event_time:
            missing_fields.append('time')
        if not event_duration:
            missing_fields.append('duration')

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
            elif 'duration' in missing_fields:
                await update.message.reply_text("What is the duration of the event in minutes?")
                return ASKING_DURATION
        else:
            # All fields are present, proceed to create the event
            return await finalize_event_creation(update, context)

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
    elif not event_data.get('duration'):
        await update.message.reply_text("What is the duration of the event in minutes?")
        return ASKING_DURATION
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
    elif not event_data.get('duration'):
        await update.message.reply_text("What is the duration of the event in minutes?")
        return ASKING_DURATION
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
    elif not event_data.get('duration'):
        await update.message.reply_text("What is the duration of the event in minutes?")
        return ASKING_DURATION
    else:
        # All fields are present, proceed to create the event
        return await finalize_event_creation(update, context)

# Handler to ask for the event duration
async def ask_duration(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_data = context.user_data.get('event_data', {})
    duration_str = update.message.text
    try:
        duration = float(duration_str)
        event_data['duration'] = duration
        context.user_data['event_data'] = event_data
    except ValueError:
        await update.message.reply_text("Invalid duration format. Please provide the duration in minutes.")
        return ASKING_DURATION

    # All fields are present, proceed to create the event
    return await finalize_event_creation(update, context)

# Function to finalize event creation
async def finalize_event_creation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_data = context.user_data.get('event_data', {})
    event_name = event_data.get('name')
    event_date = event_data.get('date')
    event_time = event_data.get('time')
    event_duration = event_data.get('duration')
    event_description = event_data.get('description', '')

    chat_id = update.effective_chat.id
    user = update.effective_user

    # Combine date and time into a datetime object
    event_datetime_str = f"{event_date}T{event_time}"
    naive_datetime = datetime.fromisoformat(event_datetime_str)
    timezone_str = context.user_data.get('timezone', 'UTC')
    user_timezone = pytz.timezone(timezone_str)
    event_datetime = user_timezone.localize(naive_datetime)

    # Calculate event end time based on duration
    if event_duration is None:
        event_duration = 60  # Default duration in minutes
    else:
        event_duration = float(event_duration)
    event_end = event_datetime + timedelta(minutes=event_duration)

    # Create the calendar event
    c = Calendar()
    e = Event()
    e.name = event_name
    e.begin = event_datetime
    e.end = event_end
    e.description = event_description
    c.events.add(e)

    # Save the event to the database
    db_session = SessionLocal()
    new_event = EventModel(
        event_name=event_name,
        event_description=event_description,
        event_start=event_datetime,
        event_end=event_end,
        event_duration=event_duration,
        event_timezone=timezone_str,
        user_id=user.id,
        user_username=user.username,
    )
    db_session.add(new_event)
    db_session.commit()
    db_session.close()

    # Send a summary to the user
    start_time_str = event_datetime.strftime('%Y-%m-%d %H:%M')
    end_time_str = event_end.strftime('%Y-%m-%d %H:%M')
    summary_message = (
        f"**Event:** {e.name}\n"
        f"**Date and Time:** {start_time_str} - {end_time_str}\n"
        f"**Duration:** {event_duration} minutes"
    )
    await update.message.reply_text(summary_message, parse_mode='Markdown')

    # Send the .ics file to the user
    ics_file = io.BytesIO(str(c).encode('utf-8'))
    ics_file.name = f"{e.name.replace(' ', '_')}.ics"
    await context.bot.send_document(chat_id=chat_id, document=InputFile(ics_file))

    # Log the successful event creation
    logger.info(f"Event created for user {user.id} ({user.username}): {event_name} at {event_datetime}")

    # Provide user feedback
    await update.message.reply_text("Your event has been saved successfully!")

    # Sync with Google Calendar if connected
    credentials = get_user_credentials(user.id)
    if credentials:
        sync_event_to_google_calendar(new_event, credentials)

    # Schedule event reminder
    schedule_event_reminder(new_event, user.id)

    # Clear user data
    context.user_data.clear()

    return ConversationHandler.END

# Function to retrieve events
async def show_events(update: Update, context: ContextTypes.DEFAULT_TYPE, period=None):
    user = update.effective_user
    command = update.message.text.lower() if not period else f"/{period}"
    db_session = SessionLocal()

    # Determine the date range based on the command
    timezone_str = context.user_data.get('timezone', 'UTC')
    user_timezone = pytz.timezone(timezone_str)
    now = datetime.now(user_timezone)

    if '/today' in command or period == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        title = "Today's Events"
    elif '/week' in command or period == 'week':
        start = now - timedelta(days=now.weekday())
        end = start + timedelta(days=7)
        title = "This Week's Events"
    elif '/upcoming' in command or period == 'upcoming':
        start = now
        end = now + timedelta(days=30)
        title = "Upcoming Events"
    else:
        await update.message.reply_text("Invalid command.")
        return

    # Query events
    events = db_session.query(EventModel).filter(
        EventModel.user_id == user.id,
        EventModel.event_start >= start,
        EventModel.event_start < end
    ).order_by(EventModel.event_start).all()
    db_session.close()

    # Generate the message
    if events:
        message = f"**{title}:**\n"
        for event in events:
            start_time = event.event_start.strftime('%Y-%m-%d %H:%M')
            message += f"- **{event.event_name}** at {start_time} (Duration: {event.event_duration} minutes)\n"
    else:
        message = "You have no events in this period."

    await update.message.reply_text(message, parse_mode='Markdown')

# Google Calendar Integration
def get_user_credentials(user_id):
    creds = None
    token_file = f"token_{user_id}.pickle"
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    return creds

async def connect_google(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    flow = Flow.from_client_secrets_file(
        GOOGLE_CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # Use out-of-band flow
    )

    auth_url, _ = flow.authorization_url(prompt='consent')
    context.user_data['flow'] = flow
    await update.message.reply_text(f"Please authorize access by visiting this URL:\n{auth_url}\n\nThen send me the authorization code.")

    return 'AUTH_CODE'

async def receive_auth_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    auth_code = update.message.text
    flow = context.user_data.get('flow')

    if flow:
        flow.fetch_token(code=auth_code)
        creds = flow.credentials
        # Save the credentials for the user
        token_file = f"token_{user.id}.pickle"
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
        await update.message.reply_text("Google Calendar connected successfully!")
    else:
        await update.message.reply_text("Authorization flow not found. Please try again.")

    return ConversationHandler.END

def sync_event_to_google_calendar(event, credentials):
    service = build('calendar', 'v3', credentials=credentials)
    event_body = {
        'summary': event.event_name,
        'description': event.event_description,
        'start': {
            'dateTime': event.event_start.isoformat(),
            'timeZone': event.event_timezone,
        },
        'end': {
            'dateTime': event.event_end.isoformat(),
            'timeZone': event.event_timezone,
        },
    }
    service.events().insert(calendarId='primary', body=event_body).execute()

# Event Reminder Functions
def send_event_reminder(context: ContextTypes.DEFAULT_TYPE):
    job_context = context.job.context
    user_id = job_context['user_id']
    event_name = job_context['event_name']
    chat_id = job_context['chat_id']
    context.bot.send_message(chat_id=chat_id, text=f"🔔 Reminder: Your event '{event_name}' is coming up soon.")

def schedule_event_reminder(event, user_id):
    reminder_time = event.event_start - timedelta(minutes=10)  # Remind 10 minutes before
    if reminder_time < datetime.now(pytz.timezone(event.event_timezone)):
        return  # Do not schedule if the event is in the past
    job_context = {
        'user_id': user_id,
        'event_name': event.event_name,
        'chat_id': event.user_id  # Assuming chat_id is the same as user_id
    }
    scheduler.add_job(
        send_event_reminder,
        'date',
        run_date=reminder_time,
        args=[ContextTypes.DEFAULT_TYPE],
        kwargs={'job_context': job_context}
    )

# Daily Summary Function
def send_daily_summary(context: ContextTypes.DEFAULT_TYPE):
    user_id = context.job.context['user_id']
    chat_id = context.job.context['chat_id']
    # Create a fake update and context to call show_events
    class FakeUpdate:
        effective_user = type('User', (), {'id': user_id})
        effective_chat = type('Chat', (), {'id': chat_id})
        message = type('Message', (), {'text': '/today'})

    update = FakeUpdate()
    context = ContextTypes.DEFAULT_TYPE()
    context.user_data = {'timezone': 'UTC'}  # Replace with actual timezone if stored
    asyncio.run_coroutine_threadsafe(show_events(update, context, period='today'), asyncio.get_event_loop())

def schedule_daily_summary(user_id, chat_id):
    scheduler.add_job(
        send_daily_summary,
        'cron',
        hour=8,  # Sends at 8 AM every day
        args=[ContextTypes.DEFAULT_TYPE],
        kwargs={'job_context': {'user_id': user_id, 'chat_id': chat_id}}
    )

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
            ASKING_DURATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_duration)],
        },
        fallbacks=[],
    )
    application.add_handler(conversation_handler)

    # Command handlers for showing events
    application.add_handler(CommandHandler(['today', 'week', 'upcoming'], show_events))

    # Conversation handler for Google Calendar connection
    google_calendar_handler = ConversationHandler(
        entry_points=[CommandHandler('connect_google', connect_google)],
        states={
            'AUTH_CODE': [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_auth_code)],
        },
        fallbacks=[],
    )
    application.add_handler(google_calendar_handler)

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    # Start the bot in a separate thread
    main()
    # Start the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
