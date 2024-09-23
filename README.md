# Paraplan Telegram Bot with LLM Integration

Paraplan is a Telegram bot that converts user messages into calendar events and generates corresponding `.ics` files. It now includes:

- **LLM Integration**: Uses OpenAI's GPT-3.5 Turbo model to parse natural language event descriptions.
- **User Activity Logging**: Logs every user interaction for auditing and debugging.

## **Features**

- Parses complex natural language event descriptions.
- Generates `.ics` calendar files for each event.
- Sends the `.ics` file back to the user for download.
- Provides a summary of the event details.
- Logs all user activities.

## **Prerequisites**

- **Python 3.7 or higher**
- **A Telegram account**
- **A Telegram Bot API token** (obtained from [BotFather](https://t.me/BotFather))
- **An OpenAI API key** (sign up at [OpenAI API](https://platform.openai.com/))

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/paraplan-bot.git
cd paraplan-bot
