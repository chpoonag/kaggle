import logging
import sys
import requests
import warnings

def setup_logging(log_file, include_executed_command=True):
    """
    Set up logging to a specified file and optionally include the executed command in the log.

    Args:
        log_file (str): The path to the log file.
        include_executed_command (bool): Whether to include the executed command in the log. Default is True.
    """
    logging.basicConfig(filename=log_file, 
                        level=logging.DEBUG, 
                        format='%(message)s',    # '%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    if include_executed_command:
        command = ' '.join(sys.argv)
        logging.info(f"\nCommand executed: \n{command}\n")
    
    # Redirect stdout and stderr to the log file
    sys.stdout = open(log_file, 'a')
    sys.stderr = open(log_file, 'a')

def load_logger_event(log_dir):   
    """
    Load logger events from a specified directory.

    Args:
        log_dir (str): The directory containing the log events.

    Returns:
        EventAccumulator: An EventAccumulator object containing the loaded events.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc

class TelegramBot:
    def __init__(self, bot_token, chat_id):
        """
        Initialize the TelegramBot with a bot token and chat ID.

        Args:
            bot_token (str): The token for the Telegram bot.
            chat_id (str): The chat ID to send messages to.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/"

    def send_message(self, message):
        """
        Send a message to the specified chat ID using the Telegram bot.

        Args:
            message (str): The message to send.

        Returns:
            dict: The response from the Telegram API.
        """
        url = f"{self.base_url}sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            warnings.warn(f"Failed to send message: {response.text}")
        return response.json()