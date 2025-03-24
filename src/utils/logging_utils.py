import logging
import sys
import requests
import warnings
import datetime

class TimestampedFile:
    def __init__(self, file, original_stdout, enable_console_output=True):
        self.file = file
        self.original_stdout = original_stdout
        self.enable_console_output = enable_console_output  # Option to enable/disable console output
        self.buffer = ""  # Buffer to accumulate partial lines

    def write(self, message):
        # Accumulate the message in the buffer
        self.buffer += message
        
        # Split the buffer into lines
        lines = self.buffer.splitlines(keepends=True)
        
        # Process complete lines
        for line in lines[:-1]:  # All lines except the last one
            self._write_line(line)
        
        # Update the buffer with the last (possibly incomplete) line
        self.buffer = lines[-1] if lines else ""

    def _write_line(self, line):
        # Add timestamp to the line
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {line}"
        
        # Write to the file
        self.file.write(log_message)
        
        # Write to the original stdout (console) if enabled
        if self.enable_console_output:
            self.original_stdout.write(log_message)

    def flush(self):
        # Flush any remaining data in the buffer
        if self.buffer:
            self._write_line(self.buffer)
            self.buffer = ""
        
        # Flush the file and original stdout
        self.file.flush()
        if self.enable_console_output:
            self.original_stdout.flush()

def redirect_output_with_timestamps(log_file='output.log', enable_console_output=True):
    '''
    Logging the output of the script to a file with timestamps.
    Args:
        log_file (str): The path to the log file.
        enable_console_output (bool): Whether to enable console output. Default is True.
        
    Returns:
        None
    
    Example:
        redirect_output_with_timestamps(log_file='output.log', enable_console_output=True)
        print('Hello, World!')    # This will be logged to the file with a timestamp and also printed on the console.
    '''
    # Open the log file in write mode
    log_file_obj = open(log_file, 'w')
    
    # Create a TimestampedFile object that wraps the log file and original stdout
    timestamped_file = TimestampedFile(log_file_obj, sys.stdout, enable_console_output)
    
    # Redirect sys.stdout and sys.stderr to the TimestampedFile object
    sys.stdout = timestamped_file
    sys.stderr = timestamped_file


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