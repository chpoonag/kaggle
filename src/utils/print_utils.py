import builtins
class FormattedPrint:
    def __init__(self, prefix="\t", suffix="", color=None, end="\n", font_style=None):
        """
        General purpose print formatter that auto-adds prefix/suffix, supports simple color and font styling.
        
        :param prefix: String to prepend to all print output lines. Default: tab character.
        :param suffix: String to append to all print output lines. Default: empty.
        :param color: Optional text color using ANSI escape codes (e.g. 'red', 'green', etc.).
        :param end: String appended at end of print calls. Default newline.
        :param font_style: Optional font style e.g. 'bold', 'underline'.
        """
        self.prefix = prefix
        self.suffix = suffix
        self.end = end
        self._original_print = builtins.print
        self._active = False
        self.color_codes = {
            "black": "\033[30m", "red": "\033[31m", "green": "\033[32m",
            "yellow": "\033[33m", "blue": "\033[34m", "magenta": "\033[35m",
            "cyan": "\033[36m", "white": "\033[97m",
        }
        self.style_codes = {
            "bold": "\033[1m",
            "underline": "\033[4m",
        }
        self.color = self.color_codes.get(color, "") if color else ""
        self.font_style = self.style_codes.get(font_style, "") if font_style else ""
        self.reset_code = "\033[0m" if (self.color or self.font_style) else ""

    def _formatted_print(self, *args, **kwargs):
        # Compose formatted message parts for first arg if exists and is string
        if args:
            msg = str(args[0])
            msg = f"{self.color}{self.font_style}{self.prefix}{msg}{self.suffix}{self.reset_code}"
            new_args = (msg,) + args[1:]
        else:
            # If no args, just print prefix + suffix
            new_args = (f"{self.color}{self.font_style}{self.prefix}{self.suffix}{self.reset_code}",)
        kwargs['end'] = self.end
        self._original_print(*new_args, **kwargs)

    def activate(self):
        """Override built-in print function with formatted print."""
        if not self._active:
            builtins.print = self._formatted_print
            self._active = True

    def deactivate(self):
        """Restore original print function."""
        if self._active:
            builtins.print = self._original_print
            self._active = False
