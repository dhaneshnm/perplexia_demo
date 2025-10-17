import datetime

class DateTimeTool:
    """A simple DateTimeTool tool for giving current date."""
    @staticmethod
    def answer_datetime() -> str:
        print(datetime.datetime.now().strftime('%B %d, %Y'))
        return datetime.datetime.now().strftime('%B %d, %Y')
