import datetime

class DateTime:
    @staticmethod
    def get_current_date() -> str:
        """
        return current date in string format.     
        """
        return datetime.datetime.now().strftime('%B %d, %Y')
