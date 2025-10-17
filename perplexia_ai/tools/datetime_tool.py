import datetime

class DateTimeTool:
    @staticmethod
    def answer_datetime(question: str) -> str:
        """
        Answer date/time questions. If the question is about today's date, return the current date.
        Otherwise, return a message indicating that only current date is supported.
        Args:
            question: The user's question about date/time.
        """
        if 'date today' in question.lower() or 'today' in question.lower():
            return datetime.datetime.now().strftime('%B %d, %Y')
        return "Sorry, only current date queries are supported by the DateTimeTool."
