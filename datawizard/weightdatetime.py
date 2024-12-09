from datetime import datetime
from math import log
from math import exp
from typing import Union

from dbbinance.fetcher import check_convert_to_datetime

__version__ = 0.002


class DateTimeWeight:
    def __init__(self,
                 start_datetime: Union[datetime, int, str, None],
                 end_datetime: Union[datetime, int, str, None]):
        self.start_datetime = check_convert_to_datetime(start_datetime, utc_aware=False)
        self.end_datetime = check_convert_to_datetime(end_datetime, utc_aware=False)
        self.total_minutes = (self.end_datetime - self.start_datetime).total_seconds() // 60

    def calculate_weight(self, current_datetime: Union[datetime, int, str, None], start_weight=0.5, end_weight=1.0):
        current_ = check_convert_to_datetime(current_datetime, utc_aware=False)
        elapsed_minutes = (current_ - self.start_datetime).total_seconds() // 60

        if elapsed_minutes <= 0 or elapsed_minutes >= self.total_minutes:
            return None  # Вес вне диапазона дат

        k = (log(end_weight / start_weight)) / self.total_minutes

        return start_weight * exp(k * elapsed_minutes)
