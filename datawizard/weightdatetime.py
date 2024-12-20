from datetime import datetime
from math import log
from math import exp
from typing import Union

from dbbinance.fetcher import check_convert_to_datetime

__version__ = 0.003


class DateTimeWeight:
    def __init__(self,
                 start_datetime: Union[datetime, int, str, None],
                 end_datetime: Union[datetime, int, str, None],
                 start_weight: float = 0.5,
                 end_weight: float = 1.0):
        """
        Initialize the object with start and end datetimes.

        Args:
            start_datetime (Union[datetime, int, str, None]): Start date and time.
            end_datetime (Union[datetime, int, str, None]): End date and time.
            start_weight (float, optional): Initial weight at the start time. Defaults to 0.5.
            end_weight (float, optional): Final weight at the end time. Defaults to 1.0.
        """
        self.start_datetime = check_convert_to_datetime(start_datetime, utc_aware=False)
        self.end_datetime = check_convert_to_datetime(end_datetime, utc_aware=False)
        self.total_minutes = (self.end_datetime - self.start_datetime).total_seconds() // 60
        self.start_weight = start_weight
        self.k = (log(end_weight / self.start_weight)) / self.total_minutes

    def calculate_weight(self, current_datetime: Union[datetime, int, str, None], ) -> Union[float, None]:
        """
        Calculate the weight with checking if the current time is within the defined range,
        based on the current datetime within the range of start and end times.

        Args:
            current_datetime (Union[datetime, int, str, None]): The current date and time to evaluate.
        Returns:
            float or None: Calculated weight value.
        """

        current_ = check_convert_to_datetime(current_datetime, utc_aware=False)
        elapsed_minutes = (current_ - self.start_datetime).total_seconds() // 60
        if elapsed_minutes <= 0 or elapsed_minutes >= self.total_minutes:
            return None  # weight out of datetime range

        return self.start_weight * exp(self.k * elapsed_minutes)

    def calc_weight(self, current_datetime: datetime) -> float:
        """
        Calculate the weight without checking if the current time is within the defined range.

        Args:
            current_datetime (datetime): The current date and time to evaluate.

        Returns:
            float: Calculated weight value

        """
        elapsed_minutes = (current_datetime - self.start_datetime).total_seconds() // 60

        return self.start_weight * exp(self.k * elapsed_minutes)
