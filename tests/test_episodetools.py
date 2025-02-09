import logging
import pandas as pd
from multiprocessing import freeze_support, get_logger

from datawizard.episodestools import prepare_episodes_start_end_lst
from dbbinance.fetcher import check_convert_to_datetime
from dbbinance.fetcher.datautils import convert_timeframe_to_freq

logger = get_logger()

if __name__ == '__main__':
    freeze_support()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('test_episodetools.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('numba').setLevel(logging.INFO)
    logging.getLogger('LoadDbIndicators').setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # _start_datetime = '2023-07-20 01:00:00'
    # _start_datetime = '2024-05-20 01:00:00'
    # _end_datetime = '2024-06-11 04:00:00'
    # _start_datetime = '2024-06-11 04:01:00'
    # _end_datetime = '2024-07-30 01:00:00'

    # _start_datetime = '2023-03-20 01:00:00'
    # _start_datetime = '2023-07-20 01:00:00'
    # _start_datetime = datetime.datetime.strptime('2024-03-01 01:00:00', Constants.default_datetime_format)

    # _end_datetime = datetime.datetime.strptime('2024-07-30 01:00:00', Constants.default_datetime_format)
    # '2023-03-20 01:00:00 - 2024-08-14 08:15:00'
    # _start_datetime = '2023-03-20 01:00:00'
    # _end_datetime = '2024-08-14 08:15:00'

    # _start_datetime = '2023-12-01 01:00:00'
    # _end_datetime = '2024-12-10 01:00:00'
    # _start_datetime = '2024-08-14 08:30:00'
    # _end_datetime = '2024-10-30 01:00:00'

    # _start_datetime = '2023-03-20 01:00:00'
    # _end_datetime = '2024-07-30 01:00:00'
    # _start_datetime = '2023-11-01 01:00:00'
    # _end_datetime = '2024-09-01 01:00:00'

    _start_datetime = '2023-07-20 01:00:00'
    _end_datetime = '2024-09-20 01:00:00'

    dates_range = pd.Series(
        index=pd.date_range(start=check_convert_to_datetime(_start_datetime), end=check_convert_to_datetime(_end_datetime),
                            freq=convert_timeframe_to_freq('1m')), dtype=int)

    episodes_start_end_lst = prepare_episodes_start_end_lst(3780*2,
                                                            dates_range,
                                                            min_timeframes_per_episode=940,
                                                            max_timeframes_per_episode=990,
                                                            timeframe='15m',
                                                            offset=None)
    print(len(episodes_start_end_lst))

    min_start = min(episodes_start_end_lst, key=lambda x: x[0])[0]
    max_end = max(episodes_start_end_lst, key=lambda x: x[1])[1]

    print("Min start:", min_start)
    print("Max end:", max_end)

    # print(episodes_start_end_lst)

