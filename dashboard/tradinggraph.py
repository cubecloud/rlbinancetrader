import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np


class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, render_range):
        self.volume = deque(maxlen=render_range)
        self.net_worth = deque(maxlen=render_range)
        self.render_data = deque(maxlen=render_range)
        self.render_range = render_range

        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16, 8))

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=self.ax1)

        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
        # self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')

        # Add paddings to make graph easier to view
        # plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

        # we need to set layers
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')

        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        # Show the graph with matplotlib
        plt.show()

    # Render the environment to the screen
    def render(self, _date, _open, _high, _low, _close, _volume, _net_worth, _trades):
        # append volume and net_worth to deque list
        self.volume.append(_volume)
        self.net_worth.append(_net_worth)

        # before appending to deque list, need to convert Date to special format
        _date = mpl_dates.date2num([pd.to_datetime(_date)])[0]
        self.render_data.append([_date, _open, _high, _low, _close])

        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8 / 24, colorup='green', colordown='red', alpha=0.8)

        # Put all dates to one list and fill ax2 sublot with volume
        date_render_range = [i[0] for i in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(date_render_range, self.volume, 0)

        # draw our net_worth graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(date_render_range, self.net_worth, color="blue")

        # beautify the x-labels (Our Date format)
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in _trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
            if trade_date in date_render_range:
                if trade['type'] == 'buy':
                    high_low = trade['low'] - 10
                    self.ax1.scatter(trade_date, high_low, c='green', label='green', s=120, edgecolors='none',
                                     marker="^")
                else:
                    high_low = trade['high'] + 10
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s=120, edgecolors='none', marker="v")

        # we need to set layers every step, because we are clearing subplots every step
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')

        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        plt.show(block=False)
        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

        """Display image with OpenCV - no interruption"""
        # # redraw the canvas
        # self.fig.canvas.draw()
        # # convert canvas to image
        # img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        #
        # # img is rgb, convert to opencv's default bgr
        # image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #
        # # display image with OpenCV or any operation you like
        # cv2.imshow("Bitcoin trading bot", image)
        #
        # if cv2.waitKey(25) & 0xFF == ord("q"):
        #     cv2.destroyAllWindows()
        #     return
