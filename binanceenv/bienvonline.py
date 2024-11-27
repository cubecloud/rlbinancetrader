from binanceenv.bienv import BinanceEnvCash
from datawizard.dataprocessor import OnlineProcessorBase  # Ensure this is the correct import for OnlineDbProcessor

class BinanceEnvOnline(BinanceEnvCash):
    def __init__(self,
                 timeframe: str,
                 discretization: str,
                 symbol_pair: str = 'BTCUSDT',
                 market: str = 'spot',
                 target_balance: float = 100_000.,
                 target_minimum_trade: float = 5.,
                 target_maximum_trade: float = 100.,
                 verbose: int = 0,
                 use_period: str = 'train',
                 **kwargs):
        super().__init__(data_processor_kwargs=kwargs,  # Pass any necessary kwargs to the base class
                         target_balance=target_balance,
                         target_minimum_trade=target_minimum_trade,
                         target_maximum_trade=target_maximum_trade)

        self.timeframe = timeframe
        self.discretization = discretization
        self.symbol_pair = symbol_pair
        self.market = market
        self.verbose = verbose
        self.use_period = use_period

        # Initialize the OnlineDbProcessor
        self.data_processor = OnlineProcessorBase(
            timeframe=self.timeframe,
            discretization=self.discretization,
            symbol_pair=self.symbol_pair,
            market=self.market,
            verbose=self.verbose
        )

    def get_new_data(self):
        # Implement logic to fetch new online data
        # This could involve calling methods from self.data_processor
        self.ohlcv_df, self.indicators_df = self.data_processor.get_ohlcv_and_indicators_sample(
            timedelta='1d',  # Adjust as necessary
            index_type='target_time'
        )

    def step(self, action):
        # Fetch new data before taking a step
        self.get_new_data()
        return super().step(action)

    def reset(self):
        # Fetch new data before resetting the environment
        self.get_new_data()
        return super().reset()