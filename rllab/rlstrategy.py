from backtesting import Strategy
from binanceenv.actionspace import actions_4_reversed_dict

__version__ = 0.004


class RLStrategyBase(Strategy):
    def _check_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        return params

    def init(self):
        super().init()

    def next(self):
        action_code = self.data.Action[-1]
        action_text = actions_4_reversed_dict[action_code]
        amount = self.data.Amount[-1]
        # commission = abs(amount) * self.data.Close[-1] * self.commission

        if action_text == 'Buy':
            self.buy()
            # self.buy(size=amount)
            # pnl = amount * self.data.Close[-1] - commission
        elif action_text == 'Sell':
            self.position.close()
            # self.sell(size=amount)
            # pnl = -amount * self.data.Close[-1] - commission
        elif action_text == 'Close':
            self.position.close()
            # pnl = self.position.pl - commission
        else:
            pass
            # pnl = 0

        # Update PNL data
        # self.data.Pnl[-1] = pnl


class RLActionStrategy(RLStrategyBase):
    pass
