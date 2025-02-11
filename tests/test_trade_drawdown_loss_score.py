import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PriceSeriesGenerator:
    @staticmethod
    def random_walk(start_price: float = 100,
                    steps: int = 100,
                    volatility: float = 0.02,
                    seed: int = None) -> pd.Series:
        """Generate a random walk price series with Gaussian noise."""
        np.random.seed(seed)
        returns = np.random.normal(loc=0, scale=volatility, size=steps)
        return pd.Series(start_price * (1 + returns).cumprod())

    @staticmethod
    def trend_with_volatility(start_price: float = 100,
                              steps: int = 100,
                              trend: float = 0.001,
                              volatility: float = 0.015,
                              seed: int = None) -> pd.Series:
        """Generate a series with directional trend + volatility."""
        np.random.seed(seed)
        deterministic = np.linspace(0, trend, steps)
        stochastic = np.random.normal(scale=volatility, size=steps)
        combined = deterministic + stochastic
        return pd.Series(start_price * (1 + combined).cumprod())

    @staticmethod
    def crash_and_recovery(start_price: float = 100,
                           crash_at: int = 30,
                           crash_size: float = -0.4,
                           recovery_speed: float = 0.05,
                           volatility: float = 0.01) -> pd.Series:
        """Generate a crash event followed by partial recovery."""
        pre_crash = np.zeros(crash_at)
        crash = np.array([crash_size])
        recovery = np.linspace(recovery_speed, 0, 70)[:100 - crash_at - 1]
        combined = np.concatenate([pre_crash, crash, recovery])
        noise = np.random.normal(scale=volatility, size=len(combined))
        return pd.Series(start_price * (1 + combined + noise).cumprod())

    @staticmethod
    def composite_pattern(steps: int = 200,
                          patterns: list = [
                              (50, 0.001, 0.01),  # Slow trend
                              (30, -0.02, 0.03),  # Sharp decline
                              (70, 0.005, 0.02),  # Recovery + volatility
                              (50, 0.0, 0.04)  # Sideways market
                          ]) -> pd.Series:
        """Generate composite pattern with multiple market regimes."""
        series = []
        price = 100
        for duration, trend, vol in patterns:
            steps = min(duration, steps)
            returns = trend + np.random.normal(scale=vol, size=steps)
            price_series = price * (1 + returns).cumprod()
            series.append(price_series)
            price = price_series[-1]
            steps -= duration
            if steps <= 0: break
        return pd.Series(np.concatenate(series))


class TradeAnalysis:
    @staticmethod
    def trade_loss_drawdown_score(prices: pd.Series, loss_threshold: float = 0.089) -> float:
        """Original score calculation method (unchanged)"""
        max_drawdown = 0.0
        max_loss = 0.0
        entry_price = prices.iloc[0]
        high_price = entry_price

        for price in prices:
            if price > high_price:
                high_price = price

            current_loss = (price - entry_price) / entry_price
            drawdown = (high_price - price) / high_price
            if drawdown > max_drawdown and drawdown > -loss_threshold:
                max_drawdown = drawdown

            if current_loss < max_loss and current_loss < -loss_threshold:
                max_loss = current_loss

        return max_loss if max_drawdown < max_loss else max_drawdown


# Generate test scenarios
scenarios = {
    "Random Walk": PriceSeriesGenerator.random_walk(steps=200, volatility=0.05, seed=443),
    "Strong Uptrend": PriceSeriesGenerator.trend_with_volatility(steps=200, trend=.005, volatility=0.07, seed=445),
    "Crash & Recovery": PriceSeriesGenerator.crash_and_recovery(start_price=100, crash_at=50),
    "High Volatility": PriceSeriesGenerator.random_walk(steps=200, volatility=0.08, seed=42),
    "Composite Market": PriceSeriesGenerator.composite_pattern()
}

# Test and plot all scenarios
plt.figure(figsize=(15, 18))

for idx, (title, prices) in enumerate(scenarios.items(), 1):
    # Calculate dynamic scores
    weights = []
    pnls = []
    initial_balance = 100
    for i in range(1, len(prices)):
        pnl = prices.iloc[:i + 1].pct_change().dropna().cumsum().values[-1]

        weight = TradeAnalysis.trade_loss_drawdown_score(prices.iloc[:i + 1], loss_threshold=0.05)
        if pnl >= 0:
            pnl = pnl * (1 - weight)
        else:
            pnl = pnl * (1 + weight)
        pnls.append(pnl)
        weights.append(weight)

    # Create subplots
    ax = plt.subplot(len(scenarios), 1, idx)
    ax.plot(prices, label='Price', color='tab:blue', alpha=0.8)
    ax.set_ylabel('Price')

    # Create a twin axis for weights and PnLs
    ax2 = ax.twinx()
    ax2.plot(range(1, len(prices)), weights,
             label='Drawdown Weight',
             linestyle='--',
             color='tab:red',
             alpha=0.8)
    ax2.plot(range(1, len(prices)), pnls,
             label='PnL',
             linestyle='-',
             color='tab:orange',
             alpha=0.8)
    ax2.set_ylabel('Weight / PnL')

    ax.set_title(f"Scenario: {title}", pad=12)
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    if idx == len(scenarios):
        ax.set_xlabel('Time Steps')

plt.tight_layout()
plt.show()