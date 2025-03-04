from datetime import timedelta
from numpy import nan
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import binanceenv.orderbook
from rllab.labtools import detect_timeframe
from rllab.labmetrics import calculate_sharpe_ratio, calculate_win_rate, get_trade_metrics

from binanceenv.orderbook import Asset

__version__ = 0.005


def visualize_episode(df:pd.DataFrame, asset: Asset):
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()
    df['action'] = df['action'].astype(int)

    # Get trade metrics from environment
    metrics = get_trade_metrics(asset.orders.trades, df)

    # First pass: format all values and find maximum lengths
    formatted_values = []
    keys = []
    for key, value in metrics.items():
        keys.append(key)
        # Format value based on type
        if isinstance(value, (float, int)):
            formatted = f"{value:.2f}"
        elif isinstance(value, pd.Timedelta):
            formatted = str(value).split('.')[0]
        elif isinstance(value, pd.Timestamp):
            formatted = value.strftime('%Y-%m-%d %H:%M:%S')
        else:
            formatted = str(value)
        formatted_values.append(formatted)

    # Calculate max widths
    max_key_length = max(len(str(k)) for k in keys) + 1  # +1 for colon
    max_value_length = max(len(v) for v in formatted_values) if formatted_values else 10

    # Build metrics text with dynamic alignment
    metrics_text = "<b>TRADING METRICS</b><br>"
    for key, value_str in zip(keys, formatted_values):
        # Left-align key, right-align value
        key_part = f"{str(key):<{max_key_length}}"
        value_part = f"{value_str:>{max_value_length}}"
        metrics_text += f"{key_part} : {value_part}<br>"

    # Create subplots with modified fourth subplot
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Price and Trading Signals',
            'Portfolio Value',
            'Drawdown',
            'Position Size, Actions & Reward'
        ),
        row_heights=[0.5, 0.1, 0.1, 0.2],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}]]
    )

    # 1. Price and Actions Plot
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        name='Price',
        line=dict(color='#1f77b4'),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>Price: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Add trade markers from environment
    if asset.orders.trades.book:
        for trade in asset.orders.trades.book:
            if trade.closed:
                # Buy marker (triangle-up)
                fig.add_trace(go.Scatter(
                    x=[trade.orders.open_order.order_datetime],
                    y=[trade.orders.open_order.price],
                    mode='markers',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='triangle-up',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name='Buy',
                    hoverinfo='text',
                    hovertext=(
                        f"<b>BUY</b><br>"
                        f"Time: {trade.orders.open_order.order_datetime.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                        f"Price: {trade.orders.open_order.price:.2f}<br>"
                        f"Size: {trade.size:.4f}"
                    ),
                    showlegend=False
                ), row=1, col=1)

                # Sell marker (triangle-down)
                fig.add_trace(go.Scatter(
                    x=[trade.orders.close_order.order_datetime],
                    y=[trade.orders.close_order.price],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-down',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name='Sell',
                    hoverinfo='text',
                    hovertext=(
                        f"<b>SELL</b><br>"
                        f"Time: {trade.orders.close_order.order_datetime.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                        f"Price: {trade.orders.close_order.price:.2f}<br>"
                        f"Profit: {trade.profit:.2f}<br>"
                        f"Price diff: {trade.price_diff:.3%}<br>"
                        f"Duration: {trade.orders.close_order.order_datetime - trade.orders.open_order.order_datetime}"
                    ),
                    showlegend=False
                ), row=1, col=1)

    # 2. Portfolio Value
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['total'],
        name='Portfolio Value',
        line=dict(color='#2ca02c'),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>Portfolio: %{y:.2f}<extra></extra>'
    ), row=2, col=1)

    # 3. Drawdown
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['drawdown'],
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#d62728'),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>Drawdown: %{y:.2%}<extra></extra>'
    ), row=3, col=1)

    # 4. Combined Position Size, Actions & Reward
    # Position Size (Bar Chart)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['amount'],
        name='Position Size',
        marker_color='#17becf',
        opacity=0.7,
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>Amount: %{y:.4f}<extra></extra>'
    ), row=4, col=1)

    # Action Markers
    action_colors = {
        0: '#00CC96',  # Buy
        1: '#EF553B',  # Sell
        2: '#FFEE00',  # Hold
        3: '#636EFA'  # Wait
    }

    fig.add_trace(go.Scatter(
        x=df.index,
        y=[0.15 if a == 0 else -0.15 if a == 1 else 0.05 if a == 2 else -0.05 for a in df['action']],
        mode='markers',
        marker=dict(
            color=df['action'].map(action_colors),
            size=8,
            symbol=['triangle-up' if a == 0 else
                    'triangle-down' if a == 1 else
                    'circle' if a == 2 else
                    'square' for a in df['action']]
        ),
        name='Actions',
        hovertemplate=[
            f"<b>{df.index[i].strftime('%Y-%m-%d %H:%M:%S')}</b><br>"
            f"{['Buy', 'Sell', 'Hold', 'Wait'][int(a)]}<br>"
            f"Amount: {df['amount'].iloc[i]:.4f}"
            for i, a in enumerate(df['action'])]
    ), row=4, col=1)

    # Reward (Line Chart on Secondary Y-axis)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['reward'],
        name='Reward',
        line=dict(color='#FFA15A'),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>Reward: %{y:.8f}<extra></extra>'
    ), row=4, col=1, secondary_y=True)

    # Add annotation (same as before)
    fig.add_annotation(
        x=0.02,
        y=0.95,
        xref="paper",
        yref="paper",
        text=metrics_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(0, 0, 0, 0.7)",
        bordercolor="#2ca02c",
        borderwidth=2,
        font=dict(
            family="Courier New, monospace",
            size=13,
            color="#ffffff"
        )
    )

    # Update layout with adjusted margins
    # Adjust margins based on text width
    total_width = max_key_length + max_value_length + 3  # 3 for colon and spaces
    fig.update_layout(
        margin=dict(l=total_width * 9, t=100, b=100),  # Approximate px conversion
        height=1252,
        title_text="Crypto Trading Episode Analysis",
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    # Update axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", row=3, col=1)
    fig.update_yaxes(title_text="Position Size", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    # Update axis labels for combined subplot
    fig.update_yaxes(title_text="Position Size", row=4, col=1)
    fig.update_yaxes(title_text="Reward", secondary_y=True, row=4, col=1)

    return fig

