import pandas as pd
import plotly.graph_objects as go


def plot_load_timeseries(df: pd.DataFrame, rolling_window: int = 7 * 24) -> go.Figure:
    """
    Plot hourly load over the year with a rolling average overlay.

    Parameters
    ----------
    df             : DataFrame with columns 'Time' and 'Load [MW]'
    rolling_window : hours for rolling mean (default 7 days)
    """
    ts = pd.to_datetime(df["Time"])
    load = df["Load [MW]"].to_numpy()
    series = pd.Series(load, index=ts)
    rolling = series.rolling(rolling_window, center=True).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=load,
            mode="lines",
            name="Load",
            line=dict(color="#00d4ff", width=1),
            opacity=0.4,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling.values,
            mode="lines",
            name=f"Rolling Mean ({rolling_window // 24}d)",
            line=dict(color="#ff6b35", width=2.5),
        )
    )
    fig.update_layout(

        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Load (MW)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    return fig


def plot_load_heatmap(df: pd.DataFrame, agg: str = "mean") -> go.Figure:
    """
    Heatmap of load by month (y) and hour of day (x).

    Parameters
    ----------
    df  : DataFrame with columns 'Time' and 'Load [MW]'
    agg : aggregation per (month, hour) cell — 'mean', 'max', 'min', 'median'
    """
    ts = pd.to_datetime(df["Time"])
    grid = (
        pd.DataFrame({"load": df["Load [MW]"].values, "month": ts.dt.month, "hour": ts.dt.hour})
        .groupby(["month", "hour"])["load"]
        .agg(agg)
        .unstack("hour")
        .reindex(index=range(1, 13))
        .reindex(columns=range(24))
    )

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    z = grid.values

    hover = [
        [
            f"<b>{month_names[m]}  {hour_labels[h]}</b><br>{agg.capitalize()}: {z[m, h]:.2f} MW"
            for h in range(24)
        ]
        for m in range(12)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=hour_labels,
            y=month_names,
            colorscale="Turbo",
            hoverinfo="text",
            text=hover,
            colorbar=dict(
                title=dict(text=f"{agg.capitalize()} Load (MW)", side="right"),
                thickness=18,
                len=0.85,
            ),
            xgap=1.5,
            ygap=1.5,
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Load Heatmap – {agg.capitalize()} by Month & Hour", x=0.5, font=dict(size=16)
        ),
        xaxis=dict(
            title="Hour of Day",
            tickmode="array",
            tickvals=hour_labels[::3],
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
            showgrid=False,
        ),
        yaxis=dict(title="Month", autorange="reversed", showgrid=False),
        margin=dict(l=60, r=100, t=70, b=60),
        height=520,
    )
    return fig
