import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        width=1000,
        height=600,
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
            colorscale="inferno",
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


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Compute and plot a pairwise Pearson correlation matrix between:
      - Load [MW]
      - Wind Speed (WS50M)
      - Wind Direction Sin/Cos components (WD50M decomposed)
      - Hour of day
      - Month of year

    Wind direction is decomposed into sin/cos to handle its circular nature.
    """
    ts = pd.to_datetime(df["Time"])

    wd_rad = np.deg2rad(df["WD50M"])

    features = pd.DataFrame(
        {
            "Load (MW)": df["Load [MW]"].values,
            "Wind Speed": df["WS50M"].values,
            "Wind Dir (Northerlies)": np.sin(np.deg2rad(df["WD50M"])),
            "Wind Dir (Easterlies)": np.cos(np.deg2rad(df["WD50M"])),
            # Hour: full cycle = 2π/24
            "Hour (midnight)": np.cos(2 * np.pi * ts.dt.hour / 24),  # +1 at midnight, -1 at noon
            "Hour (dawn)": np.sin(2 * np.pi * ts.dt.hour / 24),  # +1 at 6am, -1 at 6pm
            # Month: full cycle = 2π/12
            "Season (summer)": np.cos(2 * np.pi * (ts.dt.month - 1) / 12),  # +1 Jan, -1 Jul
            "Season (autumn)": np.sin(2 * np.pi * (ts.dt.month - 1) / 12),
        }
    )

    corr = features.corr(method="pearson")

    labels = corr.columns.tolist()
    z = corr.values

    # Annotation text: show value, blank out the diagonal
    text = [
        [f"{z[i, j]:.2f}" if i != j else "" for j in range(len(labels))]
        for i in range(len(labels))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,  # centre the colorscale at zero
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Pearson r", side="right"),
                thickness=18,
                len=0.85,
                tickvals=[-1, -0.5, 0, 0.5, 1],
            ),
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        title=dict(text="Feature Correlation Matrix", x=0.5, font=dict(size=16)),
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=False, autorange="reversed"),
        width=620,
        height=580,
        margin=dict(l=120, r=80, t=70, b=100),
    )

    return fig


def plot_seasonal_wind_roses(df: pd.DataFrame) -> go.Figure:
    """
    Side-by-side wind roses for summer (Dec–Feb) and winter (Jun–Aug)
    to make seasonal directional shifts unambiguous.
    """
    ts = pd.to_datetime(df["Time"])
    df = df.copy()
    df["month"] = ts.dt.month

    summer_mask = df["month"].isin([12, 1, 2])
    winter_mask = df["month"].isin([6, 7, 8])

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=("Summer (Dec–Feb)", "Winter (Jun–Aug)"),
    )

    for col, mask, label in [
        (1, summer_mask, "Summer"),
        (2, winter_mask, "Winter"),
    ]:
        directions = df.loc[mask, "WD50M"].to_numpy()
        speeds = df.loc[mask, "WS50M"].to_numpy()
        N = mask.sum()

        n_dir = 16
        bin_width = 360 / n_dir
        dir_bins = np.arange(0, 360, bin_width)
        n_speed_bins = 6
        speed_edges = np.linspace(0, speeds.max(), n_speed_bins + 1)
        colorscale = px.colors.sequential.Plasma

        for i in range(n_speed_bins):
            mask_s = (speeds >= speed_edges[i]) & (speeds < speed_edges[i + 1])
            if i == n_speed_bins - 1:
                mask_s = (speeds >= speed_edges[i]) & (speeds <= speed_edges[i + 1])

            freqs = [
                np.sum(mask_s & (((directions % 360) >= d) & ((directions % 360) < d + bin_width)))
                / N
                * 100
                for d in dir_bins
            ]
            color_idx = int(i / (n_speed_bins - 1) * (len(colorscale) - 1))
            fig.add_trace(
                go.Barpolar(
                    r=freqs,
                    theta=dir_bins,
                    width=bin_width,
                    marker_color=colorscale[color_idx],
                    name=f"{speed_edges[i]:.1f}–{speed_edges[i + 1]:.1f} m/s"
                    if col == 1
                    else None,
                    showlegend=(col == 1),
                    legendgroup=f"speed_{i}",
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title=dict(text="Seasonal Wind Rose Comparison", x=0.5),
        polar=dict(
            radialaxis=dict(ticksuffix="%", angle=90, tickfont_size=10),
            angularaxis=dict(direction="clockwise", rotation=90),
        ),
        polar2=dict(
            radialaxis=dict(ticksuffix="%", angle=90, tickfont_size=10),
            angularaxis=dict(direction="clockwise", rotation=90),
        ),
        legend=dict(title="Wind Speed", x=1.05),
        width=850,
        height=480,
    )
    return fig
