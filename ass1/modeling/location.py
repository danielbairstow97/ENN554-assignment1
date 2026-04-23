from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from py_wake.site.xrsite import XRSite
from scipy import integrate
from scipy.stats import weibull_min
import xarray as xr

from ass1.modeling.turbine import WindTurbine


# ---------------------------------------------------------------------------
# WindResourceModel  –  backed by a PyWake XRSite
# ---------------------------------------------------------------------------
class WindResourceModel:
    """
    Wraps a PyWake XRSite built from per-sector Weibull fits.

    Parameters
    ----------
    site        : PyWake XRSite (direction-dependent Weibull)
    shape       : global (omnidirectional) Weibull shape  k
    scale       : global (omnidirectional) Weibull scale  λ  (m/s)

    The global k/λ are used for single-turbine AEP and all plots that show
    a single wind-speed distribution.  The XRSite carries the full
    directional resource and is used for wind-farm simulations.
    """

    def __init__(self, site: XRSite, shape: float, scale: float) -> None:
        self.site = site
        self.shape = shape
        self.scale = scale

    # ------------------------------------------------------------------
    # PyWake site passthrough
    # ------------------------------------------------------------------
    @property
    def pywake_site(self) -> XRSite:
        """Return the underlying PyWake XRSite for use in farm simulations."""
        return self.site

    # ------------------------------------------------------------------
    # Probability helpers  (global omnidirectional Weibull)
    # ------------------------------------------------------------------
    def pdf(self, ws: np.ndarray) -> np.ndarray:
        return weibull_min.pdf(ws, self.shape, loc=0, scale=self.scale)

    def cdf(self, ws: np.ndarray) -> np.ndarray:
        return weibull_min.cdf(ws, self.shape, loc=0, scale=self.scale)

    def ppf(self, q: float) -> float:
        """Percent-point function (inverse CDF)."""
        return float(weibull_min.ppf(q, self.shape, loc=0, scale=self.scale))

    def generate(self, num_samples: int) -> np.ndarray:
        return self.scale * np.random.weibull(self.shape, num_samples)

    # ------------------------------------------------------------------
    # Single-turbine AEP
    # ------------------------------------------------------------------
    def aep(self, turbine: WindTurbine, n_points: int = 1000) -> float:
        """
        Annual Energy Production (kWh/year) for a single turbine.

            AEP = 8760 · ∫₀^cut_out  P(U) · f(U) dU

        Uses Simpson's rule on a fine grid.
        """
        ws_grid = np.linspace(0, turbine.cut_out_wind_speed, n_points)
        power = turbine.power_at(ws_grid)
        pdf_vals = self.pdf(ws_grid)
        return 8760.0 * float(integrate.simpson(power * pdf_vals, x=ws_grid))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_power_distribution(self, turbine: WindTurbine, n_points: int = 1000):
        ws_grid = np.linspace(0, turbine.cut_out_wind_speed * 1.05, n_points)
        pdf_val = self.pdf(ws_grid)
        power = turbine.power_at(ws_grid)
        power_dist = power * pdf_val

        aep_val = self.aep(turbine)
        mean_power_kw = aep_val / 8760

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=ws_grid,
                y=pdf_val,
                mode="lines",
                name="Weibull PDF  f(U)",
                line=dict(color="#00d4ff", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.10)",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=ws_grid,
                y=power,
                mode="lines",
                name="Power curve  P(U)",
                line=dict(color="#a78bfa", width=2, dash="dot"),
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=ws_grid,
                y=power_dist,
                mode="lines",
                name="P(U)·f(U)  (AEP integrand)",
                line=dict(color="#ff6b35", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(255,107,53,0.12)",
            ),
            secondary_y=True,
        )

        for speed, label, color in [
            (turbine.cut_in_wind_speed, "Cut-in", "#34d399"),
            (turbine.rated_wind_speed, "Rated", "#fbbf24"),
            (turbine.cut_out_wind_speed, "Cut-out", "#f87171"),
        ]:
            fig.add_vline(
                x=speed,
                line=dict(color=color, width=1.5, dash="dash"),
                annotation_text=f"<b>{label}</b><br>{speed} m/s",
                annotation_position="top",
                annotation_font=dict(color=color, size=11),
            )

        fig.update_layout(
            title=dict(
                text=(
                    f"Power Distribution vs Weibull Wind Resource for {turbine.name}<br>"
                    f"<sup>k={self.shape:.2f}, λ={self.scale:.2f} m/s  |  "
                    f"AEP = {aep_val / 1e6:.2f} GWh/yr  |  "
                    f"Mean power = {mean_power_kw:.1f} kW</sup>"
                ),
                x=0.5,
            ),
            xaxis=dict(
                title="Wind Speed (m/s)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            legend=dict(x=0.65, y=0.95),
            hovermode="x unified",
        )
        fig.update_yaxes(
            title_text="Probability Density  f(U)",
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        )
        fig.update_yaxes(title_text="Power (kW)", secondary_y=True, showgrid=False)
        return fig

    def plot_cumulative_distributions(self, turbines: list[WindTurbine], n_points: int = 500):
        max_cutout = max(t.cut_out_wind_speed for t in turbines)
        ws_grid = np.linspace(0, max_cutout * 1.05, n_points)
        wind_cdf = self.cdf(ws_grid)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Wind Speed CDF", "Power CDF"))

        COLORS = ["#ff6b35", "#a78bfa", "#34d399", "#fbbf24", "#f87171"]

        # ---- Wind speed CDF -------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=ws_grid,
                y=wind_cdf,
                mode="lines",
                name="Wind CDF",
                line=dict(color="#00d4ff", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.08)",
            ),
            row=1,
            col=1,
        )

        for i, turbine in enumerate(turbines):
            color = COLORS[i % len(COLORS)]
            for speed, marker in [
                (turbine.cut_in_wind_speed, "▷"),
                (turbine.cut_out_wind_speed, "◁"),
            ]:
                cdf_at = float(self.cdf(np.array([speed]))[0])
                fig.add_trace(
                    go.Scatter(
                        x=[speed],
                        y=[cdf_at],
                        mode="markers+text",
                        marker=dict(color=color, size=9, symbol="circle"),
                        text=[f"{marker} {turbine.name}"],
                        textposition="top right",
                        textfont=dict(size=10, color=color),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

        # ---- Power output CDF (proper mixed distribution) -------------
        for i, turbine in enumerate(turbines):
            color = COLORS[i % len(COLORS)]

            p_zero = float(
                self.cdf(np.array([turbine.cut_in_wind_speed]))[0]
                + (1 - self.cdf(np.array([turbine.cut_out_wind_speed]))[0])
            )
            p_rated = float(
                1
                - self.cdf(np.array([turbine.rated_wind_speed]))[0]
                - (1 - self.cdf(np.array([turbine.cut_out_wind_speed]))[0])
            )

            ws_op = np.linspace(turbine.cut_in_wind_speed, turbine.rated_wind_speed, n_points)
            power_op = turbine.power_at(ws_op)
            wind_cdf_op = self.cdf(ws_op)
            power_cdf_cont = p_zero + (wind_cdf_op - wind_cdf_op[0])

            power_pts = (
                [0, 0] + power_op.tolist() + [turbine.rated_power_kw, turbine.rated_power_kw]
            )
            cdf_pts = [0, p_zero] + power_cdf_cont.tolist() + [1.0 - p_rated, 1.0]

            capacity_factor = (
                1
                - p_zero
                - p_rated
                + integrate.simpson(power_cdf_cont, x=power_op) / turbine.rated_power_kw
            )

            fig.add_trace(
                go.Scatter(
                    x=power_pts,
                    y=cdf_pts,
                    mode="lines",
                    name=f"{turbine.name}  (CF={capacity_factor:.1%})",
                    line=dict(color=color, width=2.5),
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=[0, turbine.rated_power_kw],
                    y=[p_zero, 1.0],
                    mode="markers",
                    marker=dict(color=color, size=8, symbol="circle"),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        p50_ws = self.ppf(0.5)
        fig.add_vline(
            x=p50_ws,
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
            annotation_text=f"P50: {p50_ws:.1f} m/s",
            annotation_font=dict(color="rgba(255,255,255,0.5)", size=10),
            row=1,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text=(
                    f"Cumulative Distributions — Weibull Resource "
                    f"(k={self.shape:.2f}, λ={self.scale:.2f} m/s)"
                ),
                x=0.5,
            ),
            legend=dict(x=0.52, y=0.15),
            hovermode="x unified",
            height=500,
        )
        fig.update_xaxes(
            title_text="Wind Speed (m/s)", showgrid=True, gridcolor="rgba(255,255,255,0.08)", col=1
        )
        fig.update_xaxes(
            title_text="Turbine Output (kW)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            col=2,
        )
        fig.update_yaxes(
            title_text="Cumulative Probability",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            range=[0, 1.02],
        )
        return fig


# ---------------------------------------------------------------------------
# WindResourceData  –  raw time-series + Weibull fitting → WindResourceModel
# ---------------------------------------------------------------------------
class WindResourceData:
    """
    Holds the raw hourly time-series and builds a PyWake-backed
    WindResourceModel from it via per-sector Weibull fitting.
    """

    #: Number of compass sectors for directional Weibull fitting
    N_SECTORS: int = 12

    def __init__(
        self,
        datetime: pd.Series,
        wind_speed: pd.Series,
        wind_direction: pd.Series,
    ) -> None:
        self.datetime = datetime
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    # ------------------------------------------------------------------
    # Weibull fitting
    # ------------------------------------------------------------------

    def create_weibull(self) -> tuple[float, float]:
        """Fit an omnidirectional two-parameter Weibull via MLE."""
        ws = np.array(self.wind_speed)
        ws = ws[ws >= 0]
        shape, _, scale = weibull_min.fit(ws, floc=0)
        return float(shape), float(scale)

    def _fit_sector_weibulls(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit one Weibull per compass sector.

        Returns
        -------
        wd_centres  : (N_SECTORS,) sector centre bearings in degrees
        sector_freq : (N_SECTORS,) probability of wind from each sector
        A_arr       : (N_SECTORS,) Weibull scale λ per sector
        k_arr       : (N_SECTORS,) Weibull shape k per sector
        """
        ws = np.array(self.wind_speed)
        wd = np.array(self.wind_direction) % 360

        n = self.N_SECTORS
        sector_size = 360.0 / n
        wd_centres = np.linspace(0, 360, n, endpoint=False)

        sector_freq = np.zeros(n)
        A_arr = np.zeros(n)
        k_arr = np.zeros(n)

        global_k, global_A = self.create_weibull()

        for i, centre in enumerate(wd_centres):
            low = (centre - sector_size / 2) % 360
            high = (centre + sector_size / 2) % 360

            mask = (wd >= low) & (wd < high) if low < high else (wd >= low) | (wd < high)

            sector_freq[i] = mask.sum() / len(ws)
            ws_sector = ws[mask & (ws > 0)]

            if len(ws_sector) < 10:
                # Too few samples in this sector — fall back to global fit
                k_arr[i] = global_k
                A_arr[i] = global_A
            else:
                k, _, A = weibull_min.fit(ws_sector, floc=0)
                k_arr[i] = float(k)
                A_arr[i] = float(A)

        sector_freq /= sector_freq.sum()  # normalise
        return wd_centres, sector_freq, A_arr, k_arr

    # ------------------------------------------------------------------
    # Build PyWake site + WindResourceModel
    # ------------------------------------------------------------------
    def create_wind_model(self) -> WindResourceModel:
        """
        Fit per-sector Weibulls, build a PyWake XRSite, and return a
        WindResourceModel wrapping it.

        The XRSite is ready for wind-farm simulations via
        ``wind_resource.pywake_site``.
        """
        wd_centres, sector_freq, A_arr, k_arr = self._fit_sector_weibulls()

        ds = xr.Dataset(
            data_vars={
                "Sector_frequency": ("wd", sector_freq),
                "Weibull_A": ("wd", A_arr),
                "Weibull_k": ("wd", k_arr),
                "TI": 0.05,
            },
            coords={"wd": wd_centres},
        )
        site = XRSite(ds)

        global_shape, global_scale = self.create_weibull()
        return WindResourceModel(site, global_shape, global_scale)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_wind_rose(
        self, mode: str = "bar", n_directional_bins: int = 16, n_speed_bins: int = 6
    ):
        directions = np.array(self.wind_direction)
        speeds = np.array(self.wind_speed)
        N = len(directions)
        bin_width = 360.0 / n_directional_bins
        dir_bins = np.arange(0, 360, bin_width)

        if mode == "bar":
            speed_edges = np.linspace(0, speeds.max(), n_speed_bins + 1)
            speed_labels = [
                f"{speed_edges[i]:.1f}–{speed_edges[i + 1]:.1f} m/s" for i in range(n_speed_bins)
            ]
            colorscale = px.colors.sequential.Plasma
            fig = go.Figure()

            for i in range(n_speed_bins):
                mask_speed = (speeds >= speed_edges[i]) & (speeds < speed_edges[i + 1])
                if i == n_speed_bins - 1:
                    mask_speed = (speeds >= speed_edges[i]) & (speeds <= speed_edges[i + 1])

                freqs = [
                    np.sum(
                        mask_speed
                        & (((directions % 360) >= d) & ((directions % 360) < d + bin_width))
                    )
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
                        name=speed_labels[i],
                        marker_color=colorscale[color_idx],
                        marker_line_color="rgba(0,0,0,0.15)",
                        marker_line_width=0.5,
                        opacity=0.85,
                    )
                )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(ticksuffix="%", angle=90, tickfont_size=10),
                    angularaxis=dict(direction="clockwise", rotation=90),
                ),
                legend=dict(title="Wind Speed", x=1.05),
            )

        elif mode == "heatmap":
            n_r_bins = 30
            speed_edges = np.linspace(0, speeds.max(), n_r_bins + 1)
            freq_grid = np.zeros((n_directional_bins, n_r_bins))

            for di, d in enumerate(dir_bins):
                mask_dir = ((directions % 360) >= d) & ((directions % 360) < d + bin_width)
                for si in range(n_r_bins):
                    mask_s = (speeds >= speed_edges[si]) & (speeds < speed_edges[si + 1])
                    freq_grid[di, si] = np.sum(mask_dir & mask_s) / N * 100

            r_vals = [
                speed_edges[si + 1] for di in range(n_directional_bins) for si in range(n_r_bins)
            ]
            theta_vals = [
                dir_bins[di] for di in range(n_directional_bins) for si in range(n_r_bins)
            ]
            color_vals = [
                freq_grid[di, si] for di in range(n_directional_bins) for si in range(n_r_bins)
            ]

            fig = go.Figure(
                go.Barpolar(
                    r=r_vals,
                    theta=theta_vals,
                    width=bin_width,
                    marker=dict(
                        color=color_vals,
                        colorscale="Inferno",
                        showscale=True,
                        colorbar=dict(title="Frequency (%)", x=1.1),
                        line_width=1,
                    ),
                    base=0,
                    opacity=0.7,
                )
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(ticksuffix=" m/s", angle=90, tickfont_size=10),
                    angularaxis=dict(direction="clockwise", rotation=90),
                ),
            )
        else:
            raise ValueError(f"mode must be 'bar' or 'heatmap', got '{mode}'")

        fig.update_layout(width=600, height=500)
        return fig

    def plot_wind_speed(
        self,
        resample: str | None = None,
        show_rolling: bool = True,
        rolling_window: int = 24,
    ):
        dt = pd.to_datetime(self.datetime)
        ws = np.array(self.wind_speed)
        series = pd.Series(ws, index=dt, name="wind_speed")
        fig = go.Figure()

        if resample:
            resampled = series.resample(resample).mean()
            fig.add_trace(
                go.Scatter(
                    x=resampled.index,
                    y=resampled.values,
                    mode="lines",
                    line=dict(color="#00d4ff", width=2),
                    name=f"Mean ({resample})",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=dt,
                    y=ws,
                    mode="lines",
                    line=dict(color="#00d4ff", width=1),
                    opacity=0.6,
                    name="Wind Speed",
                )
            )
            if show_rolling:
                rolling = series.rolling(rolling_window, center=True).mean()
                fig.add_trace(
                    go.Scatter(
                        x=rolling.index,
                        y=rolling.values,
                        mode="lines",
                        line=dict(color="#ff6b35", width=2.5),
                        name=f"Rolling Mean ({rolling_window // 24} days)",
                    )
                )

        fig.update_layout(
            xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(
                title="Wind Speed (m/s)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
            width=1200,
            height=500,
        )
        return fig

    def plot_wind_distribution(
        self,
        weibull_params: tuple | None = None,
        bins: int = 40,
    ):
        ws = np.array(self.wind_speed)
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=ws,
                nbinsx=bins,
                histnorm="probability density",
                name="Observed",
                marker=dict(color="#00d4ff", opacity=0.55, line=dict(color="#0f1117", width=0.5)),
            )
        )

        if weibull_params is not None:
            k, lam = weibull_params
            x_range = np.linspace(0, ws.max() * 1.1, 500)
            pdf = (k / lam) * (x_range / lam) ** (k - 1) * np.exp(-((x_range / lam) ** k))
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=pdf,
                    mode="lines",
                    line=dict(color="#ff6b35", width=3, dash="dash"),
                    name=f"Weibull (k={k:.2f}, λ={lam:.2f})",
                )
            )

        fig.update_layout(
            xaxis=dict(
                title="Wind Speed (m/s)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            yaxis=dict(
                title="Probability Density", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            legend=dict(x=0.75, y=0.95),
            barmode="overlay",
            width=1200,
            height=500,
        )
        return fig

    def plot_wind_speed_heatmap(self, agg: str = "mean", colorscale: str = "inferno"):
        dt = pd.to_datetime(self.datetime)
        df = pd.DataFrame({"value": np.array(self.wind_speed)}, index=dt)
        return self._plot_wind_heatmap(df, "Wind Speed (m/s)", agg=agg, colorscale=colorscale)

    def plot_wind_direction_heatmap(self, agg: str = "mean", colorscale: str = "inferno"):
        dt = pd.to_datetime(self.datetime)
        df = pd.DataFrame({"value": np.array(self.wind_direction)}, index=dt)
        return self._plot_wind_heatmap(df, "Wind Direction (°)", agg=agg, colorscale=colorscale)

    def _plot_wind_heatmap(
        self, df: pd.DataFrame, name: str, agg: str = "mean", colorscale: str = "inferno"
    ):
        df = df.copy()
        df["month"] = df.index.month
        df["hour"] = df.index.hour

        agg_funcs = {"mean": "mean", "median": "median", "max": "max", "min": "min", "std": "std"}
        if agg not in agg_funcs:
            raise ValueError(f"agg must be one of {list(agg_funcs)}, got '{agg}'")

        pivot = (
            df.groupby(["month", "hour"])["value"]
            .agg(agg_funcs[agg])
            .unstack(level="hour")
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
        z = pivot.values

        hover = [
            [
                f"<b>{month_names[m]}  {hour_labels[h]}</b><br>{agg.capitalize()}: {z[m, h]:.2f}"
                for h in range(24)
            ]
            for m in range(12)
        ]

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=hour_labels,
                y=month_names,
                colorscale=colorscale,
                hoverinfo="text",
                text=hover,
                colorbar=dict(
                    title=dict(text=f"{agg.capitalize()} {name}", side="right"),
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
                side="bottom",
                showgrid=False,
            ),
            yaxis=dict(title="Month", autorange="reversed", showgrid=False),
            margin=dict(l=60, r=100, t=70, b=60),
            width=1000,
            height=500,
        )
        return fig
