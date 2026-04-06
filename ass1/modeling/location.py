import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from scipy import integrate
from scipy.integrate import quad
from scipy.stats import weibull_min

from ass1.modeling.turbine import WindTurbine


class WindResourceModel:
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def generate(self, num_samples):
        return self.scale * np.random.weibull(self.shape, num_samples)

    def pdf(self, ws: np.ndarray):
        return weibull_min.pdf(ws, self.shape, loc=0, scale=self.scale)

    def aep(self, turbine: WindTurbine, n_points: int = 1000) -> float:
        ws_rng = np.linspace(0, turbine.cut_out_wind_speed, n_points)
        power = turbine.power_at(ws_rng)
        pdf_val = self.pdf(ws_rng)

        # return 8760 * (power * pdf_val).sum()
        aep, err = 8760 * quad(
            lambda x: turbine.power_at(x) * self.pdf(x), 0, turbine.cut_out_wind_speed
        )

        if err is not None:
            raise err

        return aep

    def plot_power_distribution(self, turbine, n_points: int = 1000):
        ws_grid = np.linspace(0, turbine.cut_out_wind_speed * 1.05, n_points)

        pdf_val = self.pdf(ws_grid)
        turbine_ws, power = turbine.power_at(ws_50=ws_grid)
        power_dist = power * pdf_val  # integrand of AEP

        aep_val = self.aep(turbine)
        mean_power_kw = aep_val / 8760

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # --- Weibull PDF ---
        fig.add_trace(
            go.Scatter(
                x=ws_grid,
                y=pdf_val,
                mode="lines",
                name="Measured Wind Speed",
                line=dict(color="#00d4ff", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.10)",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=turbine_ws,
                y=pdf_val,
                mode="lines",
                name="Modelled Turbine Wind Speed",
                line=dict(color="#ee00ff", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.10)",
            ),
            secondary_y=False,
        )

        # --- Power curve overlay ---
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

        # --- Power-weighted distribution P(U)·f(U) ---
        fig.add_trace(
            go.Scatter(
                x=ws_grid,
                y=power_dist,
                mode="lines",
                name="P(U)·f(U)  (AEP integrand)",
                line=dict(color="#ff6b35", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(255, 107, 53, 0.12)",
            ),
            secondary_y=True,
        )

        # --- Rated / cut-in / cut-out markers ---
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
                title="Wind Speed (m/s)",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
            ),
            legend=dict(x=0.65, y=0.95),
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            hovermode="x unified",
        )
        fig.update_yaxes(
            title_text="Probability Density  f(U)",
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        )
        fig.update_yaxes(
            title_text="Power (kW)",
            secondary_y=True,
            showgrid=False,
        )

        return fig

    def plot_cumulative_distributions(
        self,
        turbines: list[WindTurbine],
        n_points: int = 500,
    ):
        # Use the highest cut-out across all turbines as the x-axis limit
        max_cutout = max(t.cut_out_wind_speed for t in turbines)
        ws_grid = np.linspace(0, max_cutout * 1.05, n_points)

        # --- Wind speed CDF (same for all turbines, resource-only) ---
        wind_cdf = weibull_min.cdf(ws_grid, self.shape, loc=0, scale=self.scale)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Wind Speed CDF",
                "Power CDF",
            ),
        )

        COLORS = ["#ff6b35", "#a78bfa", "#34d399", "#fbbf24", "#f87171"]
        wind_cdf_color = "#00d4ff"
        # ---- Left panel: Wind speed CDF --------------------------------
        fig.add_trace(
            go.Scatter(
                x=ws_grid,
                y=wind_cdf,
                mode="lines",
                name="Wind CDF",
                line=dict(color=wind_cdf_color, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.08)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Mark cut-in / cut-out for each turbine on the wind CDF
        for i, turbine in enumerate(turbines):
            color = COLORS[i % len(COLORS)]
            label = turbine.name

            for speed, marker in [
                (turbine.cut_in_wind_speed, "▷"),
                (turbine.cut_out_wind_speed, "◁"),
            ]:
                cdf_at = float(weibull_min.cdf(speed, self.shape, loc=0, scale=self.scale))
                fig.add_trace(
                    go.Scatter(
                        x=[speed],
                        y=[cdf_at],
                        mode="markers+text",
                        marker=dict(color=color, size=9, symbol="circle"),
                        text=[f"{marker} {label}"],
                        textposition="top right",
                        textfont=dict(size=10, color=color),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

        # ---- Right panel: Power output CDF -----------------------------
        for i, turbine in enumerate(turbines):
            color = COLORS[i % len(COLORS)]
            label = turbine.name

            # --- Point mass probabilities ---
            p_zero = float(
                # below cut-in
                weibull_min.cdf(turbine.cut_in_wind_speed, self.shape, loc=0, scale=self.scale)
                # above cut-out
                + (
                    1
                    - weibull_min.cdf(
                        turbine.cut_out_wind_speed, self.shape, loc=0, scale=self.scale
                    )
                )
            )
            p_rated = float(
                1
                - weibull_min.cdf(turbine.rated_wind_speed, self.shape, loc=0, scale=self.scale)
                - (
                    1
                    - weibull_min.cdf(
                        turbine.cut_out_wind_speed, self.shape, loc=0, scale=self.scale
                    )
                )
            )

            turbine_ws, power_operating = turbine.power_at(ws_grid)  # monotonically increasing

            # CDF of power in the continuous region:
            # F(p) = P(P(U) ≤ p) = P(U ≤ P⁻¹(p)) = F_wind(P⁻¹(p))
            wind_cdf_operating = weibull_min.cdf(ws_grid, self.shape, loc=0, scale=self.scale)
            # Shift up by p_zero (the point mass at 0)
            power_cdf_continuous = p_zero + (wind_cdf_operating - wind_cdf_operating[0])

            # --- Assemble full CDF with point masses ---
            # Point mass at 0
            power_pts = [0, 0]
            cdf_pts = [0, p_zero]
            # Continuous rise from cut-in to rated
            power_pts += power_operating.tolist()
            cdf_pts += power_cdf_continuous.tolist()
            # Point mass at rated (jump to 1.0)
            power_pts += [turbine.rated_power_kw, turbine.rated_power_kw]
            cdf_pts += [1.0 - p_rated, 1.0]

            capacity_factor = (
                1
                - p_zero
                - p_rated
                + integrate.simpson(power_cdf_continuous, x=power_operating)
                / turbine.rated_power_kw
            )  # E[P] / P_rated via integration by parts

            fig.add_trace(
                go.Scatter(
                    x=turbine_ws,
                    y=wind_cdf,
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=power_pts,
                    y=cdf_pts,
                    mode="lines",
                    name=f"{label}  (CF={capacity_factor:.1%})",
                    line=dict(color=color, width=2.5),
                ),
                row=1,
                col=2,
            )

            # Mark point masses as markers
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

        # ---- Shared annotations ----------------------------------------
        # 50th percentile wind speed
        p50_ws = weibull_min.ppf(0.5, self.shape, loc=0, scale=self.scale)
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
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
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


class WindResourceData:
    def __init__(
        self, datetime: pl.Series, wind_speed: pl.Series, wind_direction: pl.Series
    ) -> None:
        self.datetime = datetime
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    def create_weibull(self) -> tuple[float, float]:
        ws = np.array(self.wind_speed)
        ws = ws[ws >= 0]
        shape, loc, scale = weibull_min.fit(ws, floc=0)

        return shape, scale

    def create_weibull_wind_resource(self) -> WindResourceModel:
        shape, scale = self.create_weibull()
        return WindResourceModel(shape, scale)

    def plot_wind_rose(
        self, mode: str = "bar", n_directional_bins: int = 16, n_speed_bins: int = 6
    ):
        directions = np.array(self.wind_direction)
        speeds = np.array(self.wind_speed)
        N = len(directions)

        bin_width = 360.0 / n_directional_bins
        dir_bins = np.arange(0, 360, bin_width)
        dir_labels = [f"{int(d)}°" for d in dir_bins]

        if mode == "bar":
            speed_edges = np.linspace(0, speeds.max(), n_speed_bins + 1)
            speed_labels = [
                f"{speed_edges[i]:.1f}–{speed_edges[i + 1]:.1f} m/s" for i in range(n_speed_bins)
            ]
            colorscale = px.colors.sequential.Plasma
            fig = go.Figure()
            for i in range(n_speed_bins):
                # Filter individual speed bins
                mask_speed = (speeds >= speed_edges[i]) & (speeds < speed_edges[i + 1])
                if i == n_speed_bins - 1:  # include upper edge on last bin
                    mask_speed = (speeds >= speed_edges[i]) & (speeds <= speed_edges[i + 1])

                # bin by direction
                freqs = []
                for d in dir_bins:
                    mask_dir = ((directions % 360) >= d) & ((directions % 360) < d + bin_width)
                    freqs.append(np.sum(mask_speed & mask_dir) / N * 100)

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
                    title=dict(text="Wind Rose – Frequency by Direction & Speed", x=0.5),
                    polar=dict(
                        radialaxis=dict(ticksuffix="%", angle=90, tickfont_size=10),
                        angularaxis=dict(direction="clockwise", rotation=90),
                    ),
                    legend=dict(title="Wind Speed", x=1.05),
                    template="plotly_dark",
                    paper_bgcolor="#0f1117",
                    plot_bgcolor="#0f1117",
                )
        elif mode == "heatmap":
            # Build a 2-D frequency matrix [direction × speed] then show as
            # a filled polar plot using go.Barpolar with colour = frequency.
            n_r_bins = 30
            speed_edges = np.linspace(0, speeds.max(), n_r_bins + 1)

            # frequency grid  (dir_bins × speed_bins)
            freq_grid = np.zeros((n_directional_bins, n_r_bins))
            for di, d in enumerate(dir_bins):
                mask_dir = ((directions % 360) >= d) & ((directions % 360) < d + bin_width)
                for si in range(n_r_bins):
                    mask_s = (speeds >= speed_edges[si]) & (speeds < speed_edges[si + 1])
                    freq_grid[di, si] = np.sum(mask_dir & mask_s) / N * 100

            # Flatten into traces – one go.Barpolar bar per (direction, speed) cell
            r_vals, theta_vals, color_vals = [], [], []
            for di in range(n_directional_bins):
                for si in range(n_r_bins):
                    r_vals.append(speed_edges[si + 1])  # radial position = speed
                    theta_vals.append(dir_bins[di])
                    color_vals.append(freq_grid[di, si])

            fig = go.Figure(
                go.Barpolar(
                    r=r_vals,
                    theta=theta_vals,
                    width=bin_width,
                    marker_color=color_vals,
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
                title=dict(text="Wind Rose – Frequency Heatmap", x=0.5),
                polar=dict(
                    radialaxis=dict(
                        ticksuffix=" m/s",
                        angle=90,
                        tickfont_size=10,
                    ),
                    angularaxis=dict(direction="clockwise", rotation=90),
                ),
                width=600,
                height=500,
                template="plotly_dark",
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
            )
        else:
            raise ValueError(f"mode must be 'bar' or 'heatmap', got '{mode}'")

        fig.update_layout(width=600, height=500)
        return fig

    def plot_wind_speed(
        self, resample: str | None = None, show_rolling: bool = True, rolling_window: int = 24
    ):
        """
        Plot wind speed over the period of data.

        Parameters
        ----------
        resample : str or None
            Pandas offset alias to resample data (e.g. 'D', 'W', 'ME').
            None keeps the raw data.
        show_rolling : bool
            Overlay a rolling-mean line (only used when resample is None).
        rolling_window : int
            Window size (number of samples) for the rolling mean.
        """
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
                    line=dict(color="#00d4ff", width=1, dash="solid"),
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
                        name=f"Rolling Mean ({rolling_window})",
                    )
                )

        fig.update_layout(
            title=dict(text="Wind Speed Over Time", x=0.5),
            xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(
                title="Wind Speed (m/s)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            legend=dict(x=0.01, y=0.99),
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            hovermode="x unified",
        )

        return fig

    def plot_wind_distribution(
        self,
        weibull_params: tuple | None = None,
        bins: int = 40,
        show_kde: bool = True,
    ):
        """
        Plot a histogram of wind speed distribution, optionally overlaying a
        Weibull PDF.

        Parameters
        ----------
        weibull_params : tuple or None
            (k, lam) shape and scale parameters for the Weibull distribution.
            If None, no Weibull curve is drawn.
            Pass the result of ``create_weibull()`` directly here.
        bins : int
            Number of histogram bins.
        """
        ws = np.array(self.wind_speed)

        fig = go.Figure()

        # Normalised histogram
        fig.add_trace(
            go.Histogram(
                x=ws,
                nbinsx=bins,
                histnorm="probability density",
                name="Observed",
                marker=dict(color="#00d4ff", opacity=0.55, line=dict(color="#0f1117", width=0.5)),
            )
        )

        x_range = np.linspace(0, ws.max() * 1.1, 500)

        # Weibull PDF overlay
        if weibull_params is not None:
            k, lam = weibull_params
            # Two-parameter Weibull PDF: f(x) = (k/λ)(x/λ)^(k-1) exp(-(x/λ)^k)
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
            title=dict(text="Wind Speed Distribution", x=0.5),
            xaxis=dict(
                title="Wind Speed (m/s)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            yaxis=dict(
                title="Probability Density", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            legend=dict(x=0.75, y=0.95),
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
        )

        return fig

    def plot_wind_speed_heatmap(
        self,
        agg: str = "mean",
        colorscale: str = "Viridis",
    ):
        dt = pd.to_datetime(self.datetime)

        df = pd.DataFrame({"value": np.array(self.wind_speed)}, index=dt)
        return self.plot_wind_heatmap(df, "Wind Speed (m/s)", agg=agg, colorscale=colorscale)

    def plot_wind_direction_heatmap(
        self,
        agg: str = "mean",
        colorscale: str = "Viridis",
    ):
        dt = pd.to_datetime(self.datetime)

        df = pd.DataFrame({"value": np.array(self.wind_direction)}, index=dt)
        return self.plot_wind_heatmap(df, "Wind Direction", agg=agg, colorscale=colorscale)

    def plot_wind_heatmap(
        self,
        df,
        name: str,
        agg: str = "mean",
        colorscale: str = "Viridis",
    ):
        """
        Plot a heatmap of wind speed across the year.

        Layout
        ------
        - Y-axis (rows)    : Calendar month (Jan → Dec), one row per month.
        - X-axis (columns) : Hour of day (00:00 → 23:00), 24 columns.
        - Cell colour      : Aggregate wind speed or wind direction for that (month, hour) bucket.

        """
        df["month"] = df.index.month  # 1–12
        df["hour"] = df.index.hour  # 0–23

        agg_funcs = {"mean": "mean", "median": "median", "max": "max", "min": "min", "std": "std"}
        if agg not in agg_funcs:
            raise ValueError(f"agg must be one of {list(agg_funcs)}, got '{agg}'")

        pivot = (
            df.groupby(["month", "hour"])["value"]
            .agg(agg_funcs[agg])
            .unstack(level="hour")  # columns = hours 0–23
            .reindex(index=range(1, 13))  # ensure all 12 months present
            .reindex(columns=range(24))  # ensure all 24 hours present
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

        z = pivot.values  # shape (12, 24)

        # Custom hover text: "Jan  06:00 — mean: 7.3 m/s"
        hover = [
            [
                f"<b>{month_names[m]}  {hour_labels[h]}</b><br>{agg.capitalize()}: {z[m, h]:.2f} m/s"
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
            title=dict(
                text=f"{name} Heatmap – {agg.capitalize()} by Month & Hour of Day",
                x=0.5,
                font=dict(size=16),
            ),
            xaxis=dict(
                title="Hour of Day",
                tickmode="array",
                tickvals=hour_labels[::3],  # label every 3 hours
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                side="bottom",
                showgrid=False,
            ),
            yaxis=dict(
                title="Month",
                autorange="reversed",  # Jan at top
                showgrid=False,
            ),
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            margin=dict(l=60, r=100, t=70, b=60),
            height=520,
        )

        return fig
