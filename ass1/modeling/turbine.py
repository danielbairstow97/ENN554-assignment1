from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

# from py_wake.wind_turbine import WindTurbine


@dataclass
class WindTurbine:
    name: str
    rated_power_kw: int
    rated_wind_speed: float
    cut_in_wind_speed: float
    cut_out_wind_speed: float
    rotor_diameter: float
    hub_height: float
    power_curve: pd.DataFrame

    @classmethod
    def from_yaml(cls, yaml_path: Path, pc_path: Path) -> "WindTurbine":
        """
        Load a Turbine from a YAML file and automatically resolve and load
        its power curve CSV.

        Parameters
        ----------
        yaml_path : str | Path
            Path to the turbine YAML file.
        turbine_dir : str | Path
            Root directory that contains turbine data files. The
            ``power_curve_file`` field in the YAML is resolved relative
            to this directory.

        Returns
        -------
        Turbine
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with yaml_path.open("r") as f:
            raw: dict = yaml.safe_load(f)

        # --- Normalise None-like values produced by bare YAML keys ---
        def _clean(v):
            """Return None for empty/whitespace strings."""
            if isinstance(v, str) and not v.strip():
                return None
            return v

        raw = {k: _clean(v) for k, v in raw.items()}

        # --- Load power curve CSV ---
        if not pc_path.exists():
            raise FileNotFoundError(f"Power curve CSV not found: {pc_path}\n")

        pc_df = pd.read_csv(pc_path).rename(
            columns={"Wind Speed [m/s]": "wind_speed", "Power [kW]": "power_kw", "Cp [-]": "Cp"}
        )

        return cls(
            # Required
            name=raw["name"],
            rated_power_kw=int(raw["rated_power"]),
            rated_wind_speed=float(raw["rated_wind_speed"]),
            cut_in_wind_speed=float(raw["cut_in_wind_speed"]),
            cut_out_wind_speed=float(raw["cut_out_wind_speed"]),
            rotor_diameter=float(raw["rotor_diameter"]),
            hub_height=float(raw["hub_height"]),
            power_curve=pc_df,
        )

    @property
    def rotor_area(self) -> float:
        """Swept rotor area in m²."""
        return 3.14159265358979 * (self.rotor_diameter / 2) ** 2

    def power_at(
        self, turbine_ws: np.ndarray | None = None, ws_50: np.ndarray | None = None
    ) -> np.ndarray:
        # Recalculate wind speed seen at hub
        if turbine_ws is None:
            if ws_50 is None:
                raise ValueError("Wind speed or turbine wind speed must be provided")
            turbine_ws = ws_50 * (self.hub_height / 50.0) ** (1 / 7.0)

        curve = self.power_curve.sort_values("wind_speed")
        power = np.interp(
            turbine_ws,
            curve["wind_speed"].to_numpy(),
            curve["power_kw"].to_numpy(),
            left=0.0,  # below cut-in → 0
            right=0.0,  # above cut-out → 0
        )

        # Zero out anything outside the operational envelope
        power[(turbine_ws < self.cut_in_wind_speed) | (turbine_ws > self.cut_out_wind_speed)] = 0.0

        return turbine_ws, power

    def plot_power_curve(self):
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=self.power_curve["wind_speed"],
                    y=self.power_curve["power_kw"],
                    mode="lines",
                )
            ],
            layout=go.Layout(
                width=500,
                height=400,
                xaxis=dict(title="Wind Speed (m/s)"),
                yaxis=dict(title="Turbine Power Output (kW)"),
                template="plotly_dark",
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                hovermode="x unified",
            ),
        )

        for speed, label, color in [
            (self.cut_in_wind_speed, "Cut-in", "#34d399"),
            (self.rated_wind_speed, "Rated", "#fbbf24"),
            (self.cut_out_wind_speed, "Cut-out", "#f87171"),
        ]:
            fig.add_vline(
                x=speed,
                line=dict(color=color, width=1.5, dash="dash"),
                annotation_text=f"<b>{label}</b><br>{speed} m/s",
                annotation_position="top",
                annotation_font=dict(color=color, size=11),
            )

        return fig
