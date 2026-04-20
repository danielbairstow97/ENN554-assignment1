from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from py_wake.wind_turbines import WindTurbine as _PyWakeWindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
import yaml


class WindTurbine(_PyWakeWindTurbine):
    """
    Project wind turbine — extends PyWake's WindTurbine with:
      - YAML + CSV loading via from_yaml()
      - Project-specific metadata (rated power, cut-in/out, etc.)
      - power_at() convenience method for numpy arrays
      - Plotly power curve plotWW
    """

    def __init__(
        self,
        name: str,
        rated_power_kw: int,
        rated_wind_speed: float,
        cut_in_wind_speed: float,
        cut_out_wind_speed: float,
        rotor_diameter: float,
        hub_height: float,
        power_curve: pd.DataFrame,
    ) -> None:
        # Store project metadata before calling super().__init__
        # so it's available if PyWake calls any overridden methods during init
        self.name_str = name
        self.rotor_area = np.pi * (rotor_diameter / 2) ** 2
        self.rated_power_kw = rated_power_kw
        self.rated_wind_speed = rated_wind_speed
        self.cut_in_wind_speed = cut_in_wind_speed
        self.cut_out_wind_speed = cut_out_wind_speed
        self.power_curve = power_curve

        ws = power_curve["wind_speed"].to_numpy()
        power = power_curve["power_kw"].to_numpy()
        ct = power_curve["Cp"].to_numpy()

        super().__init__(
            name=name,
            diameter=rotor_diameter,
            hub_height=hub_height,
            powerCtFunction=PowerCtTabular(ws, power, "kW", ct),
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: Path, pc_path: Path) -> "WindTurbine":
        yaml_path, pc_path = Path(yaml_path), Path(pc_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML not found: {yaml_path}")
        if not pc_path.exists():
            raise FileNotFoundError(f"Power curve CSV not found: {pc_path}")

        with yaml_path.open("r") as f:
            raw: dict = yaml.safe_load(f)

        def _clean(v):
            return None if (isinstance(v, str) and not v.strip()) else v

        raw = {k: _clean(v) for k, v in raw.items()}

        pc_df = pd.read_csv(pc_path).rename(
            columns={
                "Wind Speed [m/s]": "wind_speed",
                "Power [kW]": "power_kw",
                "Cp [-]": "Cp",
            }
        )

        return cls(
            name=raw["name"],
            rated_power_kw=int(raw["rated_power"]),
            rated_wind_speed=float(raw["rated_wind_speed"]),
            cut_in_wind_speed=float(raw["cut_in_wind_speed"]),
            cut_out_wind_speed=float(raw["cut_out_wind_speed"]),
            rotor_diameter=float(raw["rotor_diameter"]),
            hub_height=float(raw["hub_height"]),
            power_curve=pc_df,
        )

    # ------------------------------------------------------------------
    # Power calculation
    # ------------------------------------------------------------------
    def power_at(self, ws_hub: np.ndarray) -> np.ndarray:
        """
        Power output (kW) at hub-height wind speeds, with cut-in/out applied.
        Delegates to PyWake's interpolation — no duplicated logic.
        """
        ws_hub = np.asarray(ws_hub, dtype=float)
        power = self.power(ws_hub) / 1_000  # PyWake returns watts → kW
        power[(ws_hub < self.cut_in_wind_speed) | (ws_hub > self.cut_out_wind_speed)] = 0.0
        return power

    def power_at_50m(self, ws_50: np.ndarray) -> np.ndarray:
        ws_hub = np.asarray(ws_50, dtype=float) * (self.hub_height() / 50.0) ** (1 / 7.0)
        return self.power_at(ws_hub)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot_power_curve(self) -> go.Figure:
        ws_range = np.linspace(0, self.cut_out_wind_speed + 2, 300)
        power_kw = self.power(ws_range) / 1_000

        fig = go.Figure(
            go.Scatter(
                x=ws_range,
                y=power_kw,
                mode="lines",
                line=dict(color="#00d4ff", width=2.5),
                name="Power curve",
            )
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
        fig.update_layout(
            title=dict(text=f"Power Curve – {self.name()}", x=0.5),
            xaxis=dict(
                title="Wind Speed (m/s)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            yaxis=dict(
                title="Power Output (kW)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            width=600,
            height=420,
            hovermode="x unified",
        )
        return fig
