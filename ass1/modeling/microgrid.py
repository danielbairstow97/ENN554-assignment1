import pandas as pd
import pypsa

from ass1.modeling.farm import WindFarm


class Microgrid:
    def __init__(self, farm: WindFarm):
        self.farm = farm
        n = pypsa.Network()

        # Add network components
        n.add("Bus", "Seaspray", carrier="electricity")
        n.add("Load", "demand", bus="Seaspray")

        # This is a slack generator that ensures load can be balanced.
        # This generator should contribute nothing to the network. You can
        # remove once you have the wind farm and battery setup
        n.add(
            "Generator", "dummy", bus="Seaspray", control="Slack", marginal_cost=10.0, p_nom=10_000
        )

        # Need the wind farm and some way to connect it to Seaspray

        # This just ensures anything missing gets added
        n.sanitize()
        self.n = n

    def prepare_network(self, site: pd.DataFrame):
        """
        Set time-series inputs on the network.

        Parameters
        ----------
        site : DataFrame with columns Time, Load [MW], WS50M, WD50M
        """

        site = _coerce_arrow_strings(site)
        ts = site["Time"]
        ws = site["WS50M"].to_numpy()
        wd = site["WD50M"].to_numpy()
        load = site["Load [MW]"].to_numpy()

        # This prepares the timeseries of the network
        self.n.set_snapshots(ts)

        # Sets the demand that must be met
        self.n.loads_t.p_set["demand"] = load

        # Wind Farm: Set available power of the network
        windfarm_output = self.farm.power_at_50m(ws, wd)

    def solve_network(self) -> float:
        """Solve network, returning battery size"""
        # If prepare_network correctly implemented this will succeed
        self.n.optimize(include_objective_constant=True)

        # Extract battery optimised battery size
        return 0.0


def _coerce_arrow_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any Arrow-backed string columns to numpy object dtype.
    PyPSA passes column names and component names through xarray which
    cannot handle pandas ArrowStringArray.
    """
    return df.apply(lambda col: col.astype(str) if hasattr(col.dtype, "pyarrow_dtype") else col)
