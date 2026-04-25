from dataclasses import dataclass

from ass1.modeling.farm import WindFarm
from ass1.modeling.turbine import WindTurbine


@dataclass
class BaseFinancialModel:
    """
    Base financial model providing FCR and annualisation from
    first-principle financial parameters.

    Parameters
    ----------
    nominal_discount_rate : weighted average cost of capital (WACC) or
                            required rate of return, nominal
    inflation_rate        : assumed annual inflation
    project_lifetime      : years
    """

    nominal_discount_rate: float = 0.08  # WACC — typical for wind projects
    inflation_rate: float = 0.025  # ~RBA long-run target
    project_lifetime: int = 25  # years

    @property
    def real_discount_rate(self) -> float:
        """Fisher equation: real rate from nominal rate and inflation."""
        return (1 + self.nominal_discount_rate) / (1 + self.inflation_rate) - 1

    @property
    def fcr(self) -> float:
        """
        Fixed Charge Rate — annualises CAPEX over project lifetime.

        Uses the Capital Recovery Factor (CRF) against the real discount
        rate, giving a pre-tax FCR suitable for academic LCOE calculations.

            FCR = CRF = r(1+r)^n / ((1+r)^n - 1)
        """
        r = self.real_discount_rate
        n = self.project_lifetime
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    @property
    def npv_factor(self) -> float:
        """Sum of discount factors over project lifetime — useful for NPV calcs."""
        r = self.real_discount_rate
        return sum(1 / (1 + r) ** t for t in range(1, self.project_lifetime + 1))


@dataclass
class TurbineFinancialModel(BaseFinancialModel):
    """
    Single-turbine CAPEX, OPEX and LCOE.

    Cost model based on rotor area and rated power — consistent with
    NREL land-based and offshore cost models.
    """

    wt_rotor_cost: float = 170.0  # AUD/m²  rotor system
    wt_tower_cost: float = 90.0  # AUD/m²  tower (scaled by rotor area as proxy)
    wt_other_cost: float = 1000.0  # AUD/kW  drivetrain, electrical, installation
    wt_foc: float = 60.0  # AUD/kW/yr  fixed operating cost

    def cost_turbine_capex(self, turbine: WindTurbine) -> tuple[float, dict[str, float]]:
        """Returns (total_capex_AUD, component_breakdown)."""
        costs = {
            "Rotor": self.wt_rotor_cost * turbine.rotor_area,
            "Tower": self.wt_tower_cost * turbine.rotor_area,
            "Other": self.wt_other_cost * turbine.rated_power_kw,
        }
        return sum(costs.values()), costs

    def cost_turbine_opex(self, turbine: WindTurbine) -> float:
        """Annual fixed OPEX (AUD/yr)."""
        return self.wt_foc * turbine.rated_power_kw

    def lcoe(self, turbine: WindTurbine, aep_kwh: float) -> float:
        """
        Levelised Cost of Energy (AUD/kWh).

            LCOE = (FCR × CAPEX + OPEX) / AEP

        Parameters
        ----------
        aep_kwh : Annual Energy Production in kWh/year
        """
        capex, _ = self.cost_turbine_capex(turbine)
        opex = self.cost_turbine_opex(turbine)
        return (self.fcr * capex + opex) / aep_kwh


@dataclass
class WindFarmFinancialModel(TurbineFinancialModel):
    """Extends turbine model to a full wind farm."""

    wf_foc: float = 60.0  # AUD/kW/yr  additional farm-level O&M

    def cost_farm_capex(self, farm: WindFarm) -> tuple[float, dict[str, float]]:
        """Scale single-turbine CAPEX by number of turbines."""
        turbine, n_turbines = farm.turbine, farm.configuration
        capex, components = self.cost_turbine_capex(turbine)
        farm_components = {k: v * n_turbines for k, v in components.items()}
        return sum(farm_components.values()), farm_components

    def cost_farm_opex(self, farm: WindFarm) -> tuple[float, dict[str, float]]:
        """Annual farm OPEX broken into turbine O&M and farm-level O&M."""
        turbine, n_turbines = farm.turbine, farm.configuration
        opex = {
            "Turbine O&M": self.cost_turbine_opex(turbine) * n_turbines,
            "Farm O&M": self.wf_foc * turbine.rated_power_kw * n_turbines,
        }
        return sum(opex.values()), opex

    def farm_lcoe(self, farm: WindFarm, aep_kwh: float) -> float:
        """Farm-level LCOE (AUD/kWh)."""
        capex, _ = self.cost_farm_capex(farm)
        opex, _ = self.cost_farm_opex(farm)
        return (self.fcr * capex + opex) / aep_kwh


@dataclass
class MicrogridFinancialModel(WindFarmFinancialModel):
    """
    Extends the farm model with battery storage (BESS) costs and
    the Annualised Cost of Energy (ACOE) metric for the full microgrid.
    """

    bess_capacity_cost: float = 300.0  # AUD/kWh  energy capacity cost
    bess_power_cost: float = 375.0  # AUD/kW   power conversion cost
    bess_foc: float = 10.0  # AUD/kW/yr fixed BESS O&M

    def cost_bess_capex(self, bess_power_kw: float, bess_capacity_kwh: float) -> float:
        """
        BESS CAPEX with separate power and energy components.
        Both must be costed — taking the max of one is incorrect.
        """
        return max(
            bess_power_kw * self.bess_power_cost,
            bess_capacity_kwh * self.bess_capacity_cost,
        )

    def cost_microgrid_capex(
        self,
        farm: WindFarm,
        bess_power_kw: float,
        bess_capacity_kwh: float,
    ) -> tuple[float, dict[str, float]]:

        capex = {
            "BESS CAPEX (AUD)": self.cost_bess_capex(bess_power_kw, bess_capacity_kwh),
            "Wind Farm CAPEX (AUD)": self.cost_farm_capex(farm)[0],
        }

        return sum(capex.values()), capex

    def cost_microgrid_opex(
        self,
        farm: WindFarm,
        bess_power_kw: float,
    ) -> tuple[float, dict[str, float]]:
        opex = {
            "Farm O&M": self.cost_farm_opex(farm)[0],
            "BESS O&M": self.bess_foc * bess_power_kw,
        }
        return sum(opex.values()), opex

    def acoe(
        self,
        farm: WindFarm,
        bess_power_kw: float,
        bess_capacity_kwh: float,
        load_kwh: float,
    ) -> float:
        """
        Annualised Cost of Energy (AUD/kWh) for the full microgrid.

            ACOE = (FCR × CAPEX + OPEX) / Annual Load

        Parameters
        ----------
        load_kwh : total annual load served (kWh/year)
        """
        capex, _ = self.cost_microgrid_capex(farm, bess_power_kw, bess_capacity_kwh)
        opex, _ = self.cost_microgrid_opex(farm, bess_power_kw)
        return (self.fcr * capex + opex) / load_kwh
