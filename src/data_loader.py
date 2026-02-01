"""
Data Loader Module
==================
Handles loading and preprocessing of input data for the capacity expansion model.

Data Sources:
- CEA (Central Electricity Authority) for demand and existing capacity
- IRENA for technology costs
- MNRE for renewable energy capacity factors
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """
    Loads and preprocesses all input data for the capacity expansion model.
    
    Attributes:
        data_dir (Path): Directory containing input CSV files
        demand (pd.DataFrame): Demand forecast by year
        tech_costs (pd.DataFrame): Technology cost parameters
        existing_capacity (pd.DataFrame): Current installed capacity
        hourly_profiles (pd.DataFrame): Hourly capacity factors
        emissions (pd.DataFrame): Emission factors by technology
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing CSV data files
        """
        self.data_dir = Path(data_dir)
        self.demand = None
        self.tech_costs = None
        self.existing_capacity = None
        self.hourly_profiles = None
        self.emissions = None
        
    def load_all(self) -> dict:
        """
        Load all data files and return as dictionary.
        
        Returns:
            Dictionary containing all loaded DataFrames
        """
        self.demand = self._load_demand()
        self.tech_costs = self._load_technology_costs()
        self.existing_capacity = self._load_existing_capacity()
        self.hourly_profiles = self._load_hourly_profiles()
        self.emissions = self._load_emissions()
        
        return {
            'demand': self.demand,
            'tech_costs': self.tech_costs,
            'existing_capacity': self.existing_capacity,
            'hourly_profiles': self.hourly_profiles,
            'emissions': self.emissions
        }
    
    def _load_demand(self) -> pd.DataFrame:
        """Load demand forecast data."""
        df = pd.read_csv(self.data_dir / "demand_forecast.csv")
        df.set_index('year', inplace=True)
        return df
    
    def _load_technology_costs(self) -> pd.DataFrame:
        """Load technology cost parameters."""
        df = pd.read_csv(self.data_dir / "technology_costs.csv")
        df.set_index('technology', inplace=True)
        return df
    
    def _load_existing_capacity(self) -> pd.DataFrame:
        """Load existing installed capacity."""
        df = pd.read_csv(self.data_dir / "existing_capacity.csv")
        df.set_index('technology', inplace=True)
        return df
    
    def _load_hourly_profiles(self) -> pd.DataFrame:
        """Load hourly capacity factor profiles."""
        df = pd.read_csv(self.data_dir / "hourly_profiles.csv")
        df.set_index('hour', inplace=True)
        return df
    
    def _load_emissions(self) -> pd.DataFrame:
        """Load emission factors."""
        df = pd.read_csv(self.data_dir / "emissions_factors.csv")
        df.set_index('technology', inplace=True)
        return df
    
    def get_technologies(self) -> list:
        """Get list of all technologies."""
        return list(self.tech_costs.index)
    
    def get_renewable_technologies(self) -> list:
        """Get list of renewable technologies."""
        return list(self.tech_costs[self.tech_costs['is_renewable'] == 1].index)
    
    def get_storage_technologies(self) -> list:
        """Get list of storage technologies."""
        return list(self.tech_costs[self.tech_costs['is_storage'] == 1].index)
    
    def get_dispatchable_technologies(self) -> list:
        """Get list of dispatchable (non-variable) technologies."""
        non_variable = ['coal_existing', 'coal_new', 'gas_ccgt', 'nuclear', 'hydro_large']
        return [t for t in non_variable if t in self.get_technologies()]
    
    def get_years(self) -> list:
        """Get list of planning years."""
        return list(self.demand.index)
    
    def get_hours(self) -> list:
        """Get list of hours (0-23)."""
        return list(self.hourly_profiles.index)
    
    def calculate_annual_demand_by_hour(self, year: int) -> pd.Series:
        """
        Calculate hourly demand for a given year.
        
        The demand profile is scaled from the hourly profile to match
        the annual energy requirement.
        
        Args:
            year: Planning year
            
        Returns:
            Series with demand (MW) for each hour
        """
        peak_demand = self.demand.loc[year, 'peak_demand_gw'] * 1000  # Convert to MW
        demand_factors = self.hourly_profiles['demand_factor']
        
        # Scale to peak demand
        hourly_demand = demand_factors * peak_demand
        return hourly_demand
    
    def get_capacity_factor(self, technology: str, hour: int) -> float:
        """
        Get capacity factor for a technology at a specific hour.
        
        Args:
            technology: Technology name
            hour: Hour of day (0-23)
            
        Returns:
            Capacity factor (0-1)
        """
        if technology == 'solar':
            return self.hourly_profiles.loc[hour, 'solar_cf']
        elif technology == 'wind':
            return self.hourly_profiles.loc[hour, 'wind_cf']
        elif technology == 'hydro_large':
            return self.hourly_profiles.loc[hour, 'hydro_cf']
        else:
            # Dispatchable plants - use average PLF as max capacity factor
            if technology in self.existing_capacity.index:
                return self.existing_capacity.loc[technology, 'avg_plf_pct'] / 100
            else:
                return 0.85  # Default for thermal
    
    def get_variable_cost(self, technology: str) -> float:
        """
        Get total variable cost (O&M + fuel) for a technology in Rs/kWh.
        
        Args:
            technology: Technology name
            
        Returns:
            Variable cost in Rs/kWh
        """
        row = self.tech_costs.loc[technology]
        return row['variable_om_rs_per_kwh'] + row['fuel_cost_rs_per_kwh']
    
    def get_capital_cost_annualized(self, technology: str, 
                                     discount_rate: float = 0.10,
                                     lifetime: int = 25) -> float:
        """
        Get annualized capital cost using Capital Recovery Factor.
        
        CRF = r(1+r)^n / ((1+r)^n - 1)
        
        Args:
            technology: Technology name
            discount_rate: Annual discount rate (default 10%)
            lifetime: Plant lifetime in years (default 25)
            
        Returns:
            Annualized capital cost in Rs Crore per MW per year
        """
        capex = self.tech_costs.loc[technology, 'capital_cost_cr_per_mw']
        
        # Capital Recovery Factor
        r = discount_rate
        n = lifetime
        crf = (r * (1 + r)**n) / ((1 + r)**n - 1)
        
        return capex * crf
    
    def summary(self) -> str:
        """Print summary of loaded data."""
        summary_lines = [
            "=" * 60,
            "DATA SUMMARY",
            "=" * 60,
            f"\nPlanning Horizon: {self.get_years()[0]} - {self.get_years()[-1]}",
            f"Number of Technologies: {len(self.get_technologies())}",
            f"  - Renewable: {len(self.get_renewable_technologies())}",
            f"  - Storage: {len(self.get_storage_technologies())}",
            f"\nDemand Projections:",
        ]
        
        for year in self.get_years():
            peak = self.demand.loc[year, 'peak_demand_gw']
            energy = self.demand.loc[year, 'energy_requirement_twh']
            summary_lines.append(f"  {year}: Peak={peak} GW, Energy={energy} TWh")
        
        summary_lines.append(f"\nExisting Capacity (MW):")
        for tech in self.get_technologies():
            if tech in self.existing_capacity.index:
                cap = self.existing_capacity.loc[tech, 'installed_capacity_mw']
                if cap > 0:
                    summary_lines.append(f"  {tech}: {cap:,.0f} MW")
        
        return "\n".join(summary_lines)


# Convenience function for quick loading
def load_data(data_dir: str = "data") -> DataLoader:
    """
    Quick function to load all data.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Initialized DataLoader with all data loaded
    """
    loader = DataLoader(data_dir)
    loader.load_all()
    return loader
