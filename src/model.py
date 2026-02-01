"""
Capacity Expansion Model
========================
Core optimization model using PuLP for linear programming.

This module implements a least-cost capacity expansion model for India's
power sector, determining the optimal mix of generation and storage
technologies to meet demand while satisfying policy constraints.

Mathematical Formulation:
------------------------
Minimize: Total System Cost = Capital + Fixed O&M + Variable Costs
Subject to:
    - Demand balance (generation >= demand for each hour)
    - Capacity limits (generation <= capacity × capacity_factor)
    - RE target (renewable generation >= 50% by 2030)
    - Reserve margin (firm capacity >= peak demand × 1.15)
    - Storage constraints (energy balance, power limits)
"""

import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .data_loader import DataLoader


class CapacityExpansionModel:
    """
    Linear programming model for capacity expansion planning.
    
    This model determines the least-cost mix of generation capacity
    to meet future electricity demand while satisfying renewable
    energy targets and reliability constraints.
    
    Attributes:
        data (DataLoader): Loaded input data
        config (dict): Model configuration parameters
        model (pulp.LpProblem): PuLP optimization model
        variables (dict): Decision variables
        results (dict): Optimization results
    """
    
    def __init__(self, data: DataLoader, config: Optional[dict] = None):
        """
        Initialize the capacity expansion model.
        
        Args:
            data: DataLoader object with input data
            config: Optional configuration dictionary with:
                - discount_rate: Annual discount rate (default 0.10)
                - reserve_margin: Reserve margin above peak (default 0.15)
                - re_target: Renewable energy target (default 0.50)
                - emission_limit: Optional CO2 limit (Mt/year)
        """
        self.data = data
        
        # Default configuration
        self.config = {
            'discount_rate': 0.10,
            'reserve_margin': 0.15,
            're_target': 0.50,  # 50% RE by 2030
            'emission_limit': None,  # No limit by default
            'hours_per_year': 8760,
        }
        if config:
            self.config.update(config)
        
        # Model components
        self.model = None
        self.variables = {}
        self.results = None
        
        # Sets
        self.technologies = data.get_technologies()
        self.years = data.get_years()
        self.hours = data.get_hours()
        self.re_techs = data.get_renewable_technologies()
        self.storage_techs = data.get_storage_technologies()
        
    def build(self):
        """Build the complete optimization model."""
        print("Building capacity expansion model...")
        
        # Create problem
        self.model = pulp.LpProblem("India_Capacity_Expansion", pulp.LpMinimize)
        
        # Create variables
        self._create_variables()
        
        # Set objective
        self._set_objective()
        
        # Add constraints
        self._add_demand_constraints()
        self._add_capacity_constraints()
        self._add_re_target_constraint()
        self._add_reserve_margin_constraint()
        self._add_storage_constraints()
        
        if self.config.get('emission_limit'):
            self._add_emission_constraint()
        
        print(f"Model built with {len(self.model.variables())} variables")
        print(f"and {len(self.model.constraints)} constraints")
        
    def _create_variables(self):
        """Create decision variables."""
        
        # New capacity additions by technology and year (MW)
        self.variables['new_capacity'] = pulp.LpVariable.dicts(
            "NewCap",
            [(t, y) for t in self.technologies for y in self.years],
            lowBound=0,
            cat='Continuous'
        )
        
        # Generation by technology, year, and hour (MW)
        self.variables['generation'] = pulp.LpVariable.dicts(
            "Gen",
            [(t, y, h) for t in self.technologies 
             for y in self.years for h in self.hours],
            lowBound=0,
            cat='Continuous'
        )
        
        # Storage charging by storage tech, year, hour (MW)
        self.variables['charging'] = pulp.LpVariable.dicts(
            "Charge",
            [(s, y, h) for s in self.storage_techs 
             for y in self.years for h in self.hours],
            lowBound=0,
            cat='Continuous'
        )
        
        # Storage discharging by storage tech, year, hour (MW)
        self.variables['discharging'] = pulp.LpVariable.dicts(
            "Discharge",
            [(s, y, h) for s in self.storage_techs 
             for y in self.years for h in self.hours],
            lowBound=0,
            cat='Continuous'
        )
        
        # Storage state of charge (MWh)
        self.variables['soc'] = pulp.LpVariable.dicts(
            "SOC",
            [(s, y, h) for s in self.storage_techs 
             for y in self.years for h in self.hours],
            lowBound=0,
            cat='Continuous'
        )
        
    def _get_total_capacity(self, tech: str, year: int) -> pulp.LpAffineExpression:
        """
        Get total capacity (existing + new additions up to year).
        
        Args:
            tech: Technology name
            year: Year
            
        Returns:
            Expression for total capacity in MW
        """
        # Existing capacity
        if tech in self.data.existing_capacity.index:
            existing = self.data.existing_capacity.loc[tech, 'installed_capacity_mw']
            # Account for retirements
            retirement = self.data.existing_capacity.loc[tech, 'retirement_by_2030_mw']
            years_to_2030 = 2030 - self.years[0]
            annual_retirement = retirement / years_to_2030 if years_to_2030 > 0 else 0
            years_passed = year - self.years[0]
            existing = max(0, existing - annual_retirement * years_passed)
        else:
            existing = 0
        
        # Under construction capacity (comes online by 2027)
        if tech in self.data.existing_capacity.index:
            under_construction = self.data.existing_capacity.loc[tech, 'under_construction_mw']
            if year >= 2027:
                existing += under_construction
        
        # New capacity additions
        new_additions = pulp.lpSum([
            self.variables['new_capacity'][(tech, y)]
            for y in self.years if y <= year
        ])
        
        return existing + new_additions
    
    def _set_objective(self):
        """Set the objective function (minimize total cost)."""
        
        total_cost = []
        
        for y in self.years:
            year_idx = y - self.years[0]
            discount_factor = 1 / (1 + self.config['discount_rate']) ** year_idx
            
            for t in self.technologies:
                # Annualized capital cost for new capacity
                ann_capex = self.data.get_capital_cost_annualized(
                    t, self.config['discount_rate']
                )
                fixed_om = self.data.tech_costs.loc[t, 'fixed_om_lakh_per_mw_yr'] / 100  # Convert to Cr
                
                # Capital cost (only for new capacity in that year)
                total_cost.append(
                    discount_factor * ann_capex * 
                    self.variables['new_capacity'][(t, y)]
                )
                
                # Fixed O&M (for total capacity)
                total_cost.append(
                    discount_factor * fixed_om * 
                    self._get_total_capacity(t, y)
                )
                
                # Variable costs (for generation)
                var_cost = self.data.get_variable_cost(t)  # Rs/kWh
                # Convert: Rs/kWh × MW × hours = Rs × 1000 = Rs Lakh
                # Divide by 10^7 to get Rs Crore
                for h in self.hours:
                    # Weight each hour - assume each represents 365 days
                    hours_weight = 365
                    total_cost.append(
                        discount_factor * var_cost * 
                        self.variables['generation'][(t, y, h)] *
                        hours_weight / 10000  # Convert to Crore
                    )
        
        self.model += pulp.lpSum(total_cost), "Total_System_Cost"
        
    def _add_demand_constraints(self):
        """Add demand balance constraints."""
        
        for y in self.years:
            hourly_demand = self.data.calculate_annual_demand_by_hour(y)
            
            for h in self.hours:
                demand = hourly_demand.iloc[h]
                
                # Total generation from all technologies
                generation = pulp.lpSum([
                    self.variables['generation'][(t, y, h)]
                    for t in self.technologies
                ])
                
                # Net storage (discharge - charge)
                storage_net = pulp.lpSum([
                    self.variables['discharging'][(s, y, h)] -
                    self.variables['charging'][(s, y, h)]
                    for s in self.storage_techs
                ])
                
                # Demand must be met
                self.model += (
                    generation + storage_net >= demand,
                    f"Demand_Balance_{y}_{h}"
                )
    
    def _add_capacity_constraints(self):
        """Add generation capacity constraints."""
        
        for y in self.years:
            for t in self.technologies:
                if t in self.storage_techs:
                    continue  # Storage handled separately
                    
                for h in self.hours:
                    # Get capacity factor for this hour
                    cf = self.data.get_capacity_factor(t, h)
                    
                    # Generation limited by capacity × capacity factor
                    self.model += (
                        self.variables['generation'][(t, y, h)] <= 
                        self._get_total_capacity(t, y) * cf,
                        f"Cap_Limit_{t}_{y}_{h}"
                    )
                    
    def _add_re_target_constraint(self):
        """Add renewable energy target constraint for 2030."""
        
        y = 2030  # Target year
        
        # Total RE generation
        re_generation = pulp.lpSum([
            self.variables['generation'][(t, y, h)] * 365  # Annual
            for t in self.re_techs
            for h in self.hours
        ])
        
        # Total demand
        total_demand = sum(
            self.data.calculate_annual_demand_by_hour(y).iloc[h] * 365
            for h in self.hours
        )
        
        # RE must be at least target percentage
        self.model += (
            re_generation >= self.config['re_target'] * total_demand,
            "RE_Target_2030"
        )
        
    def _add_reserve_margin_constraint(self):
        """Add reserve margin constraint for reliability."""
        
        for y in self.years:
            peak_demand = self.data.demand.loc[y, 'peak_demand_gw'] * 1000  # MW
            required_firm = peak_demand * (1 + self.config['reserve_margin'])
            
            # Firm capacity from each technology
            firm_capacity = []
            for t in self.technologies:
                capacity_credit = self.data.tech_costs.loc[t, 'capacity_credit_pct'] / 100
                firm_capacity.append(
                    self._get_total_capacity(t, y) * capacity_credit
                )
            
            self.model += (
                pulp.lpSum(firm_capacity) >= required_firm,
                f"Reserve_Margin_{y}"
            )
            
    def _add_storage_constraints(self):
        """Add storage operation constraints."""
        
        for s in self.storage_techs:
            duration = self.data.tech_costs.loc[s, 'storage_duration_hrs']
            efficiency = self.data.tech_costs.loc[s, 'round_trip_efficiency']
            
            for y in self.years:
                power_capacity = self._get_total_capacity(s, y)
                energy_capacity = power_capacity * duration
                
                for h in self.hours:
                    # Charging power limit
                    self.model += (
                        self.variables['charging'][(s, y, h)] <= power_capacity,
                        f"Charge_Limit_{s}_{y}_{h}"
                    )
                    
                    # Discharging power limit
                    self.model += (
                        self.variables['discharging'][(s, y, h)] <= power_capacity,
                        f"Discharge_Limit_{s}_{y}_{h}"
                    )
                    
                    # State of charge limit
                    self.model += (
                        self.variables['soc'][(s, y, h)] <= energy_capacity,
                        f"SOC_Limit_{s}_{y}_{h}"
                    )
                    
                    # Energy balance
                    h_prev = 23 if h == 0 else h - 1
                    self.model += (
                        self.variables['soc'][(s, y, h)] ==
                        self.variables['soc'][(s, y, h_prev)] +
                        self.variables['charging'][(s, y, h)] * np.sqrt(efficiency) -
                        self.variables['discharging'][(s, y, h)] / np.sqrt(efficiency),
                        f"SOC_Balance_{s}_{y}_{h}"
                    )
                    
    def _add_emission_constraint(self):
        """Add CO2 emission limit constraint."""
        
        emission_limit = self.config['emission_limit']  # Mt CO2/year
        
        for y in self.years:
            total_emissions = pulp.lpSum([
                self.variables['generation'][(t, y, h)] * 365 *
                self.data.emissions.loc[t, 'emission_factor_kg_co2_per_kwh'] / 1e9
                for t in self.technologies
                for h in self.hours
                if t in self.data.emissions.index
            ])
            
            self.model += (
                total_emissions <= emission_limit,
                f"Emission_Limit_{y}"
            )
    
    def solve(self, solver: str = 'CBC', time_limit: int = 300) -> bool:
        """
        Solve the optimization model.
        
        Args:
            solver: Solver to use ('CBC', 'GLPK', 'CPLEX', 'GUROBI')
            time_limit: Maximum solve time in seconds
            
        Returns:
            True if optimal solution found, False otherwise
        """
        print(f"\nSolving model with {solver}...")
        
        if solver == 'CBC':
            solver_obj = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=1)
        elif solver == 'GLPK':
            solver_obj = pulp.GLPK_CMD(timeLimit=time_limit)
        else:
            solver_obj = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=1)
        
        self.model.solve(solver_obj)
        
        status = pulp.LpStatus[self.model.status]
        print(f"Optimization Status: {status}")
        
        if status == 'Optimal':
            self._extract_results()
            print(f"Optimal Cost: ₹{self.results['total_cost']:,.2f} Crore")
            return True
        else:
            print("No optimal solution found.")
            return False
    
    def _extract_results(self):
        """Extract results from solved model."""
        
        self.results = {
            'total_cost': pulp.value(self.model.objective),
            'new_capacity': {},
            'total_capacity': {},
            'generation': {},
            'generation_mix': {},
        }
        
        # Extract new capacity additions
        for t in self.technologies:
            self.results['new_capacity'][t] = {}
            self.results['total_capacity'][t] = {}
            for y in self.years:
                self.results['new_capacity'][t][y] = pulp.value(
                    self.variables['new_capacity'][(t, y)]
                )
                self.results['total_capacity'][t][y] = pulp.value(
                    self._get_total_capacity(t, y)
                )
        
        # Extract generation by technology and year
        for t in self.technologies:
            self.results['generation'][t] = {}
            for y in self.years:
                annual_gen = sum(
                    pulp.value(self.variables['generation'][(t, y, h)]) * 365
                    for h in self.hours
                ) / 1000  # Convert to GWh
                self.results['generation'][t][y] = annual_gen
        
        # Calculate generation mix percentages
        for y in self.years:
            total_gen = sum(
                self.results['generation'][t][y]
                for t in self.technologies
            )
            self.results['generation_mix'][y] = {
                t: self.results['generation'][t][y] / total_gen * 100
                if total_gen > 0 else 0
                for t in self.technologies
            }
    
    def get_capacity_df(self) -> pd.DataFrame:
        """Get capacity results as DataFrame."""
        if not self.results:
            raise ValueError("Model not solved yet")
            
        data = []
        for t in self.technologies:
            for y in self.years:
                data.append({
                    'technology': t,
                    'year': y,
                    'new_capacity_mw': self.results['new_capacity'][t][y],
                    'total_capacity_mw': self.results['total_capacity'][t][y]
                })
        return pd.DataFrame(data)
    
    def get_generation_df(self) -> pd.DataFrame:
        """Get generation results as DataFrame."""
        if not self.results:
            raise ValueError("Model not solved yet")
            
        data = []
        for t in self.technologies:
            for y in self.years:
                data.append({
                    'technology': t,
                    'year': y,
                    'generation_gwh': self.results['generation'][t][y],
                    'share_pct': self.results['generation_mix'][y][t]
                })
        return pd.DataFrame(data)
    
    def summary(self) -> str:
        """Print summary of results."""
        if not self.results:
            return "Model not solved yet"
            
        lines = [
            "=" * 60,
            "OPTIMIZATION RESULTS SUMMARY",
            "=" * 60,
            f"\nTotal System Cost: ₹{self.results['total_cost']:,.2f} Crore",
            "\nCapacity Additions by 2030 (MW):",
        ]
        
        for t in self.technologies:
            total_new = sum(self.results['new_capacity'][t][y] for y in self.years)
            if total_new > 0:
                lines.append(f"  {t}: {total_new:,.0f} MW")
        
        lines.append("\nGeneration Mix in 2030 (%):")
        for t in self.technologies:
            share = self.results['generation_mix'][2030][t]
            if share > 0.1:
                lines.append(f"  {t}: {share:.1f}%")
        
        # Calculate RE share
        re_share = sum(
            self.results['generation_mix'][2030][t]
            for t in self.re_techs
        )
        lines.append(f"\nRenewable Energy Share in 2030: {re_share:.1f}%")
        
        return "\n".join(lines)


def run_scenario(data: DataLoader, scenario_name: str, 
                 config: dict) -> CapacityExpansionModel:
    """
    Convenience function to run a single scenario.
    
    Args:
        data: DataLoader with input data
        scenario_name: Name for the scenario
        config: Configuration dictionary
        
    Returns:
        Solved model
    """
    print(f"\n{'='*60}")
    print(f"Running Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    model = CapacityExpansionModel(data, config)
    model.build()
    model.solve()
    
    print(model.summary())
    return model
