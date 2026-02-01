"""
Visualizations Module
=====================
Plotting functions for capacity expansion model results.

Creates publication-quality charts for:
- Capacity mix evolution
- Generation dispatch
- Cost breakdown
- Scenario comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from .model import CapacityExpansionModel


# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme for technologies
TECH_COLORS = {
    'coal_existing': '#4d4d4d',
    'coal_new': '#737373',
    'gas_ccgt': '#f4a582',
    'solar': '#ffd700',
    'wind': '#4daf4a',
    'hydro_large': '#377eb8',
    'nuclear': '#984ea3',
    'bess_4hr': '#ff7f00',
    'psh_8hr': '#a65628',
}

# Technology display names
TECH_NAMES = {
    'coal_existing': 'Coal (Existing)',
    'coal_new': 'Coal (New)',
    'gas_ccgt': 'Gas CCGT',
    'solar': 'Solar PV',
    'wind': 'Wind',
    'hydro_large': 'Large Hydro',
    'nuclear': 'Nuclear',
    'bess_4hr': 'Battery (4hr)',
    'psh_8hr': 'Pumped Hydro',
}


class ResultsVisualizer:
    """
    Creates visualizations for capacity expansion model results.
    
    Attributes:
        model (CapacityExpansionModel): Solved model
        output_dir (str): Directory to save figures
    """
    
    def __init__(self, model: CapacityExpansionModel, output_dir: str = "outputs/figures"):
        """
        Initialize visualizer.
        
        Args:
            model: Solved CapacityExpansionModel
            output_dir: Directory for saving figures
        """
        self.model = model
        self.output_dir = output_dir
        self.results = model.results
        
    def plot_capacity_mix(self, save: bool = True) -> plt.Figure:
        """
        Plot stacked bar chart of capacity mix evolution.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        years = self.model.years
        technologies = [t for t in self.model.technologies 
                       if any(self.results['total_capacity'][t][y] > 100 
                             for y in years)]
        
        # Prepare data for stacked bar
        bottom = np.zeros(len(years))
        
        for tech in technologies:
            values = [self.results['total_capacity'][tech][y] / 1000 
                     for y in years]  # Convert to GW
            ax.bar(years, values, bottom=bottom, 
                   label=TECH_NAMES.get(tech, tech),
                   color=TECH_COLORS.get(tech, '#999999'),
                   width=0.7)
            bottom += np.array(values)
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Installed Capacity (GW)', fontsize=12)
        ax.set_title('India Power Sector: Optimal Capacity Mix (2025-2030)', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xticks(years)
        
        # Add total labels
        for i, y in enumerate(years):
            total = sum(self.results['total_capacity'][t][y] 
                       for t in technologies) / 1000
            ax.annotate(f'{total:.0f} GW', 
                       xy=(y, total), 
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/capacity_mix.png", dpi=150, 
                       bbox_inches='tight')
        
        return fig
    
    def plot_generation_mix(self, save: bool = True) -> plt.Figure:
        """
        Plot generation mix as stacked area chart.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        years = self.model.years
        technologies = [t for t in self.model.technologies 
                       if any(self.results['generation'][t][y] > 1 
                             for y in years)]
        
        # Prepare data
        data = {tech: [self.results['generation'][tech][y] / 1000 
                       for y in years]  # Convert to TWh
                for tech in technologies}
        
        # Sort technologies by 2030 generation
        technologies = sorted(technologies, 
                            key=lambda t: self.results['generation'][t][2030],
                            reverse=True)
        
        # Create stacked area
        y_stack = np.zeros(len(years))
        for tech in technologies:
            values = np.array(data[tech])
            ax.fill_between(years, y_stack, y_stack + values,
                           label=TECH_NAMES.get(tech, tech),
                           color=TECH_COLORS.get(tech, '#999999'),
                           alpha=0.8)
            y_stack += values
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Generation (TWh)', fontsize=12)
        ax.set_title('India Power Sector: Generation Mix Evolution', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xticks(years)
        ax.set_xlim(years[0], years[-1])
        ax.set_ylim(0)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/generation_mix.png", dpi=150,
                       bbox_inches='tight')
        
        return fig
    
    def plot_generation_pie_2030(self, save: bool = True) -> plt.Figure:
        """
        Plot pie chart of generation mix in 2030.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get 2030 generation shares
        shares = self.results['generation_mix'][2030]
        
        # Filter technologies with >1% share
        labels = []
        sizes = []
        colors = []
        explode = []
        
        for tech in self.model.technologies:
            if shares[tech] > 1:
                labels.append(TECH_NAMES.get(tech, tech))
                sizes.append(shares[tech])
                colors.append(TECH_COLORS.get(tech, '#999999'))
                # Explode renewable sources
                explode.append(0.05 if tech in self.model.re_techs else 0)
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            pctdistance=0.75, labeldistance=1.1
        )
        
        # Formatting
        plt.setp(autotexts, size=10, weight='bold')
        ax.set_title('Generation Mix in 2030', fontsize=14, fontweight='bold')
        
        # Add RE total annotation
        re_share = sum(shares[t] for t in self.model.re_techs)
        ax.annotate(f'Renewable Share: {re_share:.1f}%',
                   xy=(0, -1.3), ha='center', fontsize=12, fontweight='bold',
                   color='green')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/generation_pie_2030.png", dpi=150,
                       bbox_inches='tight')
        
        return fig
    
    def plot_new_capacity_additions(self, save: bool = True) -> plt.Figure:
        """
        Plot new capacity additions by year.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        years = self.model.years
        technologies = [t for t in self.model.technologies 
                       if sum(self.results['new_capacity'][t][y] 
                             for y in years) > 100]
        
        x = np.arange(len(years))
        width = 0.8 / len(technologies)
        
        for i, tech in enumerate(technologies):
            values = [self.results['new_capacity'][tech][y] / 1000 
                     for y in years]  # GW
            ax.bar(x + i * width, values, width,
                  label=TECH_NAMES.get(tech, tech),
                  color=TECH_COLORS.get(tech, '#999999'))
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('New Capacity Addition (GW)', fontsize=12)
        ax.set_title('Annual Capacity Additions by Technology', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * len(technologies) / 2)
        ax.set_xticklabels(years)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/new_capacity.png", dpi=150,
                       bbox_inches='tight')
        
        return fig
    
    def plot_hourly_dispatch(self, year: int = 2030, 
                             save: bool = True) -> plt.Figure:
        """
        Plot hourly generation dispatch for a representative day.
        
        Args:
            year: Year to plot
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        hours = self.model.hours
        technologies = [t for t in self.model.technologies if t not in self.model.storage_techs]
        
        # Get hourly generation
        gen_data = {}
        for tech in technologies:
            gen_data[tech] = [
                self.model.variables['generation'][(tech, year, h)].varValue or 0
                for h in hours
            ]
        
        # Sort by baseload first
        baseload_order = ['nuclear', 'hydro_large', 'coal_existing', 'coal_new', 
                         'gas_ccgt', 'wind', 'solar']
        technologies = [t for t in baseload_order if t in technologies]
        
        # Stacked area
        y_stack = np.zeros(len(hours))
        for tech in technologies:
            values = np.array(gen_data[tech]) / 1000  # GW
            ax.fill_between(hours, y_stack, y_stack + values,
                           label=TECH_NAMES.get(tech, tech),
                           color=TECH_COLORS.get(tech, '#999999'),
                           alpha=0.8)
            y_stack += values
        
        # Add demand line
        demand = self.model.data.calculate_annual_demand_by_hour(year) / 1000  # GW
        ax.plot(hours, demand, 'k--', linewidth=2, label='Demand')
        
        # Formatting
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Generation (GW)', fontsize=12)
        ax.set_title(f'Hourly Dispatch Pattern ({year})', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(0, 23)
        ax.set_ylim(0)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/hourly_dispatch_{year}.png", dpi=150,
                       bbox_inches='tight')
        
        return fig
    
    def plot_re_share_evolution(self, save: bool = True) -> plt.Figure:
        """
        Plot renewable energy share over time.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        years = self.model.years
        re_shares = [
            sum(self.results['generation_mix'][y][t] 
                for t in self.model.re_techs)
            for y in years
        ]
        
        # Plot line
        ax.plot(years, re_shares, 'go-', linewidth=2, markersize=10)
        
        # Add target line
        ax.axhline(y=50, color='r', linestyle='--', linewidth=2, 
                  label='2030 Target (50%)')
        
        # Fill area
        ax.fill_between(years, 0, re_shares, alpha=0.3, color='green')
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Renewable Energy Share (%)', fontsize=12)
        ax.set_title('Progress Towards 50% RE Target', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xticks(years)
        ax.set_ylim(0, 70)
        
        # Add labels
        for i, (y, share) in enumerate(zip(years, re_shares)):
            ax.annotate(f'{share:.1f}%', 
                       xy=(y, share), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/re_share_evolution.png", dpi=150,
                       bbox_inches='tight')
        
        return fig
    
    def create_all_plots(self) -> Dict[str, plt.Figure]:
        """
        Create all standard visualizations.
        
        Returns:
            Dictionary of figure names to figures
        """
        print("Creating visualizations...")
        
        figures = {
            'capacity_mix': self.plot_capacity_mix(),
            'generation_mix': self.plot_generation_mix(),
            'generation_pie_2030': self.plot_generation_pie_2030(),
            'new_capacity': self.plot_new_capacity_additions(),
            'hourly_dispatch': self.plot_hourly_dispatch(),
            're_share': self.plot_re_share_evolution(),
        }
        
        print(f"Created {len(figures)} visualizations")
        return figures


def compare_scenarios(models: Dict[str, CapacityExpansionModel],
                      output_dir: str = "outputs/figures") -> plt.Figure:
    """
    Compare results across multiple scenarios.
    
    Args:
        models: Dictionary of scenario name to solved model
        output_dir: Directory to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = list(models.keys())
    
    # Plot 1: Total Cost Comparison
    ax1 = axes[0]
    costs = [models[s].results['total_cost'] / 1e5 for s in scenarios]  # Lakh Crore
    bars = ax1.bar(scenarios, costs, color='steelblue')
    ax1.set_ylabel('Total System Cost (₹ Lakh Crore)', fontsize=11)
    ax1.set_title('Cost Comparison', fontsize=12, fontweight='bold')
    ax1.bar_label(bars, fmt='%.1f')
    
    # Plot 2: 2030 Capacity Mix Comparison
    ax2 = axes[1]
    
    width = 0.15
    x = np.arange(len(scenarios))
    key_techs = ['solar', 'wind', 'coal_existing', 'bess_4hr']
    
    for i, tech in enumerate(key_techs):
        values = [models[s].results['total_capacity'][tech][2030] / 1000 
                 for s in scenarios]
        ax2.bar(x + i * width, values, width,
               label=TECH_NAMES.get(tech, tech),
               color=TECH_COLORS.get(tech, '#999999'))
    
    ax2.set_ylabel('Capacity in 2030 (GW)', fontsize=11)
    ax2.set_title('Capacity Comparison (2030)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * len(key_techs) / 2)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario_comparison.png", dpi=150, 
               bbox_inches='tight')
    
    return fig
