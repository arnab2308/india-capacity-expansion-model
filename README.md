# India Power Sector Capacity Expansion Optimization Model (2025-2030)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arnab2308/india-capacity-expansion-model/blob/main/notebooks/India_Capacity_Expansion_Model.ipynb)

## Overview

A least-cost capacity expansion planning model for India's power sector, optimizing the generation mix to meet 2030 electricity demand while achieving the 500 GW non-fossil fuel target under the Panchamrit climate commitments.

## Key Features

- **Linear Programming Optimization** using PuLP
- **Multi-year planning horizon** (2025-2030)
- **Hourly dispatch modeling** with representative days
- **Technology options**: Solar, Wind, Coal, Gas, Battery Storage, Pumped Hydro
- **Scenario analysis**: Base case, High RE, High Demand, Low Storage Cost
- **Real data** from CEA, MNRE, and IRENA sources

## Quick Start (Google Colab)

1. Click the "Open in Colab" badge above
2. Run all cells - no local installation required
3. View results and visualizations

## Project Structure

```
india-capacity-expansion-model/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── demand_forecast.csv       # CEA demand projections
│   ├── technology_costs.csv      # Capital, O&M, fuel costs
│   ├── existing_capacity.csv     # Current installed capacity
│   ├── hourly_profiles.csv       # Solar/wind capacity factors by hour
│   └── emissions_factors.csv     # CO2 emissions by technology
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading utilities
│   ├── model.py                  # Core optimization model
│   └── visualizations.py         # Plotting functions
│
├── notebooks/
│   └── India_Capacity_Expansion_Model.ipynb   # Main notebook
│
└── outputs/
    ├── optimal_capacity_mix.csv
    ├── generation_dispatch.csv
    └── figures/
```

## Model Description

### Objective Function
Minimize total system cost over the planning horizon:
- Capital costs (annualized)
- Fixed O&M costs
- Variable O&M costs
- Fuel costs

### Key Constraints
1. **Demand Balance**: Generation must meet demand in each hour
2. **Capacity Limits**: Generation limited by installed capacity × capacity factor
3. **RE Target**: 50% of generation from non-fossil sources by 2030
4. **Reserve Margin**: 15% above peak demand for reliability
5. **Storage**: Energy balance, charging/discharging limits

### Technologies Modeled

| Technology | Capital Cost (₹Cr/MW) | Variable Cost (₹/kWh) | Capacity Factor |
|------------|----------------------|----------------------|-----------------|
| Solar PV | 4.0 | 0 | 22% (varies hourly) |
| Wind | 6.5 | 0 | 30% (varies hourly) |
| Coal (existing) | 0 (sunk) | 2.50 | 60% |
| Coal (new) | 8.0 | 2.25 | 65% |
| Gas CCGT | 5.0 | 4.80 | 25% |
| Battery (4hr) | 1.5/MWh | 0 | N/A |
| Pumped Hydro | 8.0 | 0.05 | N/A |

## Data Sources

- **Demand Projections**: Central Electricity Authority (CEA) 20th Electric Power Survey
- **Existing Capacity**: CEA Monthly Reports (2025)
- **Cost Data**: IRENA Renewable Power Generation Costs 2024, CEA Technical Standards
- **RE Targets**: India's NDC under Paris Agreement, Panchamrit Goals

## Results Summary

The model outputs include:
1. **Optimal capacity additions** by technology and year
2. **Generation dispatch** patterns
3. **System costs** breakdown
4. **Emissions trajectory**
5. **Comparison with CEA's National Electricity Plan**

## Scenarios Analyzed

| Scenario | Description |
|----------|-------------|
| Base Case | 500 GW non-fossil by 2030, CEA demand projections |
| High RE | 600 GW non-fossil target |
| High Demand | 7% annual demand growth |
| Low Storage Cost | 50% reduction in battery costs |

## Author

Arnab Banerjee

## References

1. CEA (2023). Report on Optimal Generation Capacity Mix for 2029-30
2. CEA (2023). National Electricity Plan 2022-32
3. IRENA (2024). Renewable Power Generation Costs in 2024
4. CEEW (2025). How Can India Meet Peak Power Demand With Clean Electricity?

## License

MIT License - feel free to use and modify for your own projects.
