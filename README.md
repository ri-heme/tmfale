# ALE Thermodynamics

> Given that metabolite levels drive the metabolic and regulatory network re-wiring found in ALE studies, can ALE be thought of as an optimization over global thermodynamic and/or kinetic properties of metabolism?

## Components

In this repository, you will find my scripts to analyze and run TFA on ALE data. These are included:

1. **Data Pre-processing.ipynb** - Notebook where raw metabolomic and physiological data is read and formatted, so it can be input as constraints of the MILP problem.
2. **TFA.ipynb** - Notebook where the above data is fed into a TFA model that estimates an optimal distribution of log concentrations, delta G, and fluxes.
3. **Plotting.ipynb** - Separate notebook to obtain visualization of the results.

Additionally, you will find:

- **utils.py** - Module with functions to create/adjust/run a thermodynamic model and extract relevant information from it.
- **data** - Folder containing raw data, the metabolic model, thermodynamic databases, and the pre-processed data (output of notebook 1).
- **results** - Folder containing resulting estimates and figures (outputs of notebooks 2 and 3, respectively).
