# `Steel_CBAM`

`Steel_CBAM` is a repository designed to quantify the climate-effectiveness of the European Union's Carbon Border Adjustment Mechanism (CBAM) for the global steel industry. It combines (prospective) life cycle assessment (LCA) with regionalized production data. The code was developed in the context of project TRANSIENCE (see Acknowledgements).

---

## Repository structure

```bash
steel_cbam_assessment/
├── data/                                 # Input datasets for regional steel production, emissions, and CBAM assumptions
├── figs/                                 # Final high-quality figures and plots
├── logs/                                 # Logs from processing
├── results/                              # Processed LCA results for each scenario
├── 0_set_up_lca_db_gen_acts.ipynb        # Prepares background LCA databases using premise
├── 1_steel_dbs_setup.ipynb               # Sets up steel production databases used
├── 2_steel_assessment.ipynb              # Main notebook to perform CBAM impact assessment and for creating databases
├── 3_create_additional_prospective_...   # Additional notebook for created additional prospective regionalized LCA databases for REMIND IAM
├── 4_create_additional_figs_REMIND.ipynb # Visualizations based on REMIND IAM
├── 5_create_sensitivity_figs.ipynb       # Sensitivity analysis visualizations
├── config.py                             # Configuration and parameter settings
├── db_functions.py                       # Functions for database setup and querying
├── functions.py                          # General helper functions
├── mapping.py                            # Country-region and technology mappings
├── plotting.py                           # Figure generation utilities
├── private_keys.py                       # Local credentials (excluded from version control)
├── regionalization.py                    # Regionalization logic
└── README.md                             # This file
```

---

## How to use it

The workflow is structured in five main Jupyter Notebooks:

### `0_set_up_lca_db_gen_acts.ipynb`
Generates prospective LCA databases using the [`premise`](https://github.com/polca/premise) framework. These databases reflect different technological and energy transformation pathways.

### `1_steel_dbs_setup.ipynb`
Configures region-specific steel production routes using the databases.

### `2_steel_assessment.ipynb`
Performs the main CBAM assessment by first creating new databases and then calculating environmental impacts under various scenarios, regions, and timeframes. 

### `3_create_additional_prospective_regionalized_dbs.ipynb`
Generates additional prospective LCA databases for REMIND with regionalization to REMIND regions.

### `4_create_additional_figs_REMIND.ipynb` and `5_create_sensitivity_figs.ipynb`
Optional notebooks to generate advanced visualizations based on IAM pathways and sensitivity assumptions.

 *Each notebook includes markdown cells that explain assumptions, code logic, and visualization steps.*

---

## Dependencies

To run the notebooks, you need the following:

- `KEY_PREMISE`: API key for using the `premise` framework.
- `USER_PW`: Username and password for access to the **ecoinvent** LCA database.

Install the required environment with:

```bash
conda env create -f bw_env_hydrogen.yml
conda activate bw_reg_prem
```

---

## Output & documentation

- Results from the assessments are saved in the `results/` folders.
- High-resolution visualizations are available in the `figs/` folder.
- All notebooks are extensively documented and structured for reproducibility.

Use the outputs to:

- Evaluate CBAM policy impact across world regions
- Compare multiple LCA indicators (e.g., climate change impacts and water consumption)
- Analyze different future scenarios and sensitivity cases

---

## License, citing, and scientific references

If you use this repository, the data, or any of the included code, please cite the following paper:  
*_[Insert citation here]_*

Following the Creative Commons license of the Global Steel Plant Tracker and Green Steel Tracker, we realize that this is a work derived from their material. However, we are solely liable and responsible for this derived work, and it is not endorsed by those sources in any manner.

For licensing information, see the `LICENSE` file.

---

## Acknowledgements

This research was supported through the following funding schemes:

- 🇪🇺 **Horizon Europe project TRANSIENCE** (Project No. 101137606), financed by:
  - The European Health and Digital Executive Agency (HADEA)
  - The Swiss State Secretariat for Education, Research and Innovation (SERI)
  - UK Research and Innovation (UKRI) Horizon Europe Guarantee

> The content of this work does not necessarily reflect the opinion of the European Commission or any other funding body. Responsibility lies solely with the authors.

For more on TRANSIENCE, visit: [https://www.transience.eu](https://www.transience.eu)

---

## Contributing

We welcome your contributions and suggestions!

For major changes or collaborative work, please get in touch via:

**Tom Terlouw**  
tom.terlouw@psi.ch
