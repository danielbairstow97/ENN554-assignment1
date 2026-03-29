# Ass1

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- Turbine and load data.
│
├── notebooks          <- Jupyter notebookds for doing test runs of the modelling.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         ass1 and configuration for tools like enn554.
├── enn554             <- Mike's git repo incorporated into assignment.
|
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
|
└── ass1   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ass1 a Python module
    ├── config.py               <- Store useful variables and configuration
    ├── plots.py                <- Code to create visualizations for report. #TODO
    │
    └── modeling
       ├── __init__.py
       ├── financial.py         <- Code containing financial settings of the model
       ├── loaders.py           <- Loads turbine and load data for the model.
       ├── location.py          <- Contains details about the modelled location such as wind direction and speed.
       ├── turbine.py           <- Models individual turbine performance.
       └── model.py             <- To be implemented: Model that runs the optimisation of battery size for a given wind farm at a location.
```

--------

