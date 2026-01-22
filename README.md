# LLoCO

LLoCO (**LL**M-guided **o**perations and **C**ombinatorial **O**ptimization) is a project that aims to bridge the gap between Operations Research and a non expert user. The system requires a high-level and possibly ambiguous description of the problem as input and returns a report detailing actionnable insights based on the solution found. Currently, LLoCO is able to solve Linear Programs, as well as Mixed-Integer Linear Programs. It also only supports CSV files as data input.  An architecure overview of the system is available ![LLoCO system architecture](/assets/images/LLoCO_arch.png)

## How to use
The projects is structured as follows:
.
└── LLoCo/
    ├── main.py
    ├── prompts/
    ├── ...
    ├── problems/
    │   └── problem_name/
    │       ├── user_input.md
    │       ├── data.csv
    │       └── some_more_data.csv
    └── datasets/
        └── eval_datasets

A `user_input.md` file should be written inside the corresponding `problem_name` folder under `problems`. To run LLoCO:
`python3 main.py -f problem_name`
More arguments can be found using:
`python3 main.py -h`
A file summarizing raw optimization results can be found under `problem_name` as `optim_summary.txt`. In the same folder, the full report is written inside `report.txt`.

## Install
To use LLoCO, you first need to install the required libraries:
`pip3 install -r requirements.txt`

LLoCO uses openai models to run : o3 and GPT-5. Your api key should be saved inside `api_key.txt` at the root the LLoCO directory to be found.