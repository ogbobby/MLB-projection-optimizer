from pandas import DataFrame, read_csv

from mlboptimizer.data_processing import create_dummy_dfs, transform_data
from mlboptimizer.optimizer_mlb import OptimizerMLB

# CONFIG - change constants as needed
DATE = "2021-09-04_MAIN"
TEAM_MAP = {
    # Only used if running `main_teamstack`
    "CWS": 15,
    "CIN": 15,
    "LAA": 14,
    "LAD": 14,
    "SEA": 14,
    "MIL": 14,
    "ARI": 14,
}


def read_data(date: str) -> tuple[DataFrame]:
    """Read and transform data files for input to ``OptimizerMLB``."""
    # Read in data files
    filename_dk = "./data-dk/DKSalaries_" + date + ".csv"
    filename_proj = "./data-projected/DFF_MLB_cheatsheet_" + date + ".csv"

    data_dk = read_csv(filename_dk)
    data_proj = read_csv(filename_proj)

    # Clean and process data for input to optimizer
    hitters, pitchers = transform_data(data_dk, data_proj)
    dummies = create_dummy_dfs(hitters, pitchers)

    return hitters, pitchers, dummies


def main_autostack(date: str) -> None:
    out_filename = str(input("csv filename: "))
    data = read_data(date)
    optimizer = OptimizerMLB(*data)

    # Create x # of auto stacked lineups
    num_lineups = int(input("number of lineups: "))
    optimizer.run_lineups(
        num_lineups, print_progress=True, auto_stack=True, stack_num=4, variance=2
    )

    # Export to CSV #
    optimizer.csv_output(out_filename)


def main_teamstack(date: str, team_map: dict) -> None:
    out_filename = str(input("csv filename: "))
    data = read_data(date)
    optimizer = OptimizerMLB(*data)

    for team, num_lineups in team_map.items():
        print("Team: " + team)
        optimizer.run_lineups(
            num_lineups,
            team_stack=team,
            stack_num=4,
            variance=1,
            print_progress=True,
        )
        print("-" * 10)

    # Export to CSV #
    optimizer.csv_output(out_filename)


if __name__ == "__main__":
    stack_type = input("Type of lineup stack ('auto' or 'team'): ").lower()
    if stack_type == "auto":
        main_autostack(DATE)
    elif stack_type == "team":
        main_teamstack(DATE, team_map=TEAM_MAP)
    else:
        raise ValueError("Type of lineup stack value must be 'auto' or 'team'")
