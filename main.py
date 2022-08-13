from pandas import read_csv

from mlboptimizer.data_processing import create_dummy_dfs, transform_data
from mlboptimizer.optimizer_mlb import OptimizerMLB

if __name__ == "__main__":
    ########## READ AND CLEAN DATA ##########
    date = "2021-09-04_MAIN"
    filename = str(input("csv filename: "))

    filename_dk = "./data-dk/DKSalaries_" + date + ".csv"
    filename_proj = "./data-projected/DFF_MLB_cheatsheet_" + date + ".csv"

    data_dk = read_csv(filename_dk)
    data_proj = read_csv(filename_proj)

    # Merge projections to DK salary data to make sure have correct salary AND
    # positional information for each player
    # projections for some sources only have one position
    # Filter out players with no projected points later on

    hitters, pitchers = transform_data(data_dk, data_proj)
    dummies = create_dummy_dfs(hitters, pitchers)

    optimizer = OptimizerMLB(pitchers, hitters, dummies)

    # Create x # of auto stacked lineups
    num_lineups = int(input("number of lineups: "))
    optimizer.run_lineups(
        num_lineups, print_progress=True, auto_stack=True, stack_num=4, variance=2
    )

    # Export to CSV #
    optimizer.csv_output(filename)

    # #%%
    ### Create lineups with certain teams stacked
    #    stack_teams = str(input("Teams to stack (comma separated):\n")).split(", ")
    #    num_lineups_per_team = int(input("Number of lineups per team:\n"))

    # stack_teams_dict = dict(input("Dictionary - Team : Num lineups"))
    # stack_teams_dict = {
    #     "CWS": 15,
    #     "CIN": 15,
    #     "LAA": 14,
    #     "LAD": 14,
    #     "SEA": 14,
    #     "MIL": 14,
    #     "ARI": 14,
    # }

    # for team, num_lineups_team in stack_teams_dict.items():
    #     print("Team: " + team)
    #     optimizer.run_lineups(
    #         num_lineups_team,
    #         team_stack=team,
    #         stack_num=4,
    #         variance=1,
    #         print_progress=True,
    #     )
    #     print("-" * 10)

    # # Export to CSV #
    # optimizer.csv_output(filename)
