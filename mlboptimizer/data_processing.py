"""
NOTE: Functions in this module will change the state of the input DataFrames.
"""
from pandas import DataFrame, concat, get_dummies


def merge_data(dk: DataFrame, proj: DataFrame) -> DataFrame:
    """Merge DraftKings data ``dk`` with point projection data ``proj``.

    Parameters
    ----------
    dk : DataFrame
        DraftKings data.
    proj : DataFrame
        Point projection data.

    Returns
    -------
    DataFrame
        Merged DataFrame between ``dk`` and ``proj``.

    Raises
    ------
    AssertionError
        Raised if the number of non-null rows in column "ppg_projection" is unequal to
        the length of the point projection df ``proj``.
    """
    # Create ID column to merge on
    proj["full_name"] = proj["first_name"] + " " + proj["last_name"]

    merged = dk.merge(
        proj[["full_name", "team", "opp", "ppg_projection", "value_projection"]],
        how="left",  # keep all players and filter out nulls later on
        left_on="Name",
        right_on="full_name",
    )

    # Check join worked
    # assert merged["ppg_projection"].notnull().sum() == proj.shape[0], "Error with merge"

    return merged


def int_cols(
    df: DataFrame,
    columns: list[str] = ["ppg_projection", "Salary"],
    point_col: str = "ppg_projection",
) -> DataFrame:
    """Turn ``df[columns]`` into integer type.

    The point column, 'point_col', will be multiplied by 10 to get rid of
    decimal if there is one. This is necessary as optimizer will give error if data
    is not of ``int`` type. ``float`` and any other data types used in optimizer
    will raise an error.

    Parameters
    ----------
    df : DataFrame
    columns : list, default ['ppg_projection', 'Salary']
        List of columns to turn into integer type.
    point_col : str, default 'FP'
        Column of projected points which is multiplied by 10 to eliminate
        decimal from number and make integer. Can be None if point column
        already is integer type. If not None, must be in the 'columns'
        parameter list.

    Returns
    -------
    DataFrame
        ``df`` with ``df[columns]`` dtypes converted to integer.
    """

    for col in columns:
        if col == point_col:
            df[col] = df[col] * 10
        df[col] = df[col].astype(int)

    return df


def dual_position(df: DataFrame, colname: str = "Position") -> DataFrame:
    """Create 2 separate rows for any player with dual positions (ex: "2B/SS").

    Parameters
    ----------
    df : DataFrame
    colname : str, default 'position'
        Name of column containing string position values for each player.

    Returns
    -------
    DataFrame
        ``df`` with additional rows for any player with dual positions.
    """
    # `duals` = only players with dual positions
    duals = df[df[colname].str.contains("/")].copy()

    # use first position in df
    df[colname] = df[colname].str.split("/").str[0]

    # use second position in duals
    duals[colname] = duals[colname].str.split("/").str[1]

    # append duals back to df
    df = concat([df, duals])

    return df


def position_bools(df: DataFrame, colname: str = "Position") -> DataFrame:
    """Add positional boolean columns to dataframe.

    Each added boolean column represents if a player has eligibility at the
    specified position.

    Parameters
    ----------
    df : DataFrame
    colname : str, default 'Position'
        Name of column containing string position values for each player.

    Returns
    -------
    DataFrame
        ``df`` with added boolean columns for each player's positional eligibility.
    """
    df[colname] = df[colname].str.upper()
    df["P_bool"] = df[colname].apply(lambda x: 1 if "P" in x else 0)
    df["C_bool"] = df[colname].apply(lambda x: 1 if "C" in x else 0)
    df["1B_bool"] = df[colname].apply(lambda x: 1 if "1B" in x else 0)
    df["2B_bool"] = df[colname].apply(lambda x: 1 if "2B" in x else 0)
    df["3B_bool"] = df[colname].apply(lambda x: 1 if "3B" in x else 0)
    df["SS_bool"] = df[colname].apply(lambda x: 1 if "SS" in x else 0)
    df["OF_bool"] = df[colname].apply(lambda x: 1 if "OF" in x else 0)

    return df


def hitter_pitcher_split(df) -> tuple[DataFrame]:
    """Split dataframe into two based on if player is a pitcher or hitter.

    Indexes for each created dataframe (``pitchers`` and ``hitters``) are reset and
    then used to create an ID column "index".

    NOTE: Should be run AFTER the ``dual_position()`` function.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    tuple of DataFrames
        (hitters, pitchers) - split and reindexed from ``df``.
    """
    hitters = df[~df["Position"].str.contains("P")].copy()
    pitchers = df[df["Position"].str.contains("P")].copy()

    # reindex hitter and pitcher dfs so both start at 0
    # reset index after splitting and filtering
    hitters.reset_index(inplace=True, drop=True)
    pitchers.reset_index(inplace=True, drop=True)
    # reset index again, but keep old index to use as player ID
    hitters.reset_index(inplace=True, drop=False)
    pitchers.reset_index(inplace=True, drop=False)

    return hitters, pitchers


def transform_data(
    dk: DataFrame,
    proj: DataFrame,
    point_minimum=1,
    to_int_cols=["ppg_projection", "Salary"],
    point_col="ppg_projection",
    position_col="Position",
):
    """Transform DataFrame for use in ``create_lineup()`` method.

    Data transformations performed:
        1. Filter to rows above ``point_minimum`` threshold.
        2. Convert ``to_int_columns`` to integer dtype.
        3. Create boolean columns based on positional eligibility.
        4. Split dataframe into 2 based on pitcher or hitter.
        5. Creates dummy dataframes for features like player name, opponent, team,
        etc. to be used as constraints in model.

    Parameters
    ----------
    point_minimum : int or float, default 1
        Minimum projected points value for rows in self.df
    to_int_columns : list, default ['ppg_projection', 'Salary']
        List of columns to turn into integer type.
    point_col : str, default 'ppg_projection'
        Column of projected points which is multiplied by 10 to eliminate
        decimal from number and make integer. Can be None if point column
        already is integer type. If not None, must be in the 'columns'
        parameter list.
    position_col : str
        Name of column containing string position values for each player.
    """
    df = merge_data(dk, proj)

    df = df[df[point_col] > point_minimum].copy()
    # NOTE: OPTIMIZER WILL NOT WORK WITH NON-INTEGER COLUMNS
    int_cols(df, columns=to_int_cols, point_col=point_col)
    dual_position(df, colname=position_col)
    position_bools(df, colname=position_col)

    hitters, pitchers = hitter_pitcher_split(df)
    return hitters, pitchers


def create_dummy_dfs(hitters: DataFrame, pitchers: DataFrame) -> dict[DataFrame]:
    """Create dummy dataframes to be used for certain constraints in model.

    DataFrames created used in constraints for player name, pitcher opponent,
    number of players eligible from single game, and hitter team (for stacking).

    Used when building model in ``mlboptimizer.OptimizerMLB.create_lineup()`` method.

    Parameters
    ----------
    hitters : DataFrame
        Hitters data after splitting in ``data_processing.hitter_pitcher_split()``.
    pitchers : DataFrame
        Pitchers data after splitting in ``data_processing.hitter_pitcher_split()``.

    Returns
    -------
    dict of dummy DataFrames
        Dictionary with dummy DataFrames for "hitter_name", "pitcher_opp",
        "pitcher_game", "hitter_game", and "hitter_team".
    """
    hitter_name = get_dummies(hitters["Name"])
    hitter_team = get_dummies(hitters["team"])
    hitter_game = get_dummies(hitters["Game Info"])
    pitcher_opp = get_dummies(hitters["opp"])
    pitcher_game = get_dummies(pitchers["Game Info"])

    dummies = {
        "hitter_name": hitter_name,
        "hitter_team": hitter_team,
        "hitter_game": hitter_game,
        "pitcher_opp": pitcher_opp,
        "pitcher_game": pitcher_game,
    }

    return dummies
