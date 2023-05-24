import csv
from copy import deepcopy

from ortools.sat.python import cp_model
from pandas import DataFrame, concat


class OptimizerMLB:
    """Class containing methods and attributes related to building and running the
    constraint programming optimization model.
    """

    def __init__(
        self, hitters: DataFrame, pitchers: DataFrame, dummies: dict[DataFrame]
    ):
        """
        Parameters
        ----------
        hitters : DataFrame
            Hitters data after splitting in ``data_processing.hitter_pitcher_split``.
        pitchers : DataFrame
            Pitchers data after splitting in ``data_processing.hitter_pitcher_split``.
        dummies : dict of DataFrames
            Dictionary with dummy DataFrames for "hitter_name", "pitcher_opp",
            "pitcher_game", "hitter_game", and "hitter_team".
        """
        self.pitchers = pitchers.copy()
        self.hitters = hitters.copy()
        self.dummies = dummies.copy()

        # Attributes used to hold created lineup metadata
        # Empty until calling the ``self.run_lineups`` method
        self.pitcher_indexes = []
        self.hitter_indexes = []
        self.binary_lineups_pitchers = []
        self.binary_lineups_hitters = []

        # Create model and add constant constraints across all lineups
        self._create_model()
        self._add_model_constraints()

    def run_lineups(
        self,
        num_lineups: int,
        auto_stack: bool = False,
        team_stack: str = None,
        stack_num: int = 4,
        variance: int = 0,
        print_progress: bool = False,
    ) -> None:
        """Create multiple lineups using the ``self.create_lineup`` method.

        Parameters
        ----------
        num_lineups : int
            Number of lineups to create.
        auto_stack : bool, default False
            If True, force each lineup to include at least 4 players from same team.
            ``team_stack`` must be None if ``auto_stack`` is True.
        team_stack : str, default None
            Team abbreviation to stack at least ``stack_num`` number of
            players from. ``auto_stack`` must be False if this is not None.
        stack_num : int, default 4, must be between 0 and 5 (inclusive)
            If ``team_stack`` is not None, lineup must have at least this number
            of hitters from team ``team_stack``. If this number is 0 then no
            players from ``team_stack`` will be included in lineup.
        variance : int, default 0, Max=10, Min=0
            The min number of players that must be different in every lineup.
            Uses ``binary_lineups`` parameter as list of already created lineups
            and creates constraint that each lineup must have ``variance``
            number of different players in each lineup. If parameter=0, every
            lineup would be the same as no players would need to be different
            from previous lineups. If parameter=10, then every lineup would
            have to be completely unique with no single player being the same
            in any lineup.
        print_progress : bool, default False
            If True, print update after each lineup is created.

        Returns
        -------
        None
            Created lineup metadata is held in instance attributes ``pitcher_indexes``,
            ``hitter_indexes``, ``binary_lineups_pitchers``, ``binary_lineups_hitters``.
        """
        for i in range(num_lineups):
            lineup = self.create_lineup(
                self.binary_lineups_pitchers,
                self.binary_lineups_hitters,
                auto_stack=auto_stack,
                team_stack=team_stack,
                stack_num=stack_num,
                variance=variance,
            )

            self.pitcher_indexes.append(lineup["pitcher_indexes"])
            self.hitter_indexes.append(lineup["hitter_indexes"])
            self.binary_lineups_pitchers.append(lineup["binary_pitchers"])
            self.binary_lineups_hitters.append(lineup["binary_hitters"])

            if print_progress:
                print(i + 1, "/", num_lineups, sep=" ")

        print("COMPLETE") if print_progress else None

    def create_lineup(
        self,
        binary_lineups_pitchers: list,
        binary_lineups_hitters: list,
        auto_stack: bool = False,
        team_stack: bool or None = None,
        stack_num: int = 4,
        variance: int = 0,
    ) -> dict[list]:
        """Runs optimization model to create a single lineup.

        Returns data for pitcher/hitter indexes and pitcher/hitter binary lists. The
        index data is used to link back the players selected in lineup to the dataframe
        holding each player's DraftKings information for later output to csv. The binary
        pitcher/hitter lists are used to create variance in each lineup.

        Parameters
        ----------
        binary_lineups_pitchers : list
            List of binary pitcher lineups with 1s and 0s for players who are
            in or out of lineup. Used with 'variance' parameter to output
            unique lineups.
        binary_lineups_hitters : list
            Same as ``binary_lineups_pitchers`` but with list representing
            hitters.
        auto_stack : bool, default False
            If True, force lineup to include at least 4 players from same team.
            ``team_stack`` must be None if ``auto_stack`` is True.
        team_stack : str, default None
            Team abbreviation to stack at least ``stack_num`` number of
            players from. ``auto_stack`` must be False if this is not None.
        stack_num : int, default 4, must be between 0 and 5 (inclusive)
            Minimum number of players to be stacked from at least one team.
            Only applied if either ``auto_stack`` is True or ``team_stack`` is not
            None. If ``auto_stack`` is True, the team that is stacked will be
            selected by algotithm. If ``team_stack`` is specified, this will be
            the minimum number of players from the specified team to be included
            in the lineup.
        variance : int, default 0, Max=10, Min=0
            The min number of players that must be different in every lineup.
            Uses 'binary_lineups' parameter as list of already created lineups
            and creates constraint that each lineup must have 'variance'
            number of different players in each lineup. If parameter=0, every
            lineup would be the same as no players would need to be different
            from previous lineups. If parameter=10, then every lineup would
            have to be completely unique with no single player being the same
            in any lineup.

        Returns
        -------
        dict of lists
            Dictionary with 4 keys, each corresponding to a different list of data.
            Dictionary keys are:
            {"pitcher_indexes", "hitter_indexes", "binary_pitchers", "binary_hitters"}
        """
        self._check_create_lineup_args(
            binary_lineups_pitchers=binary_lineups_pitchers,
            binary_lineups_hitters=binary_lineups_hitters,
            auto_stack=auto_stack,
            team_stack=team_stack,
            stack_num=stack_num,
            variance=variance,
        )
        # Start with copy of model with constant constraints and add flexible ones as
        # needed below based on function args
        model = deepcopy(self.model)

        # Flexible constraints
        model = self._add_variance_constraint(
            model, binary_lineups_hitters, binary_lineups_pitchers, variance=variance
        )
        if auto_stack:
            model = self._add_autostack_constraint(model, stack_num)
        if team_stack:
            model = self._add_teamstack_constraint(
                model, team=team_stack, stack_num=stack_num
            )

        # SOLVE
        solver = self._solve_model(model)

        # Returns dict of lists - Keys:
        # {"pitcher_indexes", "hitter_indexes", "binary_pitchers", "binary_hitters"}
        return self._output_lineup(solver)

    def csv_output(self, filename: str) -> list[list]:
        """Output csv file of lineups using the self.pitcher_indexes and
        self.hitter_indexes attributes and can be uploaded to Draftkings site.

        Parameters
        ----------
        filename : str
            Name of output csv file.
        """
        assert len(self.pitcher_indexes) == len(
            self.hitter_indexes
        ), "length of pitcher_indexes and hitter_indexes attributes must be equal."

        uploadable_lineups = self.read_lineup_metadata()

        with open(filename, "w") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(uploadable_lineups)
            csvFile.close()

        return "Complete"

    def read_lineup_metadata(self):
        """Read lineup metadata into uploadable CSV format.

        Returns
        -------
        list of lists
            List of individual lineup lists. Each individual lineup list created using
            the ``self._to_readable_list`` method.
        """
        uploadable_lineups = [["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]]
        cols = ["Name + ID", "Position"]

        for i in range(len(self.pitcher_indexes)):
            lineup_df = concat(
                [
                    self.pitchers.loc[self.pitcher_indexes[i], cols],
                    self.hitters.loc[self.hitter_indexes[i], cols],
                ]
            )
            lineup_list = self._to_readable_list(lineup_df)
            uploadable_lineups.append(lineup_list)

        return uploadable_lineups

    def _check_create_lineup_args(self, **kwargs):
        """Run assertion statements to ensure inputs to ``self.create_lineup()`` are
        correct.
        """
        # Make sure lineups is list and variance is in
        assert isinstance(
            kwargs["binary_lineups_pitchers"], list
        ), "'binary_lineups_pitchers' type needs to be list"
        assert isinstance(
            kwargs["binary_lineups_hitters"], list
        ), "'binary_lineups_hitters' type needs to be list"
        assert isinstance(kwargs["variance"], int), "'variance' type needs to be int"
        assert (
            kwargs["variance"] >= 0 and kwargs["variance"] <= 10
        ), "'variance' must be ≥= 0 and <=10"
        assert (
            isinstance(kwargs["team_stack"], str) or kwargs["team_stack"] is None
        ), "'team_stack' must be type string or None"
        assert not kwargs["auto_stack"] or not kwargs["team_stack"], (
            "At least one of 'auto_stack' and 'team_stack' must be " "False/None."
        )
        assert (
            kwargs["stack_num"] >= 0 and kwargs["stack_num"] <= 5
        ), "'stack_num' must be >= 0 and <= 5"

    def _create_model(self):
        """Create the Constraint Programming model object and decision variables."""
        self.model = cp_model.CpModel()

        # List of decision variables for each position
        self.pitchers_var = [
            self.model.NewBoolVar(self.pitchers.iloc[i]["Name"])
            for i in range(len(self.pitchers))
        ]
        self.hitters_var = [
            self.model.NewBoolVar(self.hitters.iloc[i]["Name"])
            for i in range(len(self.hitters))
        ]

    def _add_model_constraints(self):
        """Add constraints to ``self.model`` that are constant across all lineups."""
        # NUMBER PLAYERS constraint
        # Also doubles as an All Different constraint b/c 10 unique players
        # need to be selected
        self.model.Add(
            sum([self.pitchers_var[i] for i in range(len(self.pitchers_var))]) == 2
        )
        self.model.Add(
            sum([self.hitters_var[i] for i in range(len(self.hitters_var))]) == 8
        )

        # Make sure each player name is used only once - account for dual
        # position players
        for col in self.dummies["hitter_name"].columns:
            self.model.Add(
                sum(
                    [
                        self.hitters_var[i] * self.dummies["hitter_name"].loc[i, col]
                        for i in range(len(self.hitters_var))
                    ]
                )
                <= 1
            )
        # SALARY constraint
        self.model.Add(
            sum(
                [
                    self.pitchers_var[i] * self.pitchers.loc[i]["Salary"]
                    for i in range(len(self.pitchers_var))
                ]
            )
            + sum(
                [
                    self.hitters_var[i] * self.hitters.loc[i]["Salary"]
                    for i in range(len(self.hitters_var))
                ]
            )
            <= 50000
        )
        # POSITON constraints
        self.model.Add(
            sum(
                [
                    self.hitters_var[i] * self.hitters.loc[i]["C_bool"]
                    for i in range(len(self.hitters_var))
                ]
            )
            == 1
        )
        self.model.Add(
            sum(
                [
                    self.hitters_var[i] * self.hitters.loc[i]["1B_bool"]
                    for i in range(len(self.hitters_var))
                ]
            )
            == 1
        )
        self.model.Add(
            sum(
                [
                    self.hitters_var[i] * self.hitters.loc[i]["2B_bool"]
                    for i in range(len(self.hitters_var))
                ]
            )
            == 1
        )
        self.model.Add(
            sum(
                [
                    self.hitters_var[i] * self.hitters.loc[i]["3B_bool"]
                    for i in range(len(self.hitters_var))
                ]
            )
            == 1
        )
        self.model.Add(
            sum(
                [
                    self.hitters_var[i] * self.hitters.loc[i]["SS_bool"]
                    for i in range(len(self.hitters_var))
                ]
            )
            == 1
        )
        self.model.Add(
            sum(
                [
                    self.hitters_var[i] * self.hitters.loc[i]["OF_bool"]
                    for i in range(len(self.hitters_var))
                ]
            )
            == 3
        )
        # PITCHER NOT FACING HITTERS constraint
        for i in range(len(self.pitchers_var)):
            team = self.pitchers.loc[i, "team"]
            self.model.Add(
                8 * self.pitchers_var[i]
                + sum(
                    [
                        self.hitters_var[k] * self.dummies["pitcher_opp"].loc[k, team]
                        for k in range(len(self.hitters_var))
                    ]
                )
                <= 8
            )
        # PLAYERS FROM AT LEAST 2 GAMES constraint
        # sum pitcher and hitter from any game and has to be less than 9
        # 10 players in each game, meaning at most 9 can be selected
        # from a single game
        for game in self.dummies["pitcher_game"].columns:
            self.model.Add(
                sum(
                    [
                        self.pitchers_var[i] * self.dummies["pitcher_game"].loc[i, game]
                        for i in range(len(self.pitchers_var))
                    ]
                )
                + sum(
                    [
                        self.hitters_var[i] * self.dummies["hitter_game"].loc[i, game]
                        for i in range(len(self.hitters_var))
                    ]
                )
                <= 9
            )
        # NO MORE THAN 5 HITTERS FROM ONE TEAM constraint
        for team in self.dummies["hitter_team"].columns:
            self.model.Add(
                sum(
                    [
                        self.hitters_var[i] * self.dummies["hitter_team"].loc[i, team]
                        for i in range(len(self.hitters_var))
                    ]
                )
                <= 5
            )

    def _add_variance_constraint(
        self,
        model: cp_model.CpModel,
        binary_lineups_hitters: list[list[int]],
        binary_lineups_pitchers: list[list[int]],
        variance: int,
    ) -> cp_model.CpModel:
        """Add variance constraint to model.

        This constraint is the number of players in each new lineup that must be
        different from any previously created lineup. Uses the combination of indexes
        from each inner list in both ``binary_lineups_hitters`` and
        ``binary_lineups_pitchers``to generate new lineup with at least ``variance``
        number of different players from any previous lineup combination.

        Parameters
        ----------
        model : cp_model.CpModel
        binary_lineups_hitters : list of lists of 0s and 1s
            List of hitter lineup lists. Each inner lineup list contains 0s and 1s
            representing if a player at that index is in the lineup or not.
        binary_lineups_pitchers : list of lists of 0s and 1s
            List of pitcher lineup lists. Each inner lineup list contains 0s and 1s
            representing if a player at that index is in the lineup or not.
        variance : int
            The min number of players that must be different in every lineup. Uses
            'binary_lineups' parameters as list of already created lineups to create
            constraint that each lineup must have ``variance`` number of different
            players in each lineup. If ``variance=0``, every lineup would be the same
            as no players would need to be different from previous lineups. If
            ``variance=10``, then every lineup would have to be completely unique with
            no single player being the same in any lineup.

        Returns
        -------
        cp_model.CpModel
            Constraint programming``model`` with added "variance" constraint.
        """
        # Need to subtract or else `variance` would be number of same players allowed
        # in each lineup instead of the min number of players that must be different
        variance = 10 - variance

        # loops through each previous binary lineup and says that the sum of the same
        # players from lineup (combination of binary hitters and pitchers) cannot be
        # more than ``variance``
        for k in range(len(binary_lineups_pitchers)):
            model.Add(
                sum(
                    [
                        binary_lineups_pitchers[k][i] * self.pitchers_var[i]
                        for i in range(len(self.pitchers_var))
                    ]
                )
                + sum(
                    [
                        binary_lineups_hitters[k][i] * self.hitters_var[i]
                        for i in range(len(self.hitters_var))
                    ]
                )
                <= variance
            )

        return model

    def _add_autostack_constraint(
        self, model: cp_model.CpModel, stack_num: int = 4
    ) -> cp_model.CpModel:
        """Add auto stacking constraint to model.

        Auto stacking constraint will stack at least ``stack_num`` players from
        one of the available teams based on the projected points of each available
        player. This differs from ``self.add_teamstack_constraint`` method because
        this method does not allow for deciding which team to stack.

        Parameters
        ----------
        model : cp_model.CpModel
        stack_num : int, optional, by default 4
            Number of players to stack from a single team (team decided by model).

        Returns
        -------
        cp_model.CpModel
            Constraint programming``model`` with added stacking constraint.
        """
        # Make each unique team a decision variable
        # sum(players from single team) >= (stack_num * team_stack_var)
        teams = self.dummies["hitter_team"].columns
        team_stack_var = [model.NewBoolVar(teams[i]) for i in range(len(teams))]
        for t in range(len(teams)):
            model.Add(
                sum(
                    [
                        self.hitters_var[i]
                        * self.dummies["hitter_team"].loc[i, teams[t]]
                        for i in range(len(self.hitters_var))
                    ]
                )
                >= stack_num * team_stack_var[t]
            )
        # Forces at least one team to have at least ``stack_num`` players in lineup
        # `team_stack_var` >= 1
        model.Add(sum([team_stack_var[i] for i in range(len(team_stack_var))]) >= 1)

        return model

    def _add_teamstack_constraint(
        self, model: cp_model.CpModel, team: str, stack_num: int = 4
    ) -> cp_model.CpModel:
        """Add team stacking constraint to ``model``.

        Team stacking constraint forces the optimizer model to put at least
        ``stack_num`` number of players from the same ``team`` in a single lineup.
        This differs from ``self.add_autostack_constraint`` method because
        this method allows for user to decide which team to stack.

        Parameters
        ----------
        model : cp_model.CpModel
        team : str
            Team to stack. Should be abbreviation based on whatever abbreviations
            DraftKings uses.
        stack_num : int, optional, by default 4
            Number of players from ``team`` to stack.

        Returns
        -------
        cp_model.CpModel
            Constraint programming``model`` with added stacking constraint.
        """
        model.Add(
            sum(
                [
                    self.hitters_var[i] * self.dummies["hitter_team"].loc[i, team]
                    for i in range(len(self.hitters_var))
                ]
            )
            >= stack_num
        )

        return model

    def _solve_model(self, model: cp_model.CpModel) -> cp_model.CpSolver or None:
        """Solve to constraint programming model.

        Should be run after all constraints are added to the ``model``.

        Parameters
        ----------
        model : cp_model.CpModel

        Returns
        -------
        cp_model.CpSolver or None
            If solver finds and optimal solution, the ``CpSolver`` object will be returned,
            otherwise will return ``None``
        """
        model.Maximize(
            sum(
                [
                    self.pitchers_var[i] * self.pitchers.loc[i]["ppg_projection"]
                    for i in range(len(self.pitchers_var))
                ]
            )
            + sum(
                [
                    self.hitters_var[i] * self.hitters.loc[i]["ppg_projection"]
                    for i in range(len(self.hitters_var))
                ]
            )
        )

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # if not optimal, print the status and return None
        if status == cp_model.OPTIMAL:
            return solver
        elif status == cp_model.FEASIBLE:
            print("Feasible")
            return None
        elif status == cp_model.INFEASIBLE:
            print("Infeasible")
            return None
        elif status == cp_model.MODEL_INVALID:
            print("Model Invalid")
            return None

    def _output_lineup(self, solver: cp_model.CpSolver) -> dict[list]:
        """Return metadata for lineup based on ``solver`` solution.

        Parameters
        ----------
        solver : cp_model.CpSolver

        Returns
        -------
        dict of lists
            Dictionary contains 4 keys, each corresponding to player indexes or binary
            data. The player indexes are used to link back to the dataframe containing
            detailed information such as player ID, for later output to csv. The binary
            data is used to keep track of each lineup and create variance in each during
            the ``self.create_lineup`` method.
        """
        pitcher_indexes = []
        hitter_indexes = []

        # Binary lists of decision variable results (1 = in lineup, 0 = not in lineup)
        binary_pitchers = []
        binary_hitters = []

        # Append player indexes that solver has chosen and create binary list
        for j in range(len(self.pitchers_var)):
            if solver.Value(self.pitchers_var[j]) == 1:
                pitcher_indexes.append(self.pitchers.iloc[j]["index"])
                binary_pitchers.append(1)
            else:
                binary_pitchers.append(0)

        for j in range(len(self.hitters_var)):
            if solver.Value(self.hitters_var[j]) == 1:
                hitter_indexes.append(self.hitters.iloc[j]["index"])
                binary_hitters.append(1)
            else:
                binary_hitters.append(0)

        # Player indexes used to link back to df and get player IDs
        # Binary lineups list used for variance metric
        return {
            "pitcher_indexes": pitcher_indexes,
            "hitter_indexes": hitter_indexes,
            "binary_pitchers": binary_pitchers,
            "binary_hitters": binary_hitters,
        }

    def _to_readable_list(self, df: DataFrame) -> list[str]:
        """Turn ``df`` of players in lineup into list of player IDs.

        CSV file needs to be in this format to get uploaded correctly to DraftKings.

        Parameters
        ----------
        df : DataFrame
            Lineup of 10 players.

        Returns
        -------
        list
            Players "Name + ID" ordered according to the template needed to upload data
            to DraftKings site.
        """
        lineup_list = []
        lineup_list.extend(df[df["Position"].str.contains("P")]["Name + ID"].values)
        lineup_list.extend(df[df["Position"] == "C"]["Name + ID"].values)
        lineup_list.extend(df[df["Position"] == "1B"]["Name + ID"].values)
        lineup_list.extend(df[df["Position"] == "2B"]["Name + ID"].values)
        lineup_list.extend(df[df["Position"] == "3B"]["Name + ID"].values)
        lineup_list.extend(df[df["Position"] == "SS"]["Name + ID"].values)
        lineup_list.extend(df[df["Position"] == "OF"]["Name + ID"].values)

        return lineup_list
