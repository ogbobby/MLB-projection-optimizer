#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 07:39:52 2021

@author: jnicoliwd

MLB Optimizer

"""

import csv
import pandas as pd
from ortools.sat.python import cp_model


class OptimizerMLB:
    
    def __init__(self, df):
        """
        Parameters
        ----------
        df : pandas.DataFrame
        """
        
        self.df = df.copy()
    
    
    def int_cols(self, columns=['ppg_projection', 'Salary'], 
                 point_col='ppg_projection'):
        """
        Turns columns listed in 'columns' parameter into integer columns. The
        point column, 'point_col', will be multiplied by 10 to get rid of 
        decimal if there is one. 
        
        Need to turn all columns into integer columns as optimizer only works
        over integers and any float columns will return errors.
        
        Parameters
        ----------
        columns : list, default ['ppg_projection', 'Salary']
            List of columns to turn into integer type.
        point_col : str, default 'FP'
            Column of projected points which is multiplied by 10 to eliminate
            decimal from number and make integer. Can be None if point column
            already is integer type. If not None, must be in the 'columns' 
            parameter list.
        """
        
        for col in columns:
            if col == point_col:
                self.df[col] = self.df[col] * 10
            
            self.df[col] = self.df[col].astype(int)
            
    
    def dual_position(self, colname='Position'):
        """
        For any player with dual positions (ex: "2B/SS") create 2 separate 
        rows for each position. Use *self.valid_lineup()* method to validate
        player is only used once in lineup and not used for both positions.
        
        Parameters
        ----------
        colname : str, default 'position'
            Name of column containing string position values for each player.
        """

        # copy df of only players with dual positions
        duals = self.df[self.df[colname].str.contains("/")].copy()
        
        # use first position in self.df
        self.df[colname] = self.df[colname].str.split("/").str[0]
    
        # use second position in duals
        duals[colname] = duals[colname].str.split("/").str[1]
        
        # append duals back to self.df
        self.df = self.df.append(duals)
    
    
    def point_min_filter(self, point_minimum=1):
        """
        Filters dataframe attribute (self.df) to only keep rows with projected 
        points greater than the 'point_minimum' parameter. Functoin also resets 
        and adds 'index' column to rows so it can be used to make sure all 
        players returned for lineups are unique.
        
        Parameters
        ----------
        point_minimum : int or float, default 1
            Minimum projected points value for rows in self.df
        """
        
        assert point_minimum > 0, ("'point_minimum' parameter must be > 0")
        
        # filter df
        self.df = self.df[self.df['ppg_projection'] > point_minimum].copy()
        
        # reset index after filtering
        self.df.reset_index(inplace=True, drop=True)
        # reset index again, but keep old index to use as player ID
        self.df.reset_index(inplace=True, drop=False)
        
        
    def position_bools(self, colname='Position'):
        """
        Adds columns to dataframe (self.df) attribute. Columns added have 1 or
        0 values based on the players (row) *colname* (position) parameter 
        value.
        
        Parameters
        ----------
        colname : str, default 'position'
            Name of column containing string position values for each player.
        """
        
        # add positional boolean columns to self.df attribute using apply and 
        # lambda functions based on position_col parameter
        
        self.df['P_bool'] = self.df[colname].apply(
                    lambda x: 1 if 'P' in x else 0
                )
        self.df['C_bool'] = self.df[colname].apply(
                    lambda x: 1 if 'C' in x else 0
                )
        self.df['1B_bool'] = self.df[colname].apply(
                    lambda x: 1 if '1B' in x else 0
                )
        self.df['2B_bool'] = self.df[colname].apply(
                    lambda x: 1 if '2B' in x else 0
                )
        self.df['3B_bool'] = self.df[colname].apply(
                    lambda x: 1 if '3B' in x else 0
                )
        self.df['SS_bool'] = self.df[colname].apply(
                    lambda x: 1 if 'SS' in x else 0
                )
        self.df['OF_bool'] = self.df[colname].apply(
                    lambda x: 1 if 'OF' in x else 0
                )    

    
    def format_data(self, 
                 point_minimum=1, 
                 int_cols=['ppg_projection', 'Salary'],
                 point_col='ppg_projection', 
                 position_col='Position'
                 ):
        """
        Formats dataframe attribute (self.df), adds object attributes, and adds 
        2 new attributes - self.hitter_df and self.pitcher_df - to be used in
        the 'create_lineup()' method.
        
        Parameters
        ----------
        point_minimum : int or float, default 1
            Minimum projected points value for rows in self.df
        int_columns : list, default ['ppg_projection', 'Salary']
            List of columns to turn into integer type.
        point_col : str, default 'ppg_projection'
            Column of projected points which is multiplied by 10 to eliminate
            decimal from number and make integer. Can be None if point column
            already is integer type. If not None, must be in the 'columns' 
            parameter list.
        position_col : str
            Name of column containing string position values for each player. 
        """
        
        # filter to only players with projections
        self.df = self.df[self.df[point_col] > point_minimum].copy()
        
        # turn any numeric columns using in optimizer to integer type
            # IF NOT INTEGER OPTIMIZER WILL NOT WORK
        self.int_cols(columns=int_cols, point_col=point_col)
        
        # account for dual position players
        self.dual_position(colname=position_col)

        # create positional dataframes
        self.position_bools(colname=position_col)
        
        # separate into hitter and pitcher dfs
        self.pitcher_df = self.df[self.df['Position'].str.contains("P")].copy()
        self.hitter_df = self.df[~self.df['Position'].str.contains("P")].copy()
        
        # reindex hitter and pitcher dfs so both start at 0
        # reset index after filtering
        self.pitcher_df.reset_index(inplace=True, drop=True)
        self.hitter_df.reset_index(inplace=True, drop=True)
        # reset index again, but keep old index to use as player ID
        self.pitcher_df.reset_index(inplace=True, drop=False)
        self.hitter_df.reset_index(inplace=True, drop=False)
        
        # Add dfs for dummy values - used in create_lineup() method
        self.hitter_name_dummies = pd.get_dummies(self.hitter_df['Name'])
        self.pitcher_opp_dummies = pd.get_dummies(self.hitter_df['opp'])
        self.pitcher_game_dummies = pd.get_dummies(self.pitcher_df['Game Info'])
        self.hitter_game_dummies = pd.get_dummies(self.hitter_df['Game Info'])
        self.hitter_team_dummies = pd.get_dummies(self.hitter_df['team'])
        
    
    def create_lineup(self, binary_lineups_pitchers, binary_lineups_hitters,
                      auto_stack=False, team_stack=None, stack_num=4,
                      variance=0):
        """
        Creates and runs optimization model. 
        
        Returns tuple of two lists - (player_indexes_list, binary_lineup_list).
        The player index list is just a list with each of the 8 player indexes 
        of the players selected by the optimizer. The binary lineup list is a 
        list of 1s and 0s with a 1 indicating a player is in the lineup and 0
        being the player is not.
        
        Parameters
        ----------
        binary_lineups_pitchers : list
            List of binary pitcher lineups with 1s and 0s for players who are 
            in or out of lineup. Used with 'variance' parameter to output 
            unique lineups.
        binary_lineups_hitters : list
            Same as *binary_lineups_pitchers* but with list representing 
            hitters.
        auto_stack : bool, default False
            If True, force lineup to include at least 4 players from same team.
            *team_stack* must be None if *auto_stack* is True.
        team_stack : str, default None
            Team abbreviation to stack at least *stack_num* number of
            players from. *auto_stack* must be False if this is not None.
        stack_num : int, default 4, must be between 0 and 5 (inclusive)
            Minimum number of players to be stacked from at least one team.
            Only applied if either *auto_stack* is True or *team_stack* is not
            None. If *auto_stack* is True, the team that is stacked will be 
            selected by algotithm. If *team_stack* is specified, this will be 
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
        """
        
        ########## ASSERTIONS AND CREATE MODEL, DECISION VARIABLES ########## 
        
        # Make sure lineups is list and variance is in
        assert type(binary_lineups_pitchers) == list, (
                    "'binary_lineups_pitchers' type needs to be list"
                )
        assert type(binary_lineups_hitters) == list, (
                    "'binary_lineups_hitters' type needs to be list"
                )        
        assert type(variance) == int, (
                    "'variance' type needs to be int"
                )
        assert variance >= 0 and variance <= 10, (
                    "'variance' must be ≥= 0 and <=10"
                )
        assert type(team_stack) == str or team_stack is None, (
                    "'team_stack' must be type string or None"
                )
        assert not auto_stack or not team_stack, (
                    "At least one of 'auto_stack' and 'team_stack' must be "
                    "False/None."
                )
        assert stack_num >= 0 and stack_num <= 5, (
                    "'stack_num' must be >= 0 and <= 5"
                )
        
        # create CP model
        model = cp_model.CpModel()
        
        # create list of decision variables for each position
        pitchers = [
                model.NewBoolVar(self.pitcher_df.iloc[i]['Name']) 
                for i in range(len(self.pitcher_df))
            ]
        hitters = [
                model.NewBoolVar(self.hitter_df.iloc[i]['Name']) 
                for i in range(len(self.hitter_df))
            ]
        
        
        ###################### ADD CONSTRAINTS ######################
        
        # NUMBER PLAYERS constraint
        # Also doubles as an All Different constraint b/c 10 unique players
         # need to be selected
        model.Add(
            sum([pitchers[i] for i in range(len(pitchers))]) == 2
        )
        model.Add(
            sum([hitters[i] for i in range(len(hitters))]) == 8
        )
        
        # Make sure each player name is used only once - account for dual 
        # position players
        for col in self.hitter_name_dummies.columns:
            model.Add(
                sum([hitters[i] * self.hitter_name_dummies.loc[i, col]
                for i in range(len(hitters))]) <= 1
            )
        
        # SALARY constraint
        model.Add(
                sum([pitchers[i] * self.pitcher_df.loc[i]['Salary'] 
                for i in range(len(pitchers))])
                + sum([hitters[i] * self.hitter_df.loc[i]['Salary']
                for i in range(len(hitters))])
                <= 50000
        )
        
        # POSITON constraints
        model.Add(
            sum([hitters[i] * self.hitter_df.loc[i]['C_bool'] 
            for i in range(len(hitters))]) == 1
        )
        model.Add(
            sum([hitters[i] * self.hitter_df.loc[i]['1B_bool'] 
            for i in range(len(hitters))]) == 1
        )
        model.Add(
            sum([hitters[i] * self.hitter_df.loc[i]['2B_bool'] 
            for i in range(len(hitters))]) == 1
        )
        model.Add(
            sum([hitters[i] * self.hitter_df.loc[i]['3B_bool'] 
            for i in range(len(hitters))]) == 1
        )
        model.Add(
            sum([hitters[i] * self.hitter_df.loc[i]['SS_bool'] 
            for i in range(len(hitters))]) == 1
        )
        model.Add(
            sum([hitters[i] * self.hitter_df.loc[i]['OF_bool'] 
            for i in range(len(hitters))]) == 3
        )
        
        
        # VARIANCE constraint
        # constraint is the number of players in each new lineup that must
          # be different from lineups already created. Use 8-variance b/c
          # just using variance would be the number of same players allowed
          # in each lineup where lower=more variance and higher=lower variance
          # but want to have higher number = higher variance and lower = 
          # less varianceS
        variance = 10 - variance
        
        # loops through each previous binary lineup ('binary_lineups' is list 
          # of lists of 1s and 0s of previous lineups) and says that the sum 
          # of the same players from lineup in 'binary_lineups' and 'players' 
          # cannot be more than the variance parameter int
        # if variance parameter was 0 then no lineup would be different 
            # because all 8 players could be the same, if it was 8 then no 
            # lineup could have the same player
        for k in range(len(binary_lineups_pitchers)):
            model.Add(
                sum([binary_lineups_pitchers[k][i] * pitchers[i] 
                    for i in range(len(pitchers))])
                + sum([binary_lineups_hitters[k][i] * hitters[i] 
                    for i in range(len(hitters))])
                <= variance
            )
            
        # PITCHER NOT FACING HITTERS constraint        
        for i in range(len(pitchers)):
            team = self.pitcher_df.loc[i, 'team']    
            model.Add(
                8 * pitchers[i] 
                + sum([hitters[k] * self.pitcher_opp_dummies.loc[k, team] 
                    for k in range(len(hitters))]) 
                <= 8
            )
            
        # PLAYERS FROM AT LEAST 2 GAMES constraint
        # sum pitcher and hitter from any gameand has to be less than 9
            # 10 players in each game, meaning at most 9 can be selected 
            # from a single game
        for game in self.pitcher_game_dummies.columns:
            model.Add(
                sum([pitchers[i] * self.pitcher_game_dummies.loc[i, game] 
                for i in range(len(pitchers))])
                + sum([hitters[i] * self.hitter_game_dummies.loc[i, game] 
                for i in range(len(hitters))])
                <= 9
            )
        
        # NO MORE THAN 5 HITTERS FROM ONE TEAM constraint
        for team in self.hitter_team_dummies.columns:
            model.Add(
                sum([hitters[i] * self.hitter_team_dummies.loc[i, team] 
                for i in range(len(hitters))]) <= 5
            )
            
        # STACKING constraint - auto stack, no input on specific teams to stack
        # make each unique team a decision variable
        # sum(players from single team) >= (stack_num * team_stack_var)
        # 
        # team_stack_var >= 1  # at least one stack
        # forces at least one team to have 4 players in lineup
        if auto_stack:
            teams = self.hitter_team_dummies.columns
            team_stack_var = [
                model.NewBoolVar(teams[i]) for i in range(len(teams))
            ]
            for t in range(len(teams)):
                model.Add(
                    sum([hitters[i] * self.hitter_team_dummies.loc[i, teams[t]]
                    for i in range(len(hitters))]) >= stack_num * team_stack_var[t] 
                )
            model.Add(sum([team_stack_var[i] for i in 
                           range(len(team_stack_var))]) >= 1)
            
        if team_stack:
            model.Add(
                    sum([hitters[i] * self.hitter_team_dummies.loc[i, team_stack]
                    for i in range(len(hitters))]) >= stack_num
                )
        
        ######################## SOLVE MODEL ########################
        
        model.Maximize(
            sum([pitchers[i] * self.pitcher_df.loc[i]['ppg_projection'] 
                for i in range(len(pitchers))])
            + sum([hitters[i] * self.hitter_df.loc[i]['ppg_projection'] 
                for i in range(len(hitters))])
        )
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # if not optimal, print the status and return None
        if status == cp_model.OPTIMAL:
            pass
        elif status == cp_model.FEASIBLE:
            print('Feasible')
            return None
        elif status == cp_model.INFEASIBLE:
            print('Infeasible')
            return None
        elif status == cp_model.MODEL_INVALID:
            print('Model Invalid')
            return None
        
        
        ################### OUTPUT LIST OF PLAYER INDEXES ###################
        
        # create list to hold player indexes 
        pitcher_indexes_list = []
        hitter_indexes_list = []
        
        # Also create binary list of 0 and 1 that is equal len as 'players'
        # decision variable list with 1 being player in lineup and 0 not
        binary_pitchers_list = []
        binary_hitters_list = []
                
        # loop through and append player indexes that solver has chosen and
        # also create binary list 
        for j in range(len(pitchers)):
            if solver.Value(pitchers[j]) == 1:
                pitcher_indexes_list.append(self.pitcher_df.iloc[j]['index'])
                binary_pitchers_list.append(1)
            else:
                binary_pitchers_list.append(0)
    
        for j in range(len(hitters)):
            if solver.Value(hitters[j]) == 1:
                hitter_indexes_list.append(self.hitter_df.iloc[j]['index'])
                binary_hitters_list.append(1)
            else:
                binary_hitters_list.append(0)
                
        # return tuple with player index list and binary list
            # player index list used to more easily read data by linking 
            # back to df
            # binary lineup list used for variance metric
        return ((pitcher_indexes_list, hitter_indexes_list), 
                (binary_pitchers_list, binary_hitters_list))
        
        
    def run_lineups(self, num_lineups, auto_stack=False, team_stack=None, 
                    stack_num=4, variance=0, print_progress=False):
        """
        Creates 'num_lineups' number of lineups using the self.create_lineup()
        method.
        
        Parameters
        ----------
        num_lineups : int
            Number of lineups to create.
        auto_stack : bool, default False
            If True, force lineup to include at least 4 players from same team.
            *team_stack* must be None if *auto_stack* is True.
        team_stack : str, default None
            Team abbreviation to stack at least *stack_num* number of
            players from. *auto_stack* must be False if this is not None.
        stack_num : int, default 4, must be between 0 and 5 (inclusive)
            If *team_stack* is not None, lineup must have at least this number
            of hitters from team *team_stack*. If this number is 0 then no 
            players from *team_stack* will be included in lineup.
        variance : int, default 0, Max=10, Min=0
            The min number of players that must be different in every lineup.
            Uses 'binary_lineups' parameter as list of already created lineups 
            and creates constraint that each lineup must have 'variance' 
            number of different players in each lineup. If parameter=0, every 
            lineup would be the same as no players would need to be different 
            from previous lineups. If parameter=10, then every lineup would 
            have to be completely unique with no single player being the same
            in any lineup.
        print_progress : bool, default False
            If True, print update after each lineup is created.
        
        Returns
        -------
        Creates attributes for 'pitcher_indexes', 'hitter_indexes', 
        'binary_lineups_pitchers', 'binary_lineups_hitters' if not already
        created. If attributes already created function will create new lineups
        building upon those already created.
        """
        
        def print_num_lineup(num_lineups, i):    
            if print_progress:
                print(i+1, "/", num_lineups, sep=" ")
            else:
                pass
            
        # format data
        self.format_data()
        
        # create empty lists that will end up as list of lists
        # will create empty lists if not already created
        if not hasattr(self, "pitcher_indexes"):
            self.pitcher_indexes = []
        if not hasattr(self, "hitter_indexes"):
            self.hitter_indexes = []
        if not hasattr(self, "binary_lineups_pitchers"):
            self.binary_lineups_pitchers = []
        if not hasattr(self, "binary_lineups_hitters"):
            self.binary_lineups_hitters = []
        
        ## Create lineups ##        
        for i in range(num_lineups):
            lineup = optimizer.create_lineup(self.binary_lineups_pitchers, 
                                             self.binary_lineups_hitters,
                                             auto_stack=auto_stack,
                                             team_stack=team_stack,
                                             stack_num=stack_num,
                                             variance=variance)
            
            self.pitcher_indexes.append(lineup[0][0])
            self.hitter_indexes.append(lineup[0][1])
            self.binary_lineups_pitchers.append(lineup[1][0])
            self.binary_lineups_hitters.append(lineup[1][1])
            
            print_num_lineup(num_lineups, i)
        
        print("COMPLETE")
        
        
    def csv_output(self, filename):
        """
        Outputs csv file of lineups using the self.pitcher_indexes and 
        self.hitter_indexes attributes and can be uploaded to Draftkings site.
        
        Parameters
        ----------
        filename : str
            Name of output csv file.
        """
        
        assert len(self.pitcher_indexes) == len(self.hitter_indexes), (
            "length of pitcher_indexes and hitter_indexes attributes must be "
            "equal."
        )
        assert type(self.pitcher_indexes) == list, (
            "pitcher_indexes attribute must be list."
        )
        assert type(self.hitter_indexes) == list, (
            "hitter_indexes attribute must be list."
        )
        
        uploadable_lineups = [
            ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        ]
        cols = [
            "Name + ID", "Position"
        ]

        for i in range(len(self.pitcher_indexes)):
            lineup_df = (
                    self.pitcher_df.loc[self.pitcher_indexes[i], cols]
                        .append(self.hitter_df.loc[self.hitter_indexes[i], cols])
                    )
            
            # single lineup list
            lineup_list = []
            
            lineup_list.extend(
                lineup_df[lineup_df['Position'].str.contains("P")]["Name + ID"].values)
            lineup_list.extend(
                lineup_df[lineup_df['Position'] == "C"]["Name + ID"].values)
            lineup_list.extend(
                lineup_df[lineup_df['Position'] == "1B"]["Name + ID"].values)
            lineup_list.extend(
                lineup_df[lineup_df['Position'] == "2B"]["Name + ID"].values)
            lineup_list.extend(
                lineup_df[lineup_df['Position'] == "3B"]["Name + ID"].values)
            lineup_list.extend(
                lineup_df[lineup_df['Position'] == "SS"]["Name + ID"].values)
            lineup_list.extend(
                lineup_df[lineup_df['Position'] == "OF"]["Name + ID"].values)
            
            uploadable_lineups.append(lineup_list)
            
        
        with open(filename, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(uploadable_lineups)
            
            csvFile.close()
        
        return 'Complete'
    

#%%
if __name__ == "__main__":
    
    ########## READ AND CLEAN DATA ##########
    date = '2021-09-04_MAIN'
    filename = str(input("csv filename: "))
    
    filename_dk = './data-dk/DKSalaries_' + date + '.csv'
    filename_proj = './data-projected/DFF_MLB_cheatsheet_' + date + '.csv'
    
    data_dk = pd.read_csv(filename_dk)
    data_proj = pd.read_csv(filename_proj)
    
    # Merge projections to DK salary data to make sure have correct salary AND 
    # positional information for each player
        # projections for some sources only have one position
    # Filter out players with no projected points later on
    
    data_proj['full_name'] = data_proj['first_name'] + ' ' + data_proj['last_name']
        
    merged = data_dk.merge(
            data_proj[[
                    'full_name', 'team', 'opp', 'ppg_projection', 
                    'value_projection'
                ]],
            how='left',  # keep all players and filter out nulls later on
            left_on='Name',
            right_on='full_name'
        )
    
    merged.head()
    # check that join worked and len of data_proj == not null values in merged
    merged['ppg_projection'].notnull().sum() == data_proj.shape[0]
    
    optimizer = OptimizerMLB(merged)
    
#%%    
    ########## CREATE LINEUPS ##########
    
    ### Create x # of auto stacked lineups
    
    # create object
    #optimizer = OptimizerMLB(merged)
    
    # run lineups
    num_lineups = int(input("number of lineups: "))
    optimizer.run_lineups(num_lineups, 
                          print_progress=True,
                          auto_stack=True,
                          stack_num=4,
                          variance=2)
        
    # Export to CSV #
    optimizer.csv_output(filename)
    
#%%    
    ### Create lineups with certain teams stacked
#    stack_teams = str(input("Teams to stack (comma separated):\n")).split(", ")
#    num_lineups_per_team = int(input("Number of lineups per team:\n"))
    
    #stack_teams_dict = dict(input("Dictionary - Team : Num lineups"))
    stack_teams_dict = {
            "CWS": 15,
            "CIN": 15,
            "LAA": 14,
            "LAD": 14,
            "SEA": 14,
            "MIL": 14,
            "ARI": 14
        }
    #optimizer = OptimizerMLB(merged)
    
    for team, num_lineups_team in stack_teams_dict.items():
        print("Team: " + team)
        optimizer.run_lineups(num_lineups_team,
                              team_stack=team,
                              stack_num=4,
                              variance=1,
                              print_progress=True
                )
        print("-"*10)
    
    # Export to CSV #
    optimizer.csv_output(filename)

#%%
##### NEXT STEPS #########
    
# Teams
    
tstr = "a:1, b:2"
d = {}
for s in tstr.split(","):
    s_split = s.split(":")
    d[s_split[0]] = int(s_split[1])


