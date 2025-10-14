import numpy as np
from typing import Dict, List
from utils.logger import processing_logger


class TensorBuilder:
    """Build neural network tensors for NFL player and game data"""
    
    def __init__(self):
        """Initialize tensor dimensions"""
        self.roster_size = 64  # Per specification
        self.player_features = 670  # Per specification
    
    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================
    
    def build_player_tensor(self, player_data: Dict) -> np.ndarray:
        """
        Build 670-feature tensor for a single player
        
        Structure (670 total):
        - RosterInfo: 9 features
        - Combine: 13 features
        - CollegeCareer: 64 features
        - NFLCareer: 116 features
        - LastSeason: 117 features
        - WorstSeason: 117 features
        - BestSeason: 117 features
        - AvgSeason: 116 features
        
        Args:
            player_data: Dictionary with player information
            
        Returns:
            numpy array of shape (670,) with dtype float32
        """
        try:
            tensor = np.zeros(self.player_features, dtype=np.float32)
            idx = 0
            
            # 1. RosterInfo (9)
            roster_info = self._build_roster_info_tensor(player_data)
            tensor[idx:idx+9] = roster_info
            idx += 9
            
            # 2. Combine (13)
            combine = self._build_combine_tensor(player_data.get('combine_stats', {}))
            tensor[idx:idx+13] = combine
            idx += 13
            
            # 3. CollegeCareer (64)
            college = self._build_college_tensor(player_data.get('college_stats', {}))
            tensor[idx:idx+64] = college
            idx += 64
            
            # 4. NFLCareer (116)
            nfl_career = self._build_nfl_career_tensor(player_data.get('nfl_career_stats', {}))
            tensor[idx:idx+116] = nfl_career
            idx += 116
            
            # 5. LastSeason (117)
            last_season = self._build_season_tensor(
                player_data.get('seasonal_data', {}).get('last_season', {})
            )
            tensor[idx:idx+117] = last_season
            idx += 117
            
            # 6. WorstSeason (117)
            worst_season = self._build_season_tensor(
                player_data.get('seasonal_data', {}).get('worst_season', {})
            )
            tensor[idx:idx+117] = worst_season
            idx += 117
            
            # 7. BestSeason (117)
            best_season = self._build_season_tensor(
                player_data.get('seasonal_data', {}).get('best_season', {})
            )
            tensor[idx:idx+117] = best_season
            idx += 117
            
            # 8. AvgSeason (116)
            avg_season = self._build_season_tensor(
                player_data.get('seasonal_data', {}).get('average_season', {}),
                exclude_team=True
            )
            tensor[idx:idx+116] = avg_season
            
            return tensor
            
        except Exception as e:
            processing_logger.error(f"Failed to build player tensor: {str(e)}")
            return np.zeros(self.player_features, dtype=np.float32)
    
    def build_roster_tensor(self, players_data: List[Dict]) -> np.ndarray:
        """
        Build roster tensor from up to 64 players
        
        Args:
            players_data: List of player dictionaries
            
        Returns:
            Flattened numpy array of shape (64*670,) = (42880,)
        """
        try:
            roster_tensor = np.zeros((self.roster_size, self.player_features), dtype=np.float32)
            
            # Fill with actual players (up to 64)
            for i, player_data in enumerate(players_data[:self.roster_size]):
                roster_tensor[i] = self.build_player_tensor(player_data)
            
            # Remaining slots stay as zeros (null players)
            actual_count = min(len(players_data), self.roster_size)
            processing_logger.info(f"Built roster tensor with {actual_count} players")
            
            return roster_tensor.flatten()
            
        except Exception as e:
            processing_logger.error(f"Failed to build roster tensor: {str(e)}")
            return np.zeros(self.roster_size * self.player_features, dtype=np.float32)
    
    def build_game_tensor(self, home_roster: List[Dict], away_roster: List[Dict],
                         game_info: Dict) -> np.ndarray:
        """
        Build complete game tensor
        
        Args:
            home_roster: List of home team player data
            away_roster: List of away team player data
            game_info: Game context (temperature, dome, week, etc.)
            
        Returns:
            Concatenated tensor of shape (64*670*2 + 50,)
        """
        try:
            home_tensor = self.build_roster_tensor(home_roster)
            away_tensor = self.build_roster_tensor(away_roster)
            game_info_tensor = self._build_game_info_tensor(game_info)
            
            game_tensor = np.concatenate([home_tensor, away_tensor, game_info_tensor])
            
            processing_logger.info(f"Built game tensor with shape {game_tensor.shape}")
            return game_tensor
            
        except Exception as e:
            processing_logger.error(f"Failed to build game tensor: {str(e)}")
            total_size = (2 * self.roster_size * self.player_features) + 50
            return np.zeros(total_size, dtype=np.float32)
    
    def build_play_tensor(self, game_tensor: np.ndarray, play_state: Dict) -> np.ndarray:
        """
        Build play tensor combining game state and situation
        
        Args:
            game_tensor: Full game tensor from build_game_tensor
            play_state: Current play situation (down, quarter, etc.)
            
        Returns:
            Concatenated tensor of shape (game_tensor + 20,)
        """
        try:
            play_state_tensor = self._build_play_state_tensor(play_state)
            play_tensor = np.concatenate([game_tensor, play_state_tensor])
            return play_tensor
            
        except Exception as e:
            processing_logger.error(f"Failed to build play tensor: {str(e)}")
            return np.zeros(len(game_tensor) + 20, dtype=np.float32)
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _build_roster_info_tensor(self, player_data: Dict) -> np.ndarray:
        """Build RosterInfo section (9 features)"""
        roster_info = np.zeros(9, dtype=np.float32)
        
        roster_info[0] = abs(hash(str(player_data.get('pfr_id', 'unknown')))) % 1000000
        roster_info[1] = self._position_to_num(player_data.get('position', ''))
        roster_info[2] = self._safe_float(player_data.get('roster_tier', 1))
        
        draft_info = player_data.get('draft_info', {})
        if isinstance(draft_info, dict):
            roster_info[3] = abs(hash(str(draft_info.get('team', '')))) % 100
            roster_info[4] = self._safe_float(draft_info.get('year', 0))
            roster_info[5] = self._safe_float(draft_info.get('pick', 0))
        
        roster_info[6] = self._safe_float(player_data.get('roster_season', 2024))
        roster_info[7] = abs(hash(str(player_data.get('current_team', '')))) % 100
        roster_info[8] = self._safe_float(player_data.get('age', 25))
        
        return roster_info
    
    def _build_combine_tensor(self, combine_stats: Dict) -> np.ndarray:
        """Build Combine section (13 features)"""
        combine = np.zeros(13, dtype=np.float32)
        
        combine[0] = self._safe_float(combine_stats.get('year', 0))
        combine[1] = self._position_to_num(combine_stats.get('position', ''))
        combine[2] = self._safe_float(combine_stats.get('height', 0))
        combine[3] = self._safe_float(combine_stats.get('weight', 0))
        combine[4] = self._safe_float(combine_stats.get('forty_yard', 0))
        combine[5] = self._safe_float(combine_stats.get('bench', 0))
        combine[6] = self._safe_float(combine_stats.get('broad_jump', 0))
        combine[7] = self._safe_float(combine_stats.get('shuttle', 0))
        combine[8] = self._safe_float(combine_stats.get('three_cone', 0))
        combine[9] = self._safe_float(combine_stats.get('vertical', 0))
        
        return combine
    
    def _build_college_tensor(self, college_stats: Dict) -> np.ndarray:
        """Build CollegeCareer section (64 features)"""
        college = np.zeros(64, dtype=np.float32)
        idx = 0
        
        # Basic info (5)
        college[idx] = self._safe_float(college_stats.get('seasons', 0))
        college[idx+1] = self._safe_float(college_stats.get('first_season_school', 0))
        college[idx+2] = self._safe_float(college_stats.get('last_season_school', 0))
        college[idx+3] = self._safe_float(college_stats.get('first_school_seasons', 0))
        college[idx+4] = self._safe_float(college_stats.get('last_school_seasons', 0))
        idx += 5
        
        # Passing (5)
        passing = college_stats.get('passing', {})
        college[idx:idx+5] = [
            self._safe_float(passing.get('completions', 0)),
            self._safe_float(passing.get('attempts', 0)),
            self._safe_float(passing.get('yards', 0)),
            self._safe_float(passing.get('touchdowns', 0)),
            self._safe_float(passing.get('interceptions', 0))
        ]
        idx += 5
        
        # Rushing (3)
        rushing = college_stats.get('rushing', {})
        college[idx:idx+3] = [
            self._safe_float(rushing.get('attempts', 0)),
            self._safe_float(rushing.get('yards', 0)),
            self._safe_float(rushing.get('touchdowns', 0))
        ]
        idx += 3
        
        # Receiving (3)
        receiving = college_stats.get('receiving', {})
        college[idx:idx+3] = [
            self._safe_float(receiving.get('receptions', 0)),
            self._safe_float(receiving.get('yards', 0)),
            self._safe_float(receiving.get('touchdowns', 0))
        ]
        idx += 3
        
        # Defense (11)
        defense = college_stats.get('defense', {})
        college[idx:idx+11] = [
            self._safe_float(defense.get('tackles', 0)),
            self._safe_float(defense.get('sacks', 0)),
            self._safe_float(defense.get('interceptions', 0)),
            self._safe_float(defense.get('int_yards', 0)),
            self._safe_float(defense.get('int_td', 0)),
            self._safe_float(defense.get('pd', 0)),
            self._safe_float(defense.get('fr', 0)),
            self._safe_float(defense.get('fr_yards', 0)),
            self._safe_float(defense.get('ff', 0)),
            self._safe_float(defense.get('tfl', 0)),
            self._safe_float(defense.get('qb_hits', 0))
        ]
        idx += 11
        
        # Kicking (6)
        kicking = college_stats.get('kicking', {})
        college[idx:idx+6] = [
            self._safe_float(kicking.get('fgm', 0)),
            self._safe_float(kicking.get('fga', 0)),
            self._safe_float(kicking.get('xpm', 0)),
            self._safe_float(kicking.get('xpa', 0)),
            self._safe_float(kicking.get('punts', 0)),
            self._safe_float(kicking.get('punt_yards', 0))
        ]
        idx += 6
        
        # Team stats (15)
        team_stats = college_stats.get('team', {})
        college[idx:idx+15] = [
            self._safe_float(team_stats.get('pass_completions', 0)),
            self._safe_float(team_stats.get('pass_attempts', 0)),
            self._safe_float(team_stats.get('pass_yards', 0)),
            self._safe_float(team_stats.get('pass_td', 0)),
            self._safe_float(team_stats.get('rush_attempts', 0)),
            self._safe_float(team_stats.get('rush_yards', 0)),
            self._safe_float(team_stats.get('rush_td', 0)),
            self._safe_float(team_stats.get('total_plays', 0)),
            self._safe_float(team_stats.get('pass_1d', 0)),
            self._safe_float(team_stats.get('rush_1d', 0)),
            self._safe_float(team_stats.get('pen_1d', 0)),
            self._safe_float(team_stats.get('penalties', 0)),
            self._safe_float(team_stats.get('pen_yards', 0)),
            self._safe_float(team_stats.get('fumbles', 0)),
            self._safe_float(team_stats.get('interceptions', 0))
        ]
        idx += 15
        
        # Opp stats (15)
        opp_stats = college_stats.get('opp', {})
        college[idx:idx+15] = [
            self._safe_float(opp_stats.get('pass_completions', 0)),
            self._safe_float(opp_stats.get('pass_attempts', 0)),
            self._safe_float(opp_stats.get('pass_yards', 0)),
            self._safe_float(opp_stats.get('pass_td', 0)),
            self._safe_float(opp_stats.get('rush_attempts', 0)),
            self._safe_float(opp_stats.get('rush_yards', 0)),
            self._safe_float(opp_stats.get('rush_td', 0)),
            self._safe_float(opp_stats.get('total_plays', 0)),
            self._safe_float(opp_stats.get('pass_1d', 0)),
            self._safe_float(opp_stats.get('rush_1d', 0)),
            self._safe_float(opp_stats.get('pen_1d', 0)),
            self._safe_float(opp_stats.get('penalties', 0)),
            self._safe_float(opp_stats.get('pen_yards', 0)),
            self._safe_float(opp_stats.get('fumbles', 0)),
            self._safe_float(opp_stats.get('interceptions', 0))
        ]
        
        return college
    
    def _build_nfl_career_tensor(self, nfl_stats: Dict) -> np.ndarray:
        """Build NFLCareer section (116 features)"""
        nfl = np.zeros(116, dtype=np.float32)
        idx = 0
        
        # Basic info (3)
        nfl[idx] = self._safe_float(nfl_stats.get('seasons_played', 0))
        nfl[idx+1] = self._safe_float(nfl_stats.get('games_played', 0))
        nfl[idx+2] = self._safe_float(nfl_stats.get('games_started', 0))
        idx += 3
        
        # Passing (11)
        passing = nfl_stats.get('passing', {})
        nfl[idx:idx+11] = [
            self._safe_float(passing.get('record', 0)),
            self._safe_float(passing.get('completions', 0)),
            self._safe_float(passing.get('attempts', 0)),
            self._safe_float(passing.get('yards', 0)),
            self._safe_float(passing.get('touchdowns', 0)),
            self._safe_float(passing.get('interceptions', 0)),
            self._safe_float(passing.get('first_downs', 0)),
            self._safe_float(passing.get('longest', 0)),
            self._safe_float(passing.get('sacked', 0)),
            self._safe_float(passing.get('4qc', 0)),
            self._safe_float(passing.get('gwd', 0))
        ]
        idx += 11
        
        # Rushing (5)
        rushing = nfl_stats.get('rushing', {})
        nfl[idx:idx+5] = [
            self._safe_float(rushing.get('attempts', 0)),
            self._safe_float(rushing.get('yards', 0)),
            self._safe_float(rushing.get('touchdowns', 0)),
            self._safe_float(rushing.get('first_downs', 0)),
            self._safe_float(rushing.get('longest', 0))
        ]
        idx += 5
        
        # Receiving (6)
        receiving = nfl_stats.get('receiving', {})
        nfl[idx:idx+6] = [
            self._safe_float(receiving.get('targets', 0)),
            self._safe_float(receiving.get('receptions', 0)),
            self._safe_float(receiving.get('yards', 0)),
            self._safe_float(receiving.get('touchdowns', 0)),
            self._safe_float(receiving.get('first_downs', 0)),
            self._safe_float(receiving.get('longest', 0))
        ]
        idx += 6
        
        # Defense (15)
        defense = nfl_stats.get('defense', {})
        nfl[idx:idx+15] = [
            self._safe_float(defense.get('interceptions', 0)),
            self._safe_float(defense.get('int_yards', 0)),
            self._safe_float(defense.get('int_td', 0)),
            self._safe_float(defense.get('int_longest', 0)),
            self._safe_float(defense.get('pd', 0)),
            self._safe_float(defense.get('ff', 0)),
            self._safe_float(defense.get('fumbles', 0)),
            self._safe_float(defense.get('fr', 0)),
            self._safe_float(defense.get('fr_yards', 0)),
            self._safe_float(defense.get('fr_td', 0)),
            self._safe_float(defense.get('sacks', 0)),
            self._safe_float(defense.get('solo_tackles', 0)),
            self._safe_float(defense.get('assisted_tackles', 0)),
            self._safe_float(defense.get('tfl', 0)),
            self._safe_float(defense.get('qb_hits', 0))
        ]
        idx += 15
        
        # Kicking (15)
        kicking = nfl_stats.get('kicking', {})
        nfl[idx:idx+15] = [
            self._safe_float(kicking.get('fga_0_19', 0)),
            self._safe_float(kicking.get('fgm_0_19', 0)),
            self._safe_float(kicking.get('fga_20_29', 0)),
            self._safe_float(kicking.get('fgm_20_29', 0)),
            self._safe_float(kicking.get('fga_30_39', 0)),
            self._safe_float(kicking.get('fgm_30_39', 0)),
            self._safe_float(kicking.get('fga_40_49', 0)),
            self._safe_float(kicking.get('fgm_40_49', 0)),
            self._safe_float(kicking.get('fga_50_plus', 0)),
            self._safe_float(kicking.get('fgm_50_plus', 0)),
            self._safe_float(kicking.get('longest', 0)),
            self._safe_float(kicking.get('xpa', 0)),
            self._safe_float(kicking.get('xpm', 0)),
            self._safe_float(kicking.get('punts', 0)),
            self._safe_float(kicking.get('punt_yards', 0))
        ]
        idx += 15
        
        # Team Performance (30)
        team_perf = nfl_stats.get('team_performance', {})
        nfl[idx:idx+30] = [
            self._safe_float(team_perf.get('off_points', 0)),
            self._safe_float(team_perf.get('off_yards', 0)),
            self._safe_float(team_perf.get('off_plays', 0)),
            self._safe_float(team_perf.get('off_turnovers', 0)),
            self._safe_float(team_perf.get('off_fumbles', 0)),
            self._safe_float(team_perf.get('off_1d', 0)),
            self._safe_float(team_perf.get('pass_cmp', 0)),
            self._safe_float(team_perf.get('pass_att', 0)),
            self._safe_float(team_perf.get('pass_yds', 0)),
            self._safe_float(team_perf.get('pass_td', 0)),
            self._safe_float(team_perf.get('rush_att', 0)),
            self._safe_float(team_perf.get('rush_yds', 0)),
            self._safe_float(team_perf.get('rush_td', 0)),
            self._safe_float(team_perf.get('penalties', 0)),
            self._safe_float(team_perf.get('pen_yards', 0)),
            self._safe_float(team_perf.get('def_points', 0)),
            self._safe_float(team_perf.get('def_yards', 0)),
            self._safe_float(team_perf.get('def_plays', 0)),
            self._safe_float(team_perf.get('def_turnovers', 0)),
            self._safe_float(team_perf.get('def_fumbles', 0)),
            self._safe_float(team_perf.get('def_1d', 0)),
            self._safe_float(team_perf.get('def_pass_cmp', 0)),
            self._safe_float(team_perf.get('def_pass_att', 0)),
            self._safe_float(team_perf.get('def_pass_yds', 0)),
            self._safe_float(team_perf.get('def_pass_td', 0)),
            self._safe_float(team_perf.get('def_rush_att', 0)),
            self._safe_float(team_perf.get('def_rush_yds', 0)),
            self._safe_float(team_perf.get('def_rush_td', 0)),
            self._safe_float(team_perf.get('opp_penalties', 0)),
            self._safe_float(team_perf.get('opp_pen_yards', 0))
        ]
        
        return nfl
    
    def _build_season_tensor(self, season_stats: Dict, exclude_team: bool = False) -> np.ndarray:
        """Build seasonal tensor (117 or 116 features)"""
        size = 116 if exclude_team else 117
        season = np.zeros(size, dtype=np.float32)
        idx = 0
        
        if not exclude_team:
            season[idx] = abs(hash(str(season_stats.get('team', '')))) % 100
            idx += 1
        
        # Games played/started (2)
        season[idx] = self._safe_float(season_stats.get('games_played', 0))
        season[idx+1] = self._safe_float(season_stats.get('games_started', 0))
        idx += 2
        
        # Individual stats - similar to NFL career but for single season
        # Fill remaining with zeros (placeholder for full implementation)
        remaining_features = size - idx
        season[idx:idx+remaining_features] = 0
        
        return season
    
    def _build_game_info_tensor(self, game_info: Dict) -> np.ndarray:
        """Build game context tensor (50 features)"""
        info_tensor = np.zeros(50, dtype=np.float32)
        
        # Basic game info (5)
        info_tensor[0] = self._safe_float(game_info.get('temperature', 70))
        info_tensor[1] = 1.0 if game_info.get('dome', False) else 0.0
        info_tensor[2] = self._safe_float(game_info.get('wind_speed', 0))
        info_tensor[3] = self._safe_float(game_info.get('week', 0))
        info_tensor[4] = self._safe_float(game_info.get('season', 2024))
        
        # Team records (4)
        info_tensor[5] = self._safe_float(game_info.get('home_wins', 0))
        info_tensor[6] = self._safe_float(game_info.get('home_losses', 0))
        info_tensor[7] = self._safe_float(game_info.get('away_wins', 0))
        info_tensor[8] = self._safe_float(game_info.get('away_losses', 0))
        
        # Playoff flag
        info_tensor[9] = 1.0 if game_info.get('playoff', False) else 0.0
        
        # Weather conditions (5 binary flags)
        weather = str(game_info.get('weather', '')).lower()
        info_tensor[10] = 1.0 if 'clear' in weather else 0.0
        info_tensor[11] = 1.0 if 'cloudy' in weather else 0.0
        info_tensor[12] = 1.0 if 'rain' in weather else 0.0
        info_tensor[13] = 1.0 if 'snow' in weather else 0.0
        info_tensor[14] = 1.0 if 'fog' in weather else 0.0
        
        # Surface type (2)
        surface = str(game_info.get('surface', '')).lower()
        info_tensor[15] = 1.0 if 'grass' in surface else 0.0
        info_tensor[16] = 1.0 if 'turf' in surface else 0.0
        
        # Time of day
        info_tensor[17] = self._safe_float(game_info.get('start_time_hour', 13))
        
        # Remaining features reserved for future use
        
        return info_tensor
    
    def _build_play_state_tensor(self, play_state: Dict) -> np.ndarray:
        """Build play situation tensor (20 features)"""
        state_tensor = np.zeros(20, dtype=np.float32)
        
        state_tensor[0] = self._safe_float(play_state.get('quarter', 1))
        state_tensor[1] = self._safe_float(play_state.get('time_remaining', 900))
        state_tensor[2] = self._safe_float(play_state.get('down', 1))
        state_tensor[3] = self._safe_float(play_state.get('yards_to_go', 10))
        state_tensor[4] = self._safe_float(play_state.get('yard_line', 50))
        state_tensor[5] = self._safe_float(play_state.get('home_score', 0))
        state_tensor[6] = self._safe_float(play_state.get('away_score', 0))
        state_tensor[7] = self._safe_float(play_state.get('possession', 0))  # 0=away, 1=home
        
        # Red zone flag (within 20 yards of endzone)
        yard_line = self._safe_float(play_state.get('yard_line', 50))
        state_tensor[8] = 1.0 if yard_line <= 20 or yard_line >= 80 else 0.0
        
        # Goal to go flag
        yards_to_go = self._safe_float(play_state.get('yards_to_go', 10))
        yard_line_signed = yard_line if play_state.get('possession', 0) == 1 else 100 - yard_line
        state_tensor[9] = 1.0 if yards_to_go >= yard_line_signed else 0.0
        
        # Score differential
        state_tensor[10] = self._safe_float(play_state.get('home_score', 0)) - self._safe_float(play_state.get('away_score', 0))
        
        # Two minute warning
        state_tensor[11] = 1.0 if play_state.get('two_minute_warning', False) else 0.0
        
        # Timeouts
        state_tensor[12] = self._safe_float(play_state.get('timeouts_home', 3))
        state_tensor[13] = self._safe_float(play_state.get('timeouts_away', 3))
        
        # Remaining features reserved for future use
        
        return state_tensor
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert value to float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _position_to_num(self, position: str) -> float:
        """Convert position string to numeric code"""
        position_map = {
            'QB': 1.0, 'RB': 2.0, 'WR': 3.0, 'TE': 4.0,
            'OL': 5.0, 'DL': 6.0, 'LB': 7.0, 'DB': 8.0,
            'K': 9.0, 'P': 10.0
        }
        return position_map.get(str(position).upper(), 0.0)

