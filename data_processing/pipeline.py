import numpy as np
from typing import Dict, List, Optional

from data_processing.tensor_builder import TensorBuilder
from database.operations import DatabaseOperations
from database.models import Player, Team, Season, Game, Play, PlayerSeason
from utils.logger import processing_logger
from utils.error_handler import DataValidationError


class DataPipeline:
    """Process scraped data into database and tensors"""
    
    def __init__(self):
        """Initialize pipeline components"""
        self.tensor_builder = TensorBuilder()
        self.db_ops = DatabaseOperations()
        
        processing_logger.info("DataPipeline initialized")
    
    # ========================================================================
    # PLAYER PROCESSING
    # ========================================================================
    
    def process_scraped_player(self, scraped_player: Dict) -> Player:
        """
        Process scraped player data into database
        
        Args:
            scraped_player: Dictionary from PlayerScraper
            
        Returns:
            Database Player object
        """
        try:
            # Clean and validate data
            cleaned_data = self._clean_player_data(scraped_player)
            
            # Prepare database record
            player_data = {
                'name': cleaned_data['name'],
                'pfr_id': cleaned_data['player_id'],
                'position': cleaned_data['position'],
                'combine_stats': cleaned_data.get('combine_stats', {}),
                'college_stats': cleaned_data.get('college_stats', {})
            }
            
            # Save to database (upsert)
            db_player = self.db_ops.create_or_update_player(player_data)
            
            processing_logger.info(f"Processed player: {db_player.name}")
            return db_player
            
        except Exception as e:
            processing_logger.error(f"Failed to process player: {str(e)}")
            raise
    
    def _clean_player_data(self, player_data: Dict) -> Dict:
        """Clean and normalize player data"""
        cleaned = player_data.copy()
        
        # Trim whitespace
        if 'name' in cleaned:
            cleaned['name'] = str(cleaned['name']).strip()
        
        # Normalize position
        if 'position' in cleaned:
            cleaned['position'] = str(cleaned['position']).upper().strip()
        
        # Ensure required fields exist
        if 'combine_stats' not in cleaned:
            cleaned['combine_stats'] = {}
        if 'college_stats' not in cleaned:
            cleaned['college_stats'] = {}
        if 'nfl_career_stats' not in cleaned:
            cleaned['nfl_career_stats'] = {}
        
        return cleaned
    
    def _validate_position(self, position: str) -> bool:
        """Validate player position"""
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P']
        return position.upper() in valid_positions
    
    # ========================================================================
    # GAME PROCESSING
    # ========================================================================
    
    def process_scraped_game(self, scraped_game: Dict) -> Game:
        """
        Process scraped game data into database
        
        Args:
            scraped_game: Dictionary from GameScraper
            
        Returns:
            Database Game object
        """
        try:
            # Get or create season
            season = self.db_ops.create_or_get_season(scraped_game['season'])
            
            # Get or create teams
            home_team = self.db_ops.create_or_update_team({
                'name': scraped_game['home_team'],
                'abbreviation': scraped_game['home_team'],
                'pfr_id': scraped_game['home_team']
            })
            
            away_team = self.db_ops.create_or_update_team({
                'name': scraped_game['away_team'],
                'abbreviation': scraped_game['away_team'],
                'pfr_id': scraped_game['away_team']
            })
            
            # Create game record
            game_data = {
                'season_id': season.id,
                'week': scraped_game.get('week', 0),
                'home_team_id': home_team.id,
                'away_team_id': away_team.id,
                'home_score': scraped_game.get('home_score', 0),
                'away_score': scraped_game.get('away_score', 0),
                'pfr_game_id': scraped_game['game_id'],
                'is_complete': True
            }
            
            db_game = self.db_ops.create_or_update_game(game_data)
            
            # Process plays
            if 'plays' in scraped_game and scraped_game['plays']:
                self._process_plays(db_game.id, scraped_game['plays'])
            
            processing_logger.info(f"Processed game: {scraped_game['game_id']}")
            return db_game
            
        except Exception as e:
            processing_logger.error(f"Failed to process game: {str(e)}")
            raise
    
    def _process_plays(self, game_id: int, plays: List[Dict]):
        """Process play-by-play data"""
        try:
            plays_data = []
            
            for i, play in enumerate(plays):
                # Build play tensor
                play_state = {
                    'quarter': play.get('quarter', 1),
                    'down': play.get('down', 1),
                    'yards_to_go': play.get('yards_to_go', 10),
                    'yard_line': play.get('yard_line', 50)
                }
                
                # Simple play tensor (just state for now, game tensor would be added later)
                play_tensor = self.tensor_builder._build_play_state_tensor(play_state)
                
                play_data = {
                    'game_id': game_id,
                    'play_number': i + 1,
                    'quarter': play.get('quarter', 1),
                    'play_type': play.get('play_type', 'unknown'),
                    'yards_gained': play.get('yards_gained', 0),
                    'touchdown': play.get('touchdown', False),
                    'field_goal': play.get('field_goal', False),
                    'play_state_tensor': play_tensor.tolist()
                }
                
                plays_data.append(play_data)
            
            # Bulk insert
            if plays_data:
                self.db_ops.bulk_create_plays(plays_data)
                processing_logger.info(f"Processed {len(plays_data)} plays")
                
        except Exception as e:
            processing_logger.error(f"Failed to process plays: {str(e)}")
    
    # ========================================================================
    # ROSTER & GAME TENSOR BUILDING
    # ========================================================================
    
    def process_team_roster(self, team_id: int, season_id: int) -> np.ndarray:
        """
        Build roster tensor for a team in a season
        
        Args:
            team_id: Team database ID
            season_id: Season database ID
            
        Returns:
            Flattened roster tensor (64*670,)
        """
        try:
            # Get players from database
            players = self.db_ops.get_players_by_team_season(team_id, season_id)
            
            # Convert to player data dicts for tensor builder
            players_data = []
            for player in players:
                player_dict = {
                    'pfr_id': player.pfr_id,
                    'name': player.name,
                    'position': player.position,
                    'combine_stats': player.combine_stats or {},
                    'college_stats': player.college_stats or {},
                    'nfl_career_stats': {},
                    'seasonal_data': {}
                }
                players_data.append(player_dict)
            
            # Build tensor
            roster_tensor = self.tensor_builder.build_roster_tensor(players_data)
            
            processing_logger.info(f"Built roster tensor with {len(players)} players")
            return roster_tensor
            
        except Exception as e:
            processing_logger.error(f"Failed to build roster tensor: {str(e)}")
            # Return zeros on error
            return np.zeros(64 * 670, dtype=np.float32)
    
    def build_game_tensors(self, game_id: int) -> np.ndarray:
        """
        Build complete game tensor from database
        
        Args:
            game_id: Game database ID
            
        Returns:
            Complete game tensor
        """
        try:
            # Get game from database
            game = self.db_ops.db.query(Game).filter_by(id=game_id).first()
            
            if not game:
                raise DataValidationError(f"Game {game_id} not found")
            
            # Build home and away roster tensors
            home_roster_tensor = self.process_team_roster(game.home_team_id, game.season_id)
            away_roster_tensor = self.process_team_roster(game.away_team_id, game.season_id)
            
            # Game info
            game_info = {
                'week': game.week,
                'season': game.season.year,
                'home_score': game.home_score or 0,
                'away_score': game.away_score or 0
            }
            
            game_info_tensor = self.tensor_builder._build_game_info_tensor(game_info)
            
            # Combine into game tensor
            game_tensor = np.concatenate([
                home_roster_tensor,
                away_roster_tensor,
                game_info_tensor
            ])
            
            processing_logger.info(f"Built game tensor for game {game_id}")
            return game_tensor
            
        except Exception as e:
            processing_logger.error(f"Failed to build game tensor: {str(e)}")
            raise
    
    def close(self):
        """Clean up resources"""
        self.db_ops.db.close()
