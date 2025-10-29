from typing import Dict, Any, List
from datetime import datetime

from scraping.game_scraper import GameScraper
from scraping.player_scraper import PlayerScraper
from data_processing.pipeline import DataPipeline
from database.operations import DatabaseOperations
from database.models import Player, PlayerSeason, Game, Season
from utils.logger import processing_logger as logger


class ChronologicalTrainingPipeline:
    """Process training data chronologically to maintain player state accuracy"""
    
    def __init__(self):
        self.game_scraper = GameScraper()
        self.player_scraper = PlayerScraper()
        self.pipeline = DataPipeline()
        self.player_state_cache = {}  # Cache player tensors
        
        logger.info("ChronologicalTrainingPipeline initialized")
    
    def process_season(
        self,
        season: int,
        start_week: int = 1,
        end_week: int = 18
    ) -> Dict[str, Any]:
        """
        Process entire season chronologically
        
        Args:
            season: Season year
            start_week: Starting week
            end_week: Ending week (18 for regular season)
            
        Returns:
            Processing results
        """
        try:
            logger.info(f"Processing season {season}, weeks {start_week}-{end_week}")
            
            results = {
                'season': season,
                'weeks_processed': 0,
                'total_games': 0,
                'total_plays': 0,
                'status': 'success'
            }
            
            # Initialize player states at season start
            self._initialize_season_player_states(season)
            
            # Process each week chronologically
            for week in range(start_week, end_week + 1):
                try:
                    week_result = self.process_week(season, week)
                    
                    results['weeks_processed'] += 1
                    results['total_games'] += week_result.get('games_processed', 0)
                    results['total_plays'] += week_result.get('plays_processed', 0)
                    
                    logger.info(f"Completed week {week}: {week_result['games_processed']} games")
                    
                except Exception as e:
                    logger.error(f"Failed to process week {week}: {str(e)}")
                    continue
            
            logger.info(f"Season {season} processing complete: {results['total_games']} games")
            return results
            
        except Exception as e:
            logger.error(f"Season processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            self.game_scraper.close()
    
    def process_week(self, season: int, week: int) -> Dict[str, Any]:
        """
        Process all games for a specific week
        
        Args:
            season: Season year
            week: Week number
            
        Returns:
            Week processing results
        """
        try:
            logger.info(f"Processing {season} Week {week}")
            
            # Get all games for this week
            game_urls = self.game_scraper.get_week_games(season, week)
            
            games_processed = 0
            plays_processed = 0
            
            for game_url in game_urls:
                try:
                    # Scrape game data
                    game_data = self.game_scraper.scrape_game_data(game_url, season, week)
                    
                    # Process game to database
                    db_game = self.pipeline.process_scraped_game(game_data)
                    
                    games_processed += 1
                    plays_processed += len(game_data.get('plays', []))
                    
                    # Update player states based on this game
                    self._update_player_states_from_game(db_game)
                    
                except Exception as e:
                    logger.error(f"Failed to process game {game_url}: {str(e)}")
                    continue
            
            return {
                'games_processed': games_processed,
                'plays_processed': plays_processed,
                'player_tensors_updated': len(self.player_state_cache)
            }
            
        except Exception as e:
            logger.error(f"Week processing failed: {str(e)}")
            return {
                'games_processed': 0,
                'plays_processed': 0,
                'error': str(e)
            }
    
    def _initialize_season_player_states(self, season: int):
        """Initialize player states at start of season with career data"""
        try:
            logger.info(f"Initializing player states for {season}")
            
            with DatabaseOperations() as db_ops:
                # Get all players who played in this season
                season_obj = db_ops.create_or_get_season(season)
                player_seasons = db_ops.db.query(PlayerSeason).filter_by(
                    season_id=season_obj.id
                ).all()
                
                for ps in player_seasons:
                    player = ps.player
                    
                    # Build initial tensor from career data
                    player_tensor = self.pipeline.tensor_builder.build_player_tensor({
                        'pfr_id': player.pfr_id,
                        'name': player.name,
                        'position': player.position,
                        'combine_stats': player.combine_stats or {},
                        'college_stats': player.college_stats or {},
                        'nfl_career_stats': {},  # Would get from previous seasons
                        'seasonal_data': {}
                    })
                    
                    self.player_state_cache[player.pfr_id] = player_tensor
                
                logger.info(f"Initialized {len(self.player_state_cache)} player states")
                
        except Exception as e:
            logger.error(f"Failed to initialize player states: {str(e)}")
    
    def _update_player_states_from_game(self, game: Game):
        """Update player tensors based on game results"""
        try:
            # In production, would:
            # 1. Extract player performances from game
            # 2. Update cumulative stats
            # 3. Rebuild player tensors with updated stats
            # 4. Store in cache for next game
            
            # Simplified for now
            pass
            
        except Exception as e:
            logger.error(f"Failed to update player states: {str(e)}")
    
    def get_training_data(
        self,
        seasons: List[int]
    ) -> Dict[str, Any]:
        """
        Generate training dataset from processed seasons
        
        Args:
            seasons: List of seasons to include
            
        Returns:
            Training data dictionary
        """
        try:
            logger.info(f"Generating training data for seasons: {seasons}")
            
            training_samples = []
            
            with DatabaseOperations() as db_ops:
                for season in seasons:
                    season_obj = db_ops.db.query(Season).filter_by(year=season).first()
                    if not season_obj:
                        continue
                    
                    # Get all games for season
                    games = db_ops.db.query(Game).filter_by(season_id=season_obj.id).all()
                    
                    for game in games:
                        # Build game tensor
                        game_tensor = self.pipeline.build_game_tensors(game.id)
                        
                        # Build target (outcome)
                        target = {
                            'home_score': game.home_score,
                            'away_score': game.away_score,
                            'winner': 'home' if game.home_score > game.away_score else 'away'
                        }
                        
                        training_samples.append({
                            'features': game_tensor,
                            'target': target
                        })
            
            return {
                'status': 'success',
                'num_samples': len(training_samples),
                'samples': training_samples
            }
            
        except Exception as e:
            logger.error(f"Failed to generate training data: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }