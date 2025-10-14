import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestTensorBuilderInitialization:
    """Test tensor builder setup"""
    
    def test_tensor_builder_imports(self):
        """TensorBuilder class should be importable"""
        from data_processing.tensor_builder import TensorBuilder
        assert TensorBuilder is not None
    
    def test_tensor_builder_initialization(self):
        """Should initialize with correct dimensions"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        assert builder.roster_size == 64
        assert builder.player_features == 670
    
    def test_tensor_builder_has_required_methods(self):
        """TensorBuilder should have all required methods"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        required_methods = [
            'build_player_tensor',
            'build_roster_tensor',
            'build_game_tensor',
            'build_play_tensor'
        ]
        
        for method in required_methods:
            assert hasattr(builder, method)
            assert callable(getattr(builder, method))


class TestPlayerTensor:
    """Test 670-feature player tensor building"""
    
    def test_player_tensor_shape(self):
        """Player tensor should be exactly 670 features"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        mock_player = {
            'pfr_id': 'test001',
            'name': 'Test Player',
            'position': 'QB',
            'combine_stats': {},
            'college_stats': {},
            'nfl_career_stats': {},
            'seasonal_data': {}
        }
        
        tensor = builder.build_player_tensor(mock_player)
        
        assert tensor.shape == (670,)
        assert tensor.dtype == np.float32
    
    def test_player_tensor_components(self):
        """Player tensor should have all component sections"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Build with known data
        mock_player = {
            'pfr_id': 'BradTo00',
            'name': 'Tom Brady',
            'position': 'QB',
            'age': 45,
            'combine_stats': {'height': 76, 'weight': 225, 'forty_yard': 4.6},
            'college_stats': {'passing': {'yards': 11000, 'touchdowns': 100}},
            'nfl_career_stats': {'passing': {'yards': 89000, 'touchdowns': 649}},
            'seasonal_data': {'last_season': {}, 'best_season': {}, 'worst_season': {}, 'average_season': {}}
        }
        
        tensor = builder.build_player_tensor(mock_player)
        
        # Check some values were set (not all zeros)
        assert np.count_nonzero(tensor) > 0
        
        # Check height was captured (index 11 in combine section)
        assert tensor[11] == 76  # height in combine tensor
    
    def test_player_tensor_with_missing_data(self):
        """Player tensor should handle missing data gracefully"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Minimal player data
        minimal_player = {
            'pfr_id': 'unknown',
            'name': 'Unknown',
            'position': 'Unknown'
        }
        
        tensor = builder.build_player_tensor(minimal_player)
        
        # Should still be 670 features (filled with defaults/zeros)
        assert tensor.shape == (670,)
        assert tensor.dtype == np.float32


class TestRosterTensor:
    """Test 64-player roster tensor"""
    
    def test_roster_tensor_shape(self):
        """Roster tensor should flatten 64 players × 670 features"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Create 10 mock players
        players = [
            {
                'pfr_id': f'player{i:03d}',
                'name': f'Player {i}',
                'position': 'QB' if i == 0 else 'RB',
                'combine_stats': {},
                'college_stats': {},
                'nfl_career_stats': {},
                'seasonal_data': {}
            }
            for i in range(10)
        ]
        
        tensor = builder.build_roster_tensor(players)
        
        # Should be flattened 1D tensor
        expected_size = 64 * 670  # 64 player slots × 670 features
        assert tensor.shape == (expected_size,)
        assert tensor.dtype == np.float32
    
    def test_roster_tensor_null_players(self):
        """Roster should pad with null players (zeros) up to 64"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Only 5 players
        players = [
            {
                'pfr_id': f'player{i:03d}',
                'name': f'Player {i}',
                'position': 'QB',
                'combine_stats': {},
                'college_stats': {},
                'nfl_career_stats': {},
                'seasonal_data': {}
            }
            for i in range(5)
        ]
        
        tensor = builder.build_roster_tensor(players)
        
        # Should still be 64*670 (padded with zeros for remaining 59 slots)
        expected_size = 64 * 670
        assert tensor.shape == (expected_size,)
    
    def test_roster_tensor_max_64_players(self):
        """Roster should cap at 64 players maximum"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Try to create 100 players
        players = [
            {
                'pfr_id': f'player{i:03d}',
                'name': f'Player {i}',
                'position': 'QB',
                'combine_stats': {},
                'college_stats': {},
                'nfl_career_stats': {},
                'seasonal_data': {}
            }
            for i in range(100)
        ]
        
        tensor = builder.build_roster_tensor(players)
        
        # Should still be exactly 64*670 (extras ignored)
        expected_size = 64 * 670
        assert tensor.shape == (expected_size,)


class TestGameTensor:
    """Test game tensor with home/away rosters"""
    
    def test_game_tensor_shape(self):
        """Game tensor should combine home roster + away roster + game info"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Mock rosters
        home_roster = [
            {
                'pfr_id': f'home{i:03d}',
                'name': f'Home Player {i}',
                'position': 'QB',
                'combine_stats': {},
                'college_stats': {},
                'nfl_career_stats': {},
                'seasonal_data': {}
            }
            for i in range(15)
        ]
        
        away_roster = [
            {
                'pfr_id': f'away{i:03d}',
                'name': f'Away Player {i}',
                'position': 'QB',
                'combine_stats': {},
                'college_stats': {},
                'nfl_career_stats': {},
                'seasonal_data': {}
            }
            for i in range(15)
        ]
        
        game_info = {
            'temperature': 72,
            'dome': True,
            'week': 1,
            'season': 2024
        }
        
        tensor = builder.build_game_tensor(home_roster, away_roster, game_info)
        
        # Should be: home(64*670) + away(64*670) + game_info(50)
        expected_size = (64 * 670) + (64 * 670) + 50
        assert tensor.shape == (expected_size,)
        assert tensor.dtype == np.float32
    
    def test_game_tensor_game_info(self):
        """Game tensor should encode game info (temperature, dome, etc.)"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        home_roster = []
        away_roster = []
        
        game_info = {
            'temperature': 72,
            'dome': True,
            'week': 1,
            'season': 2024
        }
        
        tensor = builder.build_game_tensor(home_roster, away_roster, game_info)
        
        # Game info is at the end (last 50 features)
        game_info_start = (64 * 670) * 2
        
        # Temperature should be at game_info_start
        assert tensor[game_info_start] == 72
        
        # Dome flag should be 1.0 (true)
        assert tensor[game_info_start + 1] == 1.0


class TestPlayTensor:
    """Test play-level tensor combining game state and situation"""
    
    def test_play_tensor_shape(self):
        """Play tensor should be game tensor + play state"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Build a minimal game tensor
        game_tensor = np.zeros((64 * 670 * 2 + 50), dtype=np.float32)
        
        play_state = {
            'quarter': 1,
            'time_remaining': 900,
            'down': 1,
            'yards_to_go': 10,
            'yard_line': 25,
            'home_score': 0,
            'away_score': 0
        }
        
        play_tensor = builder.build_play_tensor(game_tensor, play_state)
        
        # Should be game_tensor size + play_state size (20)
        expected_size = len(game_tensor) + 20
        assert play_tensor.shape == (expected_size,)
        assert play_tensor.dtype == np.float32
    
    def test_play_state_components(self):
        """Play tensor should encode game situation"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        game_tensor = np.zeros((64 * 670 * 2 + 50), dtype=np.float32)
        
        play_state = {
            'quarter': 3,
            'time_remaining': 300,
            'down': 2,
            'yards_to_go': 5,
            'yard_line': 50,
            'home_score': 14,
            'away_score': 10
        }
        
        play_tensor = builder.build_play_tensor(game_tensor, play_state)
        
        # Play state is at the end (last 20 features)
        play_state_start = len(game_tensor)
        
        # Check values are set
        assert play_tensor[play_state_start] == 3  # quarter
        assert play_tensor[play_state_start + 1] == 300  # time_remaining


class TestTensorSafety:
    """Test error handling and edge cases"""
    
    def test_player_tensor_returns_670_on_error(self):
        """Should return 670-element tensor even on error"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Invalid player data
        invalid_player = None
        
        tensor = builder.build_player_tensor(invalid_player)
        
        # Should gracefully return zeros
        assert tensor.shape == (670,)
    
    def test_safe_float_conversion(self):
        """Should safely convert various data types to float"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        # Test internal safe_float method
        assert builder._safe_float("123.45") == 123.45
        assert builder._safe_float(None) == 0.0
        assert builder._safe_float("invalid") == 0.0
        assert builder._safe_float(100) == 100.0
    
    def test_position_to_num_mapping(self):
        """Should map positions to numbers consistently"""
        from data_processing.tensor_builder import TensorBuilder
        
        builder = TensorBuilder()
        
        assert builder._position_to_num('QB') == 1.0
        assert builder._position_to_num('RB') == 2.0
        assert builder._position_to_num('WR') == 3.0
        assert builder._position_to_num('TE') == 4.0
        assert builder._position_to_num('Unknown') == 0.0
