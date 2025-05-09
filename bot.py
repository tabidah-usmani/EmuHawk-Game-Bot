import csv
import os
from command import Command
import numpy as np
from buttons import Buttons
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

class Bot:
    def __init__(self):
        self.my_command = Command()
        self.buttn = Buttons()
        try:
            # Load the scaler and multi-label model
            self.scaler = joblib.load('scaler.pkl')
            self.model = joblib.load('game_bot_model.pkl')
        except FileNotFoundError as e:
            print(f"Error loading model or scaler: {e}. Using default behavior.")
            # Initialize with default scaler and model to prevent crashes
            self.scaler = StandardScaler()
            self.model = None
        
        # Define expected feature names in the same order as training
        self.feature_names = ['player_x', 'opponent_x', 'player_y', 'opponent_y', 'player_health',
                            'opponent_health', 'player_is_jumping', 'player_is_crouching',
                            'player_in_move', 'opponent_in_move', 'player_move_id',
                            'opponent_move_id', 'diff', 'game_time']

    def fight(self, current_game_state, player):
        if current_game_state is None:
            print("Warning: Received None GameState. Returning default command.")
            return self.my_command

        # If model failed to load, return neutral command to keep game running
        if self.model is None:
            return self.my_command

        # Extract features from the current game state
        p1 = current_game_state.player1
        p2 = current_game_state.player2
        diff = p2.x_coord - p1.x_coord if player == "1" else p1.x_coord - p2.x_coord

        # Prepare features for prediction
        if player == "1":
            features = {
                'player_x': p1.x_coord,
                'opponent_x': p2.x_coord,
                'player_y': p1.y_coord,
                'opponent_y': p2.y_coord,
                'player_health': p1.health,
                'opponent_health': p2.health,
                'player_is_jumping': p1.is_jumping,
                'player_is_crouching': p1.is_crouching,
                'player_in_move': p1.is_player_in_move,
                'opponent_in_move': p2.is_player_in_move,
                'player_move_id': p1.move_id,
                'opponent_move_id': p2.move_id,
                'diff': diff,
                'game_time': current_game_state.timer
            }
        else:  # player == "2"
            features = {
                'player_x': p2.x_coord,
                'opponent_x': p1.x_coord,
                'player_y': p2.y_coord,
                'opponent_y': p1.y_coord,
                'player_health': p2.health,
                'opponent_health': p1.health,
                'player_is_jumping': p2.is_jumping,
                'player_is_crouching': p2.is_crouching,
                'player_in_move': p2.is_player_in_move,
                'opponent_in_move': p1.is_player_in_move,
                'player_move_id': p2.move_id,
                'opponent_move_id': p1.move_id,
                'diff': diff,
                'game_time': current_game_state.timer
            }

        try:
            # Convert features to DataFrame with explicit column names
            feature_df = pd.DataFrame([features], columns=self.feature_names)
            
            # Scale the features
            feature_scaled = self.scaler.transform(feature_df)

            # Predict all button states (multi-label prediction)
            predictions = self.model.predict(feature_scaled)[0]

            # Apply predictions directly without rule-based overrides
            self.buttn.up = bool(predictions[0])
            self.buttn.down = bool(predictions[1])
            self.buttn.left = bool(predictions[2])
            self.buttn.right = bool(predictions[3])
            self.buttn.Y = bool(predictions[4])
            self.buttn.B = bool(predictions[5])
            self.buttn.A = bool(predictions[6])
            self.buttn.R = bool(predictions[7])
            self.buttn.L = bool(predictions[8])
            self.buttn.X = bool(predictions[9])
            self.buttn.select = False  # Always disable select to prevent unintended behavior
            self.buttn.start = False   # Always disable start to prevent pausing

            # Set the command based on player
            if player == "1":
                self.my_command.player_buttons = self.buttn
            else:  # player == "2"
                self.my_command.player2_buttons = self.buttn

        except Exception as e:
            print(f"Error in prediction: {e}. Returning default command.")
            # Return neutral command to keep game running
            if player == "1":
                self.my_command.player_buttons = Buttons()
            else:
                self.my_command.player2_buttons = Buttons()

        return self.my_command

    def run_command(self, com, player):
        # Empty method - all logic is in fight()
        pass