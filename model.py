import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load and preprocess the data for both players
def load_and_preprocess_data_two_player():
    # Load the dataset
    df = pd.read_csv(r"D:\AI Project\AI Project\gamebot-competition-master\PythonAPI\game_data_augmented.csv")

    # Create Player 1 samples
    p1_data = df[['player1_x_coord', 'player2_x_coord', 'player1_y_coord', 'player2_y_coord',
                  'player1_health', 'player2_health', 'player1_is_jumping', 'player1_is_crouching',
                  'player1_in_move', 'player2_in_move', 'player1_move_id', 'player2_move_id',
                  'diff', 'timer']].copy()
    p1_data.columns = ['player_x', 'opponent_x', 'player_y', 'opponent_y', 'player_health', 'opponent_health',
                       'player_is_jumping', 'player_is_crouching', 'player_in_move', 'opponent_in_move',
                       'player_move_id', 'opponent_move_id', 'diff', 'game_time']
    
    # Create Player 2 samples
    p2_data = df[['player2_x_coord', 'player1_x_coord', 'player2_y_coord', 'player1_y_coord',
                  'player2_health', 'player1_health', 'player2_is_jumping', 'player2_is_crouching',
                  'player2_in_move', 'player1_in_move', 'player2_move_id', 'player1_move_id',
                  'diff', 'timer']].copy()
    p2_data.columns = ['player_x', 'opponent_x', 'player_y', 'opponent_y', 'player_health', 'opponent_health',
                       'player_is_jumping', 'player_is_crouching', 'player_in_move', 'opponent_in_move',
                       'player_move_id', 'opponent_move_id', 'diff', 'game_time']

    # Combine both players' data
    X = pd.concat([p1_data, p2_data], ignore_index=True)
    
    # Target variables (all buttons)
    y = df[['up', 'down', 'left', 'right', 'Y', 'B', 'A', 'R', 'L', 'X', 'select', 'start']]
    y = pd.concat([y, y], ignore_index=True)  # Match the concatenation of X

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Save scaler
    joblib.dump(scaler, r'D:\AI Project\AI Project\gamebot-competition-master\PythonAPI\scaler.pkl')

    return X_train_scaled, X_val_scaled, y_train, y_val

# Hyperparameter tuning using GridSearchCV for XGBoost
def tune_model(X_train_scaled, y_train):
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [30, 50],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1]
    }

    model = xgb.XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss')

    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Train the model and evaluate
def train_model(X_train_scaled, X_val_scaled, y_train, y_val):
    model = tune_model(X_train_scaled, y_train)

    # Predict and evaluate for each button
    buttons = ['up', 'down', 'left', 'right', 'Y', 'B', 'A', 'R', 'L', 'X', 'select', 'start']
    y_pred = model.predict(X_val_scaled)

    for i, button in enumerate(buttons):
        accuracy = accuracy_score(y_val[button], y_pred[:, i])
        print(f"\nValidation Accuracy for {button}: {accuracy:.2f}")
        print(f"Classification Report for {button}:")
        print(classification_report(y_val[button], y_pred[:, i]))
        print(f"Confusion Matrix for {button}:")
        print(confusion_matrix(y_val[button], y_pred[:, i]))

    # Save model
    joblib.dump(model, r'D:\AI Project\AI Project\gamebot-competition-master\PythonAPI\game_bot_model.pkl')

    return model

# Entry point
def main():
    X_train_scaled, X_val_scaled, y_train, y_val = load_and_preprocess_data_two_player()
    train_model(X_train_scaled, X_val_scaled, y_train, y_val)

if __name__ == '__main__':
    main()