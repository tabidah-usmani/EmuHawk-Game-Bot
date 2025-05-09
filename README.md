# Street Fighter II Turbo Game Bot README

This README provides detailed instructions to set up and run the Street Fighter II Turbo game bot using the provided Python API (`model.py` and `bot.py`) with the BizHawk emulator. The bot uses a machine learning (ML) model (XGBoost) to control player actions in single-player mode (Player 1 vs. CPU) or two-player mode (Player 1 vs. Player 2, both bot-controlled). The instructions cover software dependencies, library installations, file setup, and step-by-step execution to reproduce the experiments on your machine.

## Prerequisites

### Hardware Requirements
- **Operating System**: Windows (tested on Windows 10/11)
- **CPU**: Any modern processor (e.g., Intel i5 or equivalent)
- **RAM**: At least 8 GB
- **Storage**: 500 MB free space for emulator, ROM, and Python environment
- **Graphics**: No specific requirements (emulator uses software rendering)

### Software Requirements
- **BizHawk Emulator**: Version 2.9 or later
- **Python**: Version 3.8 or 3.9 (3.10+ may have compatibility issues with some libraries)
- **Street Fighter II Turbo ROM**: `Street Fighter II Turbo (U).smc` (not provided; must be sourced legally)
- **Text Editor or IDE**: VS Code, PyCharm, or any editor for Python files
- **Command Prompt or PowerShell**: For running Python scripts

### Python Library Dependencies
The following Python libraries are required:
- `pandas>=1.5.0`: Data manipulation and preprocessing
- `scikit-learn>=1.3.0`: StandardScaler and model evaluation
- `xgboost>=2.0.0`: XGBoost classifier for ML model
- `joblib>=1.3.0`: Model and scaler serialization
- `numpy>=1.24.0`: Numerical computations

## Setup Instructions

### Step 1: Install BizHawk Emulator
1. Download BizHawk from [https://tasvideos.org/BizHawk/ReleaseHistory](https://tasvideos.org/BizHawk/ReleaseHistory) (version 2.9 or later).
2. Extract the BizHawk archive to a directory, e.g., `C:\BizHawk\`.
3. Ensure the `EmuHawk.exe` executable is present in the BizHawk directory.

### Step 2: Obtain the ROM File
1. Acquire the `Street Fighter II Turbo (U).smc` ROM file legally (not provided in this repository).
2. Place the ROM file in the appropriate folder:
   - For single-player: `single-player/Street Fighter II Turbo (U).smc`
   - For two-player: `two-players/Street Fighter II Turbo (U).smc`

### Step 3: Set Up Python Environment
1. **Install Python 3.8 or 3.9**:
   - Download from [https://www.python.org/downloads/](https://www.python.org/downloads/).
   - Install, ensuring "Add Python to PATH" is checked.
   - Verify installation:
     ```bash
     python --version
     ```
     Output should be `Python 3.8.x` or `Python 3.9.x`.
2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv gamebot_env
   .\gamebot_env\Scripts\activate
   ```
   The prompt should show `(gamebot_env)`.
3. **Install Required Libraries**:
   With the virtual environment activated, run:
   ```bash
   pip install pandas==1.5.3 scikit-learn==1.3.2 xgboost==2.0.3 joblib==1.3.2 numpy==1.24.4
   ```
   Verify installations:
   ```bash
   pip list
   ```
   Ensure all listed libraries are installed with the specified versions or later.

### Step 4: Organize Project Files
1. **Create Project Directory**:
   - Create a directory, e.g., `C:\GameBot\`.
   - Ensure the directory structure matches one of the following:
     ```
     C:\GameBot\single-player\
     ├── Street Fighter II Turbo (U).smc
     ├── controller.py
     ├── command.py
     ├── buttons.py
     ├── model.py
     ├── bot.py
     ├── scaler.pkl
     ├── game_bot_model.pkl
     ├── game_data_augmented.csv
     └── game_data_log.csv (generated during runtime)
     ```
     ```
     C:\GameBot\two-players\
     ├── Street Fighter II Turbo (U).smc
     ├── controller.py
     ├── command.py
     ├── buttons.py
     ├── model.py
     ├── bot.py
     ├── scaler.pkl
     ├── game_bot_model.pkl
     ├── game_data_augmented.csv
     └── game_data_log.csv (generated during runtime)
     ```
2. **Download Provided Files**:
   - Place `model.py` and `bot.py` (from this repository) in the `single-player/` and/or `two-players/` folders.
   - Ensure `controller.py`, `command.py`, and `buttons.py` are present (typically provided with the GameBot starter code).
   - Place `game_data_augmented.csv` in the same folder (training data, assumed provided or generated previously).
3. **Update File Paths**:
   - The provided `model.py` uses hardcoded paths (e.g., `D:\AI Project\...`). Update `model.py` to use relative paths for portability:
     ```python
     # In model.py, replace:
     df = pd.read_csv(r"D:\AI Project\AI Project\gamebot-competition-master\PythonAPI\game_data_augmented.csv")
     joblib.dump(scaler, r'D:\AI Project\AI Project\gamebot-competition-master\PythonAPI\scaler.pkl')
     joblib.dump(model, r'D:\AI Project\AI Project\gamebot-competition-master\PythonAPI\game_bot_model.pkl')
     # With:
     df = pd.read_csv("game_data_augmented.csv")
     joblib.dump(scaler, "scaler.pkl")
     joblib.dump(model, "game_bot_model.pkl")
     ```
   - The provided `bot.py` uses relative paths (`scaler.pkl`, `game_bot_model.pkl`), which are correct assuming they’re in the same directory. No changes needed unless you move files.

### Step 5: Preprocess Data and Train the Model
1. **Update `model.py` for Non-Idle Predictions**:
   The provided `model.py` lacks preprocessing to replace idle states with active moves, which could lead to idle predictions and freezing. 
2. **Run `model.py`**:
   - Navigate to the project directory (`single-player/` or `two-players/`):
     ```bash
     cd C:\GameBot\single-player
     ```
   - Activate the virtual environment (if used):
     ```bash
     .\gamebot_env\Scripts\activate
     ```
   - Run:
     ```bash
     python model.py
     ```
   - **Generated Files**:
     - `scaler.pkl`: Scaler for feature normalization
     - `game_bot_model.pkl`: Trained XGBoost model

   - **Troubleshooting**:
     - If `game_data_augmented.csv` is missing, you need to provide or generate it (e.g., from previous game logs).
     - If errors occur, verify library versions and Python version (3.8/3.9).

### Step 6: Run the Game Bot

#### Single-Player Mode (Player 1 vs. CPU)
1. **Start BizHawk Emulator**:
   - Open `EmuHawk.exe` from `C:\BizHawk\`.
   - From the **File** menu, select **Open ROM** (Ctrl+O).
   - Navigate to `C:\GameBot\single-player\` and select `Street Fighter II Turbo (U).smc`.
   - From the **Tools** menu, open **Tool Box** (Shift+T).
2. **Select Game Mode and Character**:
   - In the emulator, choose **Normal Mode**.
   - Use the controller settings (Config > Controllers) to select a character for Player 1.
   - Leave the emulator and Tool Box windows open.
3. **Run the Bot**:
   - Open a command prompt or PowerShell.
   - Navigate to `C:\GameBot\single-player\`:
     ```bash
     cd C:\GameBot\single-player
     ```
   - Activate the virtual environment (if used):
     ```bash
     .\gamebot_env\Scripts\activate
     ```
   - Run:
     ```bash
     python controller.py 1
     ```
     The `1` argument specifies controlling Player 1.
   - **Expected Output**:
     - In the terminal: “Connected to game!” or “CONNECTED SUCCESSFULLY”.
     - In the emulator: The bot controls Player 1, making moves (e.g., moving left/right, attacking).
   - **Troubleshooting**:
     - If “Connection failed”, ensure the Tool Box is open and the Gyroscope Bot icon is clicked.
     - If the bot freezes after two moves, check `game_data_log.csv` (step 7).
4. **Connect Emulator to Bot**:
   - In the BizHawk Tool Box, click the **Gyroscope Bot** icon (second icon in the top row).
   - The game should start, with the bot controlling Player 1.
5. **Game Duration**:
   - The game stops after one round (when a player’s health reaches 0 or the timer expires).
   - Repeat steps 1–4 to run another round.

#### Two-Player Mode (Player 1 vs. Player 2, Both Bots)
1. **Start BizHawk Emulator**:
   - Open `EmuHawk.exe` from `C:\BizHawk\`.
   - From **File**, select **Open ROM** (Ctrl+O).
   - Navigate to `C:\GameBot\two-players\` and select `Street Fighter II Turbo (U).smc`.
   - From **Tools**, open **Tool Box** (Shift+T).
2. **Select Game Mode and Characters**:
   - Choose **VS Battle Mode**.
   - Use controller settings to select characters for both Player 1 and Player 2.
   - Leave the emulator and Tool Box open.
3. **Run Two Bots**:
   - Open two command prompt or PowerShell windows.
   - Navigate to `C:\GameBot\two-players\` in both:
     ```bash
     cd C:\GameBot\two-players
     ```
   - Activate the virtual environment in both (if used):
     ```bash
     .\gamebot_env\Scripts\activate
     ```
   - In the first terminal, run:
     ```bash
     python controller.py 1
     ```
   - In the second terminal, run:
     ```bash
     python controller.py 2
     ```
     The `1` and `2` arguments control Player 1 and Player 2, respectively.
   - **Expected Output**:
     - Both terminals show “Connected to game!” or “CONNECTED SUCCESSFULLY”.
     - The emulator shows both players moving and fighting, controlled by the bots.
   - **Troubleshooting**:
     - Ensure both commands are run simultaneously.
     - If one bot doesn’t connect, verify the Gyroscope Bot icon is clicked.
4. **Connect Emulator to Bots**:
   - In the Tool Box, click the **Gyroscope Bot** icon.
   - Both bots should control their respective players.
5. **Game Duration**:
   - Stops after one round. Repeat steps 1–4 for another round.

