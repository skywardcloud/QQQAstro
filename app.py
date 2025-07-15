from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import pandas as pd
import requests
import os
import sqlite3
from QQQAstro import enrich  # Assuming enrich function is in QQQAstro.py
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load environment variables from .env file at the start
load_dotenv(override=True)

print(f"[DEBUG] CWD: {os.getcwd()}")
print(f"[DEBUG] Files in CWD: {os.listdir(os.getcwd())}")
print(f"[DEBUG] API Key at startup: {os.environ.get('POLYGON_API_KEY')}")

app = Flask(__name__)

DATABASE_FILE = 'astro_data.db'

def init_db():
    """Initializes the database and creates the table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # This schema should match the output of your enrich() function.
    # Using TEXT for datetime, REAL for numbers, INTEGER for whole numbers.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS enriched_data (
            timestamp TEXT, open REAL, h REAL, l REAL, close REAL, v INTEGER,
            t TEXT, n INTEGER,
            utc TEXT, hora TEXT, lagna_long REAL, lagna_sign TEXT,
            lagna_nakshatra TEXT, lagna_house INTEGER, moon_long REAL,
            moon_sign TEXT, nakshatra TEXT, moon_house INTEGER,
            any_cusp_cross INTEGER, sun_long REAL, mercury_long REAL,
            venus_long REAL, mars_long REAL, jupiter_long REAL,
            saturn_long REAL, uranus_long REAL, neptune_long REAL,
            pluto_long REAL, rahu_long REAL, ketu_long REAL,
            lunar_return INTEGER, node_hits_lagna INTEGER,
            is_day INTEGER, sunrise TEXT, sunset TEXT,
            lagna_house_change INTEGER, moon_house_change INTEGER,
            house_1_cusp REAL, house_2_cusp REAL, house_3_cusp REAL,
            house_4_cusp REAL, house_5_cusp REAL, house_6_cusp REAL,
            house_7_cusp REAL, house_8_cusp REAL, house_9_cusp REAL,
            house_10_cusp REAL, house_11_cusp REAL, house_12_cusp REAL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

def get_polygon_api_key():
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY environment variable not set. Please add it to your .env file.")
    return key

# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    ticker = request.form.get('ticker')
    from_date_str = request.form.get('from_date')
    to_date_str = request.form.get('to_date')
    interval = request.form.get('interval', '5')  # Default to 5 minutes

    from_date = datetime.strptime(from_date_str, '%Y-%m-%d').date()
    to_date = datetime.strptime(to_date_str, '%Y-%m-%d').date()

    all_results = []
    current_date = from_date
    try:
        api_key = get_polygon_api_key()
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    while current_date <= to_date:
        date_str = current_date.strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{interval}/minute/{date_str}/{date_str}?apiKey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                all_results.extend(data['results'])
        else:
            print(f"Error fetching data for {date_str}: {response.status_code} - {response.text}")
        current_date += timedelta(days=1)

    if all_results:
        df = pd.DataFrame(all_results)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        
        raw_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fetched_data.csv')
        df.to_csv(raw_data_path, index=False)
        
        return jsonify({'message': 'Data fetched and saved successfully!', 'data': df.to_html()})
    else:
        return jsonify({'error': 'No data found for the given ticker and date range.'}), 404

@app.route('/filter_data', methods=['POST'])
def filter_data():
    diff_percentage = float(request.form.get('diff_percentage'))
    raw_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fetched_data.csv')

    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path)
        
        df['h'] = pd.to_numeric(df['h'])
        df['l'] = pd.to_numeric(df['l'])

        df_filtered = df[((df['h'] - df['l']) / df['l'] * 100) >= diff_percentage]
        
        return jsonify({'message': 'Data filtered successfully!', 'data': df_filtered.to_html()})
    else:
        return jsonify({'error': 'Raw data not found. Please fetch data first.'}), 404

@app.route('/enrich_data', methods=['POST'])
def enrich_data():
    raw_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fetched_data.csv')
    enriched_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enriched_data.csv')
    
    if os.path.exists(raw_data_path):
        try:
            # The enrich function creates the enriched CSV
            enrich(raw_data_path, enriched_data_path)
            
            enriched_df = pd.read_csv(enriched_data_path)

            # --- Save to SQLite Database ---
            conn = sqlite3.connect(DATABASE_FILE)
            # Get existing columns to avoid schema mismatch
            db_cols = pd.read_sql("SELECT * FROM enriched_data LIMIT 1", conn).columns
            # Filter dataframe to only include columns that exist in the DB
            enriched_df_filtered = enriched_df[enriched_df.columns.intersection(db_cols)]
            # Append the new data to the table
            enriched_df_filtered.to_sql('enriched_data', conn, if_exists='append', index=False)
            conn.close()
            # -------------------------------
            
            return jsonify({'message': 'Data enriched and saved to database successfully!', 'data': enriched_df.to_html()})
        except Exception as e:
            return jsonify({'error': f'An error occurred during data enrichment: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Raw data not found. Please fetch data first.'}), 404

@app.route('/export_to_excel')
def export_to_excel():
    # Now, export the entire database to Excel
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        df = pd.read_sql_query("SELECT * FROM enriched_data", conn)
        conn.close()

        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], 'full_astro_database.xlsx')
        df.to_excel(excel_path, index=False)
        return send_file(excel_path, as_attachment=True)
    except Exception as e:
        return str(e)

@app.route('/predictive_analysis', methods=['POST'])
def predictive_analysis():
    try:
        # --- Read all data from the SQLite Database ---
        conn = sqlite3.connect(DATABASE_FILE)
        df = pd.read_sql_query("SELECT * FROM enriched_data", conn)
        conn.close()
        # --------------------------------------------

        if df.empty:
            return jsonify({'error': 'Database is empty. Please enrich some data first.'}), 404

        # 1. Define the Target Variable: "Sudden Change"
        # Use lowercase 'h' and 'l' which come from the Polygon API
        df['price_swing_pct'] = ((df['h'] - df['l']) / df['l']) * 100

        # 2. Prepare the Features
        categorical_features = ['hora', 'moon_sign', 'nakshatra', 'lagna_sign', 'lagna_nakshatra']
        boolean_features = [col for col in df.columns if '_cusp_cross' in col or '_house_change' in col or 'lunar_return' in col or 'node_hits_lagna' in col]
        numerical_features = [
            'lagna_long', 'moon_long', 'sun_long', 'mercury_long', 'venus_long',
            'mars_long', 'jupiter_long', 'saturn_long', 'uranus_long', 'neptune_long',
            'pluto_long', 'rahu_long', 'ketu_long',
            'house_1_cusp', 'house_2_cusp', 'house_3_cusp', 'house_4_cusp',
            'house_5_cusp', 'house_6_cusp', 'house_7_cusp', 'house_8_cusp',
            'house_9_cusp', 'house_10_cusp', 'house_11_cusp', 'house_12_cusp'
        ]

        # Filter features to only those present in the dataframe
        categorical_features = [f for f in categorical_features if f in df.columns]
        boolean_features = [f for f in boolean_features if f in df.columns]
        numerical_features = [f for f in numerical_features if f in df.columns]

        # --- Impute Missing Values ---
        # For categorical features, fill with a placeholder string
        df[categorical_features] = df[categorical_features].fillna('Missing')
        
        # For numerical and boolean features, first convert to numeric, coercing errors, then fill NaNs
        for col in numerical_features + boolean_features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Combine all features
        features = categorical_features + boolean_features + numerical_features
        
        # Drop rows only if the target variable itself is missing
        df.dropna(subset=['price_swing_pct'], inplace=True)

        if df.empty:
            return jsonify({'error': 'Not enough data to run analysis (target variable might be missing).'}), 400

        X = df[features]
        y = df['price_swing_pct']

        # Create a preprocessor to handle different feature types
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', 'passthrough', numerical_features + boolean_features)
            ],
            remainder='drop'
        )

        # 3. Create and Train the Model
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

        model.fit(X, y)

        # 4. Extract Feature Importances
        # Get feature names from the preprocessor
        cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
        encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features)
        
        # Combine all feature names in the correct order
        all_feature_names = np.concatenate([encoded_cat_features, numerical_features, boolean_features])
        
        importances = model.named_steps['regressor'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20)

        return jsonify({
            'message': 'Predictive analysis complete!',
            'feature_importances': feature_importance_df.to_html(index=False)
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=False)
