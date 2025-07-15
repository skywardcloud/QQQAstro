from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import pandas as pd
import requests
import os
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
            enrich(raw_data_path, enriched_data_path)
            
            enriched_df = pd.read_csv(enriched_data_path)
            
            return jsonify({'message': 'Data enriched and saved successfully!', 'data': enriched_df.to_html()})
        except Exception as e:
            return jsonify({'error': f'An error occurred during data enrichment: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Raw data not found. Please fetch data first.'}), 404

@app.route('/export_to_excel')
def export_to_excel():
    enriched_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enriched_data.csv')
    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enriched_data.xlsx')
    if os.path.exists(enriched_data_path):
        try:
            df = pd.read_csv(enriched_data_path)
            df.to_excel(excel_path, index=False)
            return send_file(excel_path, as_attachment=True)
        except Exception as e:
            return str(e)
    return 'Enriched data not found', 404

@app.route('/predictive_analysis', methods=['POST'])
def predictive_analysis():
    enriched_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enriched_data.csv')
    if not os.path.exists(enriched_data_path):
        return jsonify({'error': 'Enriched data not found. Please enrich data first.'}), 404

    try:
        df = pd.read_csv(enriched_data_path)

        # 1. Define the Target Variable: "Sudden Change"
        # Use lowercase 'h' and 'l' which come from the Polygon API
        df['price_swing_pct'] = ((df['h'] - df['l']) / df['l']) * 100

        # 2. Prepare the Features
        categorical_features = ['hora', 'moon_sign', 'nakshatra', 'lagna_sign', 'lagna_nakshatra']
        boolean_features = [col for col in df.columns if '_cusp_cross' in col or '_house_change' in col or 'lunar_return' in col or 'node_hits_lagna' in col]
        
        categorical_features = [f for f in categorical_features if f in df.columns]
        boolean_features = [f for f in boolean_features if f in df.columns]

        features = categorical_features + boolean_features
        
        df.dropna(subset=features + ['price_swing_pct'], inplace=True)

        if df.empty:
            return jsonify({'error': 'Not enough clean data to run analysis.'}), 400

        X = df[features]
        y = df['price_swing_pct']

        # Create a preprocessor to handle categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        # 3. Create and Train the Model
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

        model.fit(X, y)

        # 4. Extract Feature Importances
        encoded_feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        importances = model.named_steps['regressor'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': encoded_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)

        return jsonify({
            'message': 'Predictive analysis complete!',
            'feature_importances': feature_importance_df.to_html(index=False)
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=False)
