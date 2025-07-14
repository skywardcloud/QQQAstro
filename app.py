from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import pandas as pd
import requests
import os
from QQQAstro import enrich  # Assuming enrich function is in QQQAstro.py
from datetime import datetime, timedelta

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


if __name__ == '__main__':
    app.run(debug=False)
