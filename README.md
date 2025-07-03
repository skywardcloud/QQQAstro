# QQQAstro

QQQAstro enriches intraday QQQ CSV data with basic Vedic astrology markers.
All planetary positions are computed in sidereal coordinates. Sun and Moon
longitudes are taken from the Swiss Ephemeris (tropical values with the
ayanāṁśa subtracted) while the remaining planets still use Skyfield. The
output also includes the sign and nakshatra of every planet calculated with
the Swiss Ephemeris.

## Requirements
- Python 3
- pandas
- pytz
- skyfield
- swisseph

## Usage
```bash
python QQQAstro.py --csv path/to/QQQ.csv --out result.csv
```
If `--csv` is omitted the script tries to locate the file automatically in the
directory defined by the `QQQ_DATA_DIR` environment variable. If that variable
is unset it falls back to `./mnt/data`.


## Environment variable

When the `--csv` option is omitted, the script looks for a CSV file
containing `qqq` in the directory specified by the `QQQ_DATA_DIR`
environment variable. If the variable is unset, `./mnt/data` is used.
Install dependencies with:

```bash
pip install -r requirements.txt
```
This will install pandas, pytz, skyfield and swisseph.

## Running the Web Application

1.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set your Polygon.io API key as an environment variable:**

    ```bash
    export POLYGON_API_KEY='your_api_key'
    ```

3.  **Run the Flask application:**

    ```bash
    python app.py
    ```

4.  **Open your web browser and go to `http://127.0.0.1:5000`**

