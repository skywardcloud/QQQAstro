# QQQAstro

QQQAstro enriches intraday QQQ CSV data with basic Vedic astrology markers.

## Requirements
- Python 3
- pandas
- pytz

## Usage
```bash
python QQQAstro.py --csv path/to/QQQ.csv --out result.csv
```
If `--csv` is omitted the script tries to locate the file automatically in the
directory defined by the `QQQ_DATA_DIR` environment variable. If that variable
is unset it falls back to `/mnt/data`.

QQQAstro

Install dependencies with:

```bash
pip install -r requirements.txt
```

