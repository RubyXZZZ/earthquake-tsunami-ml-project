import requests
import pandas as pd
import os

START_DATE = '2013-01-01'
END_DATE = '2025-10-31'
MIN_MAGNITUDE = 6
OUTPUT_FILE = './earthquake_tsunami_raw_data.csv'

TARGET_COLUMNS = [
    'title',
    'magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap',
    'depth', 'latitude', 'longitude','year', 'month','tsunami',
    'place', 'alert', 'magType', 'rms',
    'code', 'net', 'type', 'status', 'datetime'
]

def get_earthquakes_data():
    # API ENDPOINT
    url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'

    params = {
        'format': 'geojson',
        'starttime': START_DATE,
        'endtime': END_DATE,
        'minmagnitude': MIN_MAGNITUDE,
        'orderby': 'time',
    }

    try:
        response = requests.get(url, params=params, timeout=120)

        if response.status_code == 200:
            data = response.json()
            print(f"Total earthquakes: {len(data['features'])}")

            # transfer GeoJSON to pandas dataframe
            records = []

            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                timestamp = pd.to_datetime(props.get('time'), unit='ms')

                record = {
                    'title': props.get('title'),
                    'magnitude': props.get('mag'),
                    'cdi': props.get('cdi'),
                    'mmi': props.get('mmi'),
                    'sig': props.get('sig'),
                    'nst': props.get('nst'),
                    'dmin': props.get('dmin'),
                    'gap': props.get('gap'),
                    'depth': coords[2],
                    'latitude': coords[1],
                    'longitude': coords[0],
                    'year': timestamp.year,
                    'month': timestamp.month,
                    'tsunami': props.get('tsunami'),
                    'place': props.get('place'),
                    'alert': props.get('alert'),
                    'magType': props.get('magType'),
                    'rms': props.get('rms'),
                    'code': props.get('code'),
                    'net': props.get('net'),
                    'type': props.get('type'),
                    'status': props.get('status'),
                    'datetime': timestamp,
                }

                records.append(record)

            df = pd.DataFrame(records, columns=TARGET_COLUMNS)
            print(f"Created DataFrame with {len(df)} records")

            # make sure output directory exists
            output_dir = os.path.dirname(OUTPUT_FILE)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # save CSV
            df.to_csv(OUTPUT_FILE, index=False)

            file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)


            #check data
            print(f"\nColumns ({len(df.columns)}):")

            for i, col in enumerate(TARGET_COLUMNS, 1):
                non_null = df[col].notna().sum()
                pct = non_null / len(df) * 100
                print(f"   {i:2d}. {col:15s} - {non_null:4d}/{len(df)} ({pct:5.1f}%)")

            print(df.head().to_string())



        else:
            print(f"Request failed. Status Code: {response.status_code}.  {response.text[:200]}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':

    df = get_earthquakes_data()