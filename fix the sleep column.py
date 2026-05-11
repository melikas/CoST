import pandas as pd
import os
from datetime import timedelta

# مسیرهای فایل‌ها (از مسیرهای ارائه شده توسط شما استفاده شده است)
main_file_path = r"C:\Users\umroot\Desktop\Human-Rhythms-Dataset\HRD\HRD_RAW_MinuteLevel.csv"
sleep_folder_path = r"C:\Users\umroot\Desktop\Human-Rhythms-Dataset\HRD\Cleaned_Raw_Data\Fitbit\Sleep_Intraday"
output_file_path = r"C:\Users\umroot\Desktop\Human-Rhythms-Dataset\HRD\HRD_RAW_MinuteLevel_Final_correct.csv"

# گام ۱: خواندن فایل اصلی و حذف ستون اشتباه
print("در حال خواندن فایل اصلی...")

# Pre-load sleep data mapping once
print("در حال بارگذاری نقشه داده‌های خواب...")
unique_pids = set()
sleep_map = {}

sleep_folder_path = r"C:\Users\umroot\Desktop\Human-Rhythms-Dataset\HRD\Cleaned_Raw_Data\Fitbit\Sleep_Intraday"
for fname in os.listdir(sleep_folder_path):
    if fname.endswith('.csv'):
        pid = fname[:-4]
        sleep_file = os.path.join(sleep_folder_path, fname)
        
        try:
            df_sleep = pd.read_csv(sleep_file, sep=None, engine='python')
            
            for _, row in df_sleep.iterrows():
                start_time = pd.to_datetime(row['local_date_time'])
                duration_sec = int(row['duration'])
                level = row['level']
                
                end_time = start_time + timedelta(seconds=duration_sec)
                minutes_range = pd.date_range(
                    start=start_time.floor('min'), 
                    end=end_time.floor('min'), 
                    freq='min'
                )
                
                for mn in minutes_range:
                    key = (pid, mn)
                    sleep_map[key] = level
        except Exception as e:
            print(f"Warning: Could not read sleep file for {pid}: {e}")

print(f"Loaded {len(sleep_map)} sleep data points")

# Read CSV in chunks and process each
dtype_map = {
    'pid': str,
    'sleep_level': str,
    'screen': str,
    'calls': str,
    'depression_status_baseline': str,
    'depression_status_endpoint': str,
    'depression_trajectory': str,
}

chunk_size = 50000  # Process 50k rows at a time
is_first_chunk = True

print("Processing CSV in chunks and writing output...")
for chunk in pd.read_csv(main_file_path, dtype=dtype_map, chunksize=chunk_size):
    # Convert dateTime to datetime
    chunk['dateTime'] = pd.to_datetime(chunk['dateTime'])
    
    # Downcast numeric columns
    for col in ['Steps', 'Floors', 'Fairly_Active', 'Lightly_Active', 'Sedentary', 'Very_Active']:
        chunk[col] = chunk[col].astype('float16')
    
    # Populate sleep_level from pre-loaded map
    chunk['sleep_level'] = chunk.apply(
        lambda row: sleep_map.get((row['pid'], row['dateTime']), None), 
        axis=1
    )
    
    # Write chunk directly to output file
    chunk.to_csv(output_file_path, mode='w' if is_first_chunk else 'a', 
                 index=False, header=is_first_chunk)
    is_first_chunk = False
    print(f"  Wrote {len(chunk)} rows...")

print("✓ عملیات با موفقیت انجام شد.")
print(f"✓ فایل در ذخیره شد: {output_file_path}")