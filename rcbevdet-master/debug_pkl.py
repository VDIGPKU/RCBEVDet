import mmcv
import os

pkl_path = '/mnt/datasets/nuscenes/nuscenes_infos_val_mini.pkl'
data = mmcv.load(pkl_path, file_format='pkl')
print(f"Metadata: {data.get('metadata', 'No metadata')}")
if 'infos' in data and len(data['infos']) > 0:
    info = data['infos'][0]
    print(f"Keys in info: {info.keys()}")
    
    for key in ['cams', 'sweeps', 'radars']:
        if key in info:
            val = info[key]
            if isinstance(val, dict):
                print(f"Keys in info['{key}']: {val.keys()}")
            elif isinstance(val, list):
                print(f"Length of info['{key}']: {len(val)}")
                if len(val) > 0:
                    print(f"Sample item keys in info['{key}']: {val[0].keys() if isinstance(val[0], dict) else type(val[0])}")
        else:
            print(f"Key '{key}' not found in info")

    # Check for anything else
    for k, v in info.items():
        if isinstance(v, dict) and 'RADAR' in str(v.keys()):
             print(f"Found RADAR related dict in key '{k}': {v.keys()}")
else:
    print("No infos found in pkl")
