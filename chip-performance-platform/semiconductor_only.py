import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

print("🔬 Searching for SEMICONDUCTOR-SPECIFIC datasets only...")

# ONLY semiconductor/electronics/manufacturing related searches
semiconductor_keywords = [
    'semiconductor', 'chip manufacturing', 'wafer', 'integrated circuit',
    'electronic components', 'PCB', 'microchip', 'silicon wafer',
    'fab process', 'VLSI', 'IC design', 'chip testing'
]

relevant_datasets = []

for keyword in semiconductor_keywords:
    try:
        datasets = api.dataset_list(search=keyword, max_size=20)
        for dataset in datasets:
            # Filter for truly relevant titles
            title_lower = dataset.title.lower()
            if any(sem_word in title_lower for sem_word in [
                'semiconductor', 'chip', 'wafer', 'electronic', 'pcb', 
                'microchip', 'silicon', 'integrated circuit', 'ic', 'vlsi'
            ]):
                relevant_datasets.append({
                    'ref': dataset.ref,
                    'title': dataset.title,
                    'size_mb': dataset.totalBytes / 1024 / 1024 if dataset.totalBytes else 0,
                    'keyword': keyword
                })
                print(f"🎯 Found: {dataset.title}")
    except Exception as e:
        print(f"Search error for {keyword}: {e}")

# Remove duplicates
seen = set()
unique_datasets = []
for d in relevant_datasets:
    if d['ref'] not in seen:
        unique_datasets.append(d)
        seen.add(d['ref'])

print(f"\n📋 Found {len(unique_datasets)} RELEVANT semiconductor datasets:")
print("=" * 80)

# Try to download semiconductor-specific datasets
success_count = 0
for dataset in unique_datasets:
    if dataset['size_mb'] < 100:  # Reasonable size
        print(f"\n📥 Downloading: {dataset['title']}")
        print(f"   Dataset: {dataset['ref']}")
        print(f"   Size: {dataset['size_mb']:.1f} MB")
        
        try:
            output_dir = f"data/raw/kaggle/{dataset['ref'].split('/')[-1]}"
            api.dataset_download_files(dataset['ref'], path=output_dir, unzip=True)
            print(f"   ✅ SUCCESS!")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")

print(f"\n🎉 Downloaded {success_count} SEMICONDUCTOR datasets!")
