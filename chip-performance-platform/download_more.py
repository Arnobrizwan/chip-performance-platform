import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

print("🔍 Searching for semiconductor datasets...")

# Search for datasets
keywords = ['semiconductor', 'manufacturing', 'sensor', 'quality', 'defect']
available_datasets = []

for keyword in keywords:
    try:
        datasets = api.dataset_list(search=keyword, max_size=50)
        for dataset in datasets[:5]:  # Top 5 per keyword
            available_datasets.append({
                'ref': dataset.ref,
                'title': dataset.title,
                'size_mb': dataset.totalBytes / 1024 / 1024 if dataset.totalBytes else 0,
                'downloads': dataset.downloadCount
            })
    except Exception as e:
        print(f"Search error for {keyword}: {e}")

# Remove duplicates and sort
seen = set()
unique_datasets = []
for d in available_datasets:
    if d['ref'] not in seen:
        unique_datasets.append(d)
        seen.add(d['ref'])

# Sort by download count
unique_datasets = sorted(unique_datasets, key=lambda x: x['downloads'], reverse=True)

print(f"\n📋 Found {len(unique_datasets)} unique datasets:")
print("-" * 80)

# Try to download top 10 smaller datasets
success_count = 0
for i, dataset in enumerate(unique_datasets[:10]):
    if dataset['size_mb'] < 50:  # Only small datasets
        print(f"\n{i+1}. Trying: {dataset['ref']}")
        print(f"   Title: {dataset['title']}")
        print(f"   Size: {dataset['size_mb']:.1f} MB")
        
        try:
            output_dir = f"data/raw/kaggle/{dataset['ref'].split('/')[-1]}"
            api.dataset_download_files(dataset['ref'], path=output_dir, unzip=True)
            print(f"   ✅ Downloaded successfully!")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    else:
        print(f"\n{i+1}. Skipped: {dataset['ref']} (too large: {dataset['size_mb']:.1f} MB)")

print(f"\n🎉 Successfully downloaded {success_count} additional datasets!")
