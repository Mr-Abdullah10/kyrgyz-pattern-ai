from ddgs import DDGS
import requests, os, time

symbolic_queries = [
    "kyrgyz tunduk ornament circular",
    "kyrgyz sun symbol mandala carpet",
    "tunduk yurt roof circle pattern",
    "kyrgyz round medallion ornament",
    "central asian sun motif circular textile",
    "kyrgyz cosmic symbol circular pattern",
    "tunduk felt carpet round design",
    "kyrgyz circular ornament embroidery",
    "kyrgyz wheel pattern decoration",
    "central asian mandala carpet pattern",
]

geometric_queries = [
    "kyrgyz shyrdak diamond pattern",
    "kyrgyz geometric felt carpet angular",
    "central asian diamond grid carpet",
    "shyrdak red black geometric pattern",
    "kyrgyz felt rug triangle pattern",
    "kyrgyz angular geometric ornament",
    "central asian zigzag carpet pattern",
    "kyrgyz traditional geometric textile",
]

def download_ddg(queries, folder, max_per_query=60):
    os.makedirs(folder, exist_ok=True)
    total = 0
    for query in queries:
        print(f"\nDownloading: {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_per_query))
                for i, r in enumerate(results):
                    try:
                        img_data = requests.get(r['image'], timeout=5).content
                        if len(img_data) < 5000:
                            continue
                        filename = f"{folder}/{query[:15].replace(' ','_')}_{i}.jpg"
                        with open(filename, 'wb') as f:
                            f.write(img_data)
                        total += 1
                        print(f"  Saved {total} images", end="\r")
                    except:
                        pass
            time.sleep(5)
        except Exception as e:
            print(f"  Error: {e}")
            print("  Rate limited — waiting 30 seconds...")
            time.sleep(30)
    print(f"\nTotal downloaded in {folder}: {total}")

if __name__ == '__main__':
    print("Downloading SYMBOLIC images...")
    download_ddg(symbolic_queries, "sorted/symbolic", max_per_query=60)
    
    print("\nWaiting 30 seconds...")
    time.sleep(30)
    
    print("Downloading GEOMETRIC images...")
    download_ddg(geometric_queries, "sorted/geometric", max_per_query=60)
    
    print("\nDone! Now run auto_sort.py again to re-classify")