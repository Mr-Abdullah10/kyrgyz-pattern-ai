from better_bing_image_downloader import downloader

# ── GEOMETRIC queries ─────────────────────────────────────────
geometric_queries = [
    "kyrgyz geometric pattern shyrdak",
    "kyrgyz felt carpet geometric ornament",
    "shyrdak geometric pattern red black",
    "kyrgyz ala kiyiz geometric carpet",
    "кыргыз геометрик оюм шырдак",       # Kyrgyz language query
    "kyrgyz diamond grid felt pattern",
    "central asian geometric felt carpet",
]

# ── ANIMAL / ZOOMORPHIC queries ───────────────────────────────
animal_queries = [
    "kyrgyz kochkor muyuz ram horn pattern",
    "kyrgyz bugu muyuz deer horn ornament",
    "kyrgyz zoomorphic felt pattern carpet",
    "кыргыз кочкор мүйүз оюм",           # Kyrgyz language
    "kyrgyz horn motif shyrdak carpet",
    "kyrgyz animal inspired ornament felt",
    "ram horn spiral kyrgyz pattern",
]

# ── SYMBOLIC queries ──────────────────────────────────────────
symbolic_queries = [
    "kyrgyz tunduk solar motif ornament",
    "kyrgyz symbolic pattern yurt decoration",
    "kyrgyz umai ene symbol ornament",
    "кыргыз тундук оюм символ",          # Kyrgyz language
    "kyrgyz sun symbol felt carpet",
    "kyrgyz tribal symbol ornament textile",
    "kyrgyz cultural symbol pattern embroidery",
]

# ── DOWNLOAD FUNCTION ─────────────────────────────────────────
def download_batch(queries, output_folder, limit_per_query=30):
    for query in queries:
        print(f"\nDownloading: {query}")
        downloader(
            query=query,
            limit=limit_per_query,
            output_dir=output_folder,
            filter="photo",
            verbose=True,
            badsites=["shutterstock.com", "getty", "alamy.com"],
        )

# ── RUN ───────────────────────────────────────────────────────
download_batch(geometric_queries, "geometric", limit_per_query=25)
download_batch(animal_queries,    "animal",    limit_per_query=25)
download_batch(symbolic_queries,  "symbolic",  limit_per_query=25)

print("\nDone. Now manually review and delete bad images.")