"""Download + unpack UCI-HAR into data/raw (raw data is NOT committed to git).
   python data/download.py
If both URLs fail (e.g. network policy), download manually from
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
and unzip it anywhere under data/raw/ (the loader finds it automatically)."""
import sys, urllib.request, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `src` is importable

URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://github.com/MaxBenChrist/human-activity-dataset/blob/master/UCI%20HAR%20Dataset.zip?raw=True",
]
dest = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
dest.mkdir(parents=True, exist_ok=True)
tmp = dest / "har_download.zip"
for url in URLS:
    try:
        print("downloading", url)
        urllib.request.urlretrieve(url, tmp)
        zipfile.ZipFile(tmp).testzip()
        break
    except Exception as e:                      # try the next mirror
        print("  failed:", e)
else:
    sys.exit("All downloads failed - see the docstring for the manual route.")


def unpack(z: Path):
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    z.unlink()
    for inner in dest.rglob("*.zip"):           # the UCI archive may contain a nested zip
        if "__MACOSX" not in inner.parts:
            unpack(inner)
            return


unpack(tmp)
from src.datasets.har import find_root          # noqa: E402  (verifies the layout)
print("OK ->", find_root(dest))
