from pathlib import Path
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
root = Path("T://multimodal_brain_inspired//NOD//imagenet")

bad = []
for p in root.rglob("*"):
    if p.is_file():
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append((str(p), repr(e)))

print("bad:", len(bad))
for x in bad[:50]:
    print(x)
