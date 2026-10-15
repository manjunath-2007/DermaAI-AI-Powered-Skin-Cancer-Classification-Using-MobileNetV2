"""Trace DermaAI test images back to their source in the local HAM10000 dataset.

Strategy:
  1. exact md5 match against every HAM10000 file (catches byte-identical copies)
  2. exact pixel match after decoding HAM image at native size (catches re-saves)
  3. perceptual hash (dHash 16x16 + aHash 16x16) nearest-neighbour search
     against all 10015 HAM images, resized to a common grid
"""
import hashlib
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HAM = r"D:\User\Downloads\archive"
TEST_DIR = r"C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app\test-images"
APP_DIR = r"C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app"


def ham_files():
    out = []
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        d = os.path.join(HAM, part)
        for f in os.listdir(d):
            if f.lower().endswith(".jpg"):
                out.append(os.path.join(d, f))
    return sorted(out)


def dhash(img, size=16):
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.float32)
    bits = g[:, 1:] > g[:, :-1]
    return bits.reshape(-1)


def ahash(img, size=16):
    g = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    return (g > g.mean()).reshape(-1)


def ham_array(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main():
    ham = ham_files()
    print("HAM images found:", len(ham))
    sys.stdout.flush()

    # ham metadata
    import csv
    meta = {}
    with open(os.path.join(HAM, "HAM10000_metadata.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            meta[row["image_id"]] = (row["lesion_id"], row["dx"], row["dx_type"])
    print("metadata rows:", len(meta))

    # build HAM hash index
    ham_dh, ham_ah, ham_ids, ham_pixels = [], [], [], {}
    md5_index = {}
    for i, p in enumerate(ham):
        img = Image.open(p)
        ham_dh.append(dhash(img))
        ham_ah.append(ahash(img))
        ham_ids.append(os.path.splitext(os.path.basename(p))[0])
        raw = open(p, "rb").read()
        md5_index[hashlib.md5(raw).hexdigest()] = os.path.basename(p)
        if i % 2000 == 0:
            print("  hashed", i, flush=True)
    ham_dh = np.array(ham_dh)
    ham_ah = np.array(ham_ah)
    print("index built")

    targets = []
    td = os.path.join(TEST_DIR)
    for f in sorted(os.listdir(td)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            targets.append(os.path.join(td, f))
    for f in sorted(os.listdir(APP_DIR)):
        if f.lower().startswith("whatsapp") and f.lower().endswith((".jpg", ".jpeg", ".png")):
            targets.append(os.path.join(APP_DIR, f))

    for tp in targets:
        print("\n" + "=" * 78)
        print("TARGET:", os.path.basename(tp))
        img = Image.open(tp)
        print("  size:", img.size, "mode:", img.mode)
        raw = open(tp, "rb").read()
        md5 = hashlib.md5(raw).hexdigest()
        if md5 in md5_index:
            print("  *** EXACT BYTE MATCH (md5):", md5_index[md5])
        # native-size pixel match
        arr = ham_array(tp)
        native = [h for h in ham if Image.open(h).size == (arr.shape[1], arr.shape[0])]
        exact = None
        for p in native:
            if np.array_equal(ham_array(p), arr):
                exact = p
                break
        if exact:
            print("  *** EXACT PIXEL MATCH:", os.path.basename(exact))
        q = np.array(Image.fromarray(arr).resize((600, 450), Image.LANCZOS), dtype=np.float32)
        qa = np.asarray(Image.fromarray(arr).convert("L").resize((600, 450), Image.LANCZOS), dtype=np.float32)
        best = None
        for p in native:
            b = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            err = float(np.mean(np.abs(b - qa)))
            if best is None or err < best[1]:
                best = (p, err)
        if best:
            iid = os.path.splitext(os.path.basename(best[0]))[0]
            print("  best native-600x450 pixel match: %s  MAE=%.2f" % (iid, best[1]))
            if iid in meta:
                print("      -> lesion=%s dx=%s dx_type=%s" % meta[iid])
        td_ = dhash(img)
        ta_ = ahash(img)
        dd = (ham_dh != td_).sum(axis=1)
        da = (ham_ah != ta_).sum(axis=1)
        order = np.argsort(dd)[:5]
        print("  nearest by dHash:")
        for j in order:
            iid = ham_ids[j]
            info = meta.get(iid, ("?", "?", "?"))
            print("     %s  dHash_dist=%3d  aHash_dist=%3d  dx=%s (%s)" % (iid, dd[j], da[j], info[1], info[0]))
        order = np.argsort(da)[:5]
        print("  nearest by aHash:")
        for j in order:
            iid = ham_ids[j]
            info = meta.get(iid, ("?", "?", "?"))
            print("     %s  aHash_dist=%3d  dHash_dist=%3d  dx=%s (%s)" % (iid, da[j], dd[j], info[1], info[0]))


if __name__ == "__main__":
    main()