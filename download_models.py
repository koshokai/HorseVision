#!/usr/bin/env python3
"""
Download the Horse Vision checkpoints from GitHub Releases.

    python download_models.py              # download everything (2.2 GB)
    python download_models.py --scene indoor
    python download_models.py --scene outdoor
    python download_models.py --verify     # re-check what is already on disk

Only the Python standard library is used. Files already present with the correct
SHA-256 are skipped, so the script is safe to re-run after an interruption.
"""

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

# --------------------------------------------------------------------------- #
# Edit these two lines to point at your release.
REPO = "koshokai/HorseVision"
TAG = "v1.0"
# --------------------------------------------------------------------------- #

BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

# local path  ->  (asset name in the release, sha256, approx MB)
MODELS = {
    "models/indoor/self_supervised_fish_hm3d/self_fish.tar": (
        "indoor_self_supervised_fish_hm3d_self_fish.tar",
        "c60365e44c3826efdc8ee242eb7e12612b8901479579bdf92f2ff8316f5a284d", 373),
    "models/indoor/supervised_erp/sup_erp.pth": (
        "indoor_supervised_erp_sup_erp.pth",
        "99511c0c79db80ba909cacf3566f3bab69bd7ef3577c65a16477862288f74920", 355),
    "models/indoor/supervised_st3D/sup_fish.tar": (
        "indoor_supervised_st3D_sup_fish.tar",
        "e8cb3b0725e7187660e587da8daa9437bb13d9af36e663cccc01c91b6c12e07f", 372),
    "models/outdoor/self_supervised_fish_carla/self_fish.tar": (
        "outdoor_self_supervised_fish_carla_self_fish.tar",
        "e8ec17b08c6eafdcc4e648b199d1f6fac4bb831b9810e859b24920747c7891d6", 373),
    "models/outdoor/supervised_erp/sup_erp.pth": (
        "outdoor_supervised_erp_sup_erp.pth",
        "628555162a18651548186b5948131a15ba9fd09ac0c6e3f4492b24ead1e2dbee", 355),
    "models/outdoor/supervised_fish_carla/sup_fish.tar": (
        "outdoor_supervised_fish_carla_sup_fish.tar",
        "6008a918d21e29bc1e26bc7e1c12ab710b35ed552747317606dd250c662baf09", 372),
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, expect_mb):
    """Stream to a .part file, then rename — an interrupted run never leaves a
    truncated file that a later run would mistake for a complete one."""
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0)) or expect_mb * 1024 * 1024
            done = 0
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                pct = done * 100 // total if total else 0
                sys.stdout.write(f"\r    {done/1e6:7.1f} / {total/1e6:7.1f} MB  {pct:3d}%")
                sys.stdout.flush()
        sys.stdout.write("\n")
    except urllib.error.HTTPError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if e.code == 404:
            raise SystemExit(
                f"\n[ERROR] 404 Not Found: {url}\n"
                f"        Check that REPO ('{REPO}') and TAG ('{TAG}') at the top of\n"
                f"        this script match your release, and that the asset was uploaded."
            )
        raise
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=["indoor", "outdoor", "all"], default="all")
    ap.add_argument("--verify", action="store_true",
                    help="only verify files already on disk, download nothing")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    items = {k: v for k, v in MODELS.items()
             if args.scene == "all" or f"/{args.scene}/" in k}

    if REPO.startswith("YOUR_USERNAME") and not args.verify:
        raise SystemExit(
            "[ERROR] Please set REPO and TAG at the top of download_models.py first."
        )

    ok = missing = fixed = 0
    for rel, (asset, digest, mb) in items.items():
        dest = os.path.join(root, rel)
        print(f"\n{rel}  ({mb} MB)")

        if os.path.exists(dest):
            print("    verifying existing file ...", end=" ", flush=True)
            if sha256(dest) == digest:
                print("OK, skipping")
                ok += 1
                continue
            print("checksum MISMATCH")
            if args.verify:
                missing += 1
                continue
            print("    re-downloading")
        elif args.verify:
            print("    not present")
            missing += 1
            continue

        download(f"{BASE_URL}/{asset}", dest, mb)
        print("    verifying ...", end=" ", flush=True)
        if sha256(dest) != digest:
            os.remove(dest)
            raise SystemExit(
                "FAILED\n[ERROR] Checksum mismatch after download. The file was removed.\n"
                "        Re-run the script; if it keeps failing the release asset may be corrupt."
            )
        print("OK")
        fixed += 1

    print(f"\n{'='*60}")
    if args.verify:
        print(f"verified: {ok} ok, {missing} missing or corrupt")
        sys.exit(1 if missing else 0)
    print(f"done: {ok} already present, {fixed} downloaded")
    print("You can now run:  python test_single_conf.py")


if __name__ == "__main__":
    main()
