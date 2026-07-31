import os
import subprocess
import json
import csv
from pathlib import Path

import config as cfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = cfg.IMAGE_DIR
REFERENCE_DIR = cfg.REFERENCE_DIR
OUTPUT_CSV = PROJECT_ROOT / 'output' / 'debug' / 'compare_weights_results.csv'

PYTHON = os.environ.get('PYTHON', 'python')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def list_images():
    imgs = []
    for p in IMAGES_DIR.iterdir():
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            imgs.append(p)
    return sorted(imgs)


def list_products():
    prods = []
    if REFERENCE_DIR.exists():
        for p in REFERENCE_DIR.iterdir():
            if p.is_dir():
                prods.append(p.name)
    return sorted(prods)


def run_single(image_path, product, use_per_class, debug_dir):
    env = os.environ.copy()
    env['USE_PER_CLASS'] = '1' if use_per_class else '0'

    cmd = [PYTHON, str(PROJECT_ROOT / 'do_3_razy_sztuka' / 'main.py'),
           '--image', str(image_path),
           '--query', product,
           '--debug_dir', str(debug_dir)]

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print('Run failed:', e)
        return None

    meta_file = Path(debug_dir) / 'detections_metadata.json'
    if not meta_file.exists():
        return None
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print('Failed to read metadata:', e)
        return None


def main():
    images = list_images()
    products = list_products()

    if not images:
        print('No images found in', IMAGES_DIR)
        return
    if not products:
        print('No product reference folders found in', REFERENCE_DIR)
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for product in products:
        for img in images:
            for mode in (False, True):
                mode_name = 'with_override' if mode else 'no_override'
                debug_dir = PROJECT_ROOT / 'output' / 'debug' / product / img.stem / mode_name
                debug_dir.mkdir(parents=True, exist_ok=True)

                print(f'Running image {img.name} product {product} mode {mode_name}...')
                meta = run_single(img, product, mode, debug_dir)

                best = None
                if meta and meta.get('best_match'):
                    best = meta['best_match']

                # collect top detection final_score if any
                top_score = None
                color_score = None
                found = False
                if meta and meta.get('detections'):
                    dets = meta['detections']
                    if len(dets) > 0:
                        top = dets[0]
                        top_score = top.get('final_score')
                        color_score = top.get('color_score')
                        found = top.get('target_match') or False

                rows.append({
                    'image': img.name,
                    'product': product,
                    'mode': mode_name,
                    'best_match': best,
                    'top_final_score': top_score,
                    'top_color_score': color_score,
                    'found_target_match': found,
                    'debug_dir': str(debug_dir)
                })

    # write CSV summary
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image', 'product', 'mode', 'top_final_score', 'top_color_score', 'found_target_match', 'debug_dir']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print('Wrote results to', OUTPUT_CSV)


if __name__ == '__main__':
    main()
