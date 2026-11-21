#!/usr/bin/env python3
"""
FreshVision AI - Starter Dataset Generator
Generates a structured, synthetic seed dataset of fresh and spoiled produce samples
with realistic visual characteristics (color distributions, surface decay, mold spots)
so developers and recruiters can run and test train_model.py immediately.
"""

import os
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Reproducible generation
random.seed(42)
np.random.seed(42)

def generate_fresh_sample(filename: Path):
    """
    Generates a realistic synthetic image representing fresh produce:
    - Smooth surface gradients (vibrant red, green, or yellow)
    - Natural specular shine and clean edges
    - Absence of browning, discoloration, or microbial fuzz
    """
    img_size = (224, 224)
    # Background: clean soft kitchen/studio surface
    bg_color = (random.randint(235, 248), random.randint(235, 248), random.randint(235, 248))
    img = Image.new('RGB', img_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Fruit type variation
    fruit_type = random.choice(['apple', 'orange', 'lime', 'banana_slice'])

    if fruit_type == 'apple':
        base_color = (random.randint(190, 230), random.randint(20, 50), random.randint(30, 60))
        highlight_color = (255, 120, 120)
    elif fruit_type == 'orange':
        base_color = (random.randint(230, 255), random.randint(120, 160), random.randint(10, 30))
        highlight_color = (255, 200, 110)
    elif fruit_type == 'lime':
        base_color = (random.randint(40, 70), random.randint(180, 220), random.randint(40, 80))
        highlight_color = (160, 240, 160)
    else:  # banana_slice
        base_color = (random.randint(230, 250), random.randint(210, 230), random.randint(90, 130))
        highlight_color = (255, 255, 180)

    # Main fruit body (ellipse)
    margin = random.randint(25, 40)
    bbox = [margin, margin, 224 - margin, 224 - margin]
    draw.ellipse(bbox, fill=base_color, outline=None)

    # Natural highlight (sheen)
    hl_x1 = margin + random.randint(20, 45)
    hl_y1 = margin + random.randint(15, 35)
    draw.ellipse([hl_x1, hl_y1, hl_x1 + 35, hl_y1 + 25], fill=highlight_color)

    # Soften features
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    img.save(filename, quality=95)

def generate_spoiled_sample(filename: Path):
    """
    Generates a realistic synthetic image representing spoiled produce:
    - Oxidized enzymatic browning and dark irregular necrotic spots
    - Wrinkled borders and microbial fuzz patches
    - Uneven, degraded surface coloration
    """
    img_size = (224, 224)
    bg_color = (random.randint(230, 245), random.randint(230, 245), random.randint(230, 245))
    img = Image.new('RGB', img_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Degraded, browned base tone
    decay_type = random.choice(['brown_rot', 'moldy_blemish', 'shriveled'])

    if decay_type == 'brown_rot':
        base_color = (random.randint(90, 130), random.randint(55, 80), random.randint(30, 50))
    elif decay_type == 'moldy_blemish':
        base_color = (random.randint(120, 150), random.randint(90, 110), random.randint(60, 80))
    else:
        base_color = (random.randint(110, 140), random.randint(70, 95), random.randint(45, 65))

    margin = random.randint(25, 40)
    bbox = [margin, margin, 224 - margin, 224 - margin]
    draw.ellipse(bbox, fill=base_color, outline=None)

    # Overlay multiple necrotic decay spots & fungal blemishes
    num_spots = random.randint(6, 14)
    for _ in range(num_spots):
        spot_x = random.randint(margin + 20, 224 - margin - 35)
        spot_y = random.randint(margin + 20, 224 - margin - 35)
        spot_radius = random.randint(8, 26)
        # Mold colors: dark brown, grayish black, or powdery blue-green
        spot_color = random.choice([
            (random.randint(30, 60), random.randint(20, 40), random.randint(15, 30)),
            (random.randint(80, 110), random.randint(110, 135), random.randint(100, 120)), # grayish mold
            (random.randint(50, 75), random.randint(75, 95), random.randint(70, 85))
        ])
        draw.ellipse([spot_x, spot_y, spot_x + spot_radius, spot_y + spot_radius], fill=spot_color)

    # Add textural noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    img.save(filename, quality=95)

def create_starter_dataset(base_path: Path, samples_per_class={'train': 60, 'validation': 16, 'test': 16}):
    """
    Creates dataset/train, dataset/validation, and dataset/test with fresh/ and spoiled/ images.
    """
    print(f"Creating starter dataset at: {base_path}...")
    for split, count in samples_per_class.items():
        fresh_dir = base_path / split / 'fresh'
        spoiled_dir = base_path / split / 'spoiled'
        fresh_dir.mkdir(parents=True, exist_ok=True)
        spoiled_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, count + 1):
            generate_fresh_sample(fresh_dir / f"fresh_{split}_{i:03d}.jpg")
            generate_spoiled_sample(spoiled_dir / f"spoiled_{split}_{i:03d}.jpg")

        print(f"  [{split.upper()}] Generated {count} fresh and {count} spoiled samples.")

    print("\nStarter dataset successfully initialized!")

if __name__ == '__main__':
    dataset_dir = Path(__file__).resolve().parent / 'dataset'
    create_starter_dataset(dataset_dir)
