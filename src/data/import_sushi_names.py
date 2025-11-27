#!/usr/bin/env python3
"""
Import custom sushi names into vocabulary
Supports: TXT (one per line), CSV, JSON formats
"""

import json
import argparse
from pathlib import Path


def read_sushi_names_from_txt(file_path: str) -> list:
    """Read sushi names from text file (one per line)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def read_sushi_names_from_csv(file_path: str) -> list:
    """Read sushi names from CSV file"""
    import csv
    names = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                names.append(row[0].strip())
    return names


def read_sushi_names_from_json(file_path: str) -> list:
    """Read sushi names from JSON file (expects list or dict with 'items' key)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'items' in data:
        return data['items']
    elif isinstance(data, dict) and 'sushi_items' in data:
        return data['sushi_items']
    else:
        return list(data.values())[0] if data else []


def update_vocabulary(sushi_names: list, vocab_path: str = 'src/data/sushi_vocab.json'):
    """Update vocabulary JSON with new sushi names"""
    # Load existing vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Get existing items
    existing_items = set(vocab.get('sushi_items', []))
    
    # Add new items (avoid duplicates)
    new_items = []
    for name in sushi_names:
        # Clean and title case
        clean_name = ' '.join(name.strip().split())
        if clean_name and clean_name not in existing_items:
            new_items.append(clean_name)
            existing_items.add(clean_name)
    
    # Update vocabulary
    vocab['sushi_items'] = sorted(list(existing_items))
    
    # Save updated vocabulary
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    return new_items, len(vocab['sushi_items'])


def main():
    parser = argparse.ArgumentParser(description='Import custom sushi names into vocabulary')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input file (TXT, CSV, or JSON)')
    parser.add_argument('--vocab', type=str, default='src/data/sushi_vocab.json',
                        help='Path to vocabulary file')
    parser.add_argument('--preview', action='store_true',
                        help='Preview changes without saving')
    
    args = parser.parse_args()
    
    # Detect file format
    file_ext = Path(args.input).suffix.lower()
    
    print(f"📖 Reading sushi names from {args.input}...")
    
    if file_ext == '.txt':
        sushi_names = read_sushi_names_from_txt(args.input)
    elif file_ext == '.csv':
        sushi_names = read_sushi_names_from_csv(args.input)
    elif file_ext == '.json':
        sushi_names = read_sushi_names_from_json(args.input)
    else:
        print(f"❌ Unsupported file format: {file_ext}")
        print("   Supported formats: .txt, .csv, .json")
        return
    
    print(f"   Found {len(sushi_names)} items in input file")
    
    if args.preview:
        print("\n📋 Preview of items to be added:")
        for i, name in enumerate(sushi_names[:20], 1):
            print(f"   {i}. {name}")
        if len(sushi_names) > 20:
            print(f"   ... and {len(sushi_names) - 20} more")
        print("\nRun without --preview to update vocabulary")
    else:
        new_items, total_items = update_vocabulary(sushi_names, args.vocab)
        
        print(f"\n✅ Vocabulary updated!")
        print(f"   New items added: {len(new_items)}")
        print(f"   Total items in vocabulary: {total_items}")
        
        if new_items:
            print(f"\n📋 Sample of new items added:")
            for i, name in enumerate(new_items[:10], 1):
                print(f"   {i}. {name}")
            if len(new_items) > 10:
                print(f"   ... and {len(new_items) - 10} more")


if __name__ == '__main__':
    main()
