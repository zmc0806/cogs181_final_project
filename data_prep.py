#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Novel Data Collection Script for Char-RNN
Downloads and processes romance and science fiction novels from Project Gutenberg.
"""

import os
import re
import requests
import time
from bs4 import BeautifulSoup

# Expanded novel collections
NOVEL_COLLECTIONS = {
    "science_fiction": [
        {
            "title": "The Time Machine",
            "author": "H.G. Wells",
            "url": "https://www.gutenberg.org/files/35/35-0.txt",
            "filename": "time_machine.txt"
        },
        {
            "title": "Twenty Thousand Leagues under the Sea",
            "author": "Jules Verne",
            "url": "https://www.gutenberg.org/files/164/164-0.txt",
            "filename": "20000_leagues.txt"
        },
        {
            "title": "The War of the Worlds",
            "author": "H.G. Wells",
            "url": "https://www.gutenberg.org/files/36/36-0.txt", 
            "filename": "war_of_worlds.txt"
        },
        {
            "title": "Flatland: A Romance of Many Dimensions",
            "author": "Edwin A. Abbott",
            "url": "https://www.gutenberg.org/files/201/201-0.txt",
            "filename": "flatland.txt"
        },
        {
            "title": "Looking Backward: 2000–1887",
            "author": "Edward Bellamy",
            "url": "https://www.gutenberg.org/cache/epub/624/pg624.txt",
            "filename": "looking_backward.txt"
        },
        {
            "title": "A Journey to the Centre of the Earth",
            "author": "Jules Verne",
            "url": "https://www.gutenberg.org/files/3748/3748-0.txt",
            "filename": "journey_to_center.txt"
        }
    ],
    "romance": [
        {
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
            "filename": "pride_and_prejudice.txt"
        },
        {
            "title": "Jane Eyre",
            "author": "Charlotte Brontë",
            "url": "https://www.gutenberg.org/files/1260/1260-0.txt",
            "filename": "jane_eyre.txt"
        },
        {
            "title": "Sense and Sensibility",
            "author": "Jane Austen",
            "url": "https://www.gutenberg.org/files/161/161-0.txt",
            "filename": "sense_and_sensibility.txt"
        },
        {
            "title": "Emma",
            "author": "Jane Austen",
            "url": "https://www.gutenberg.org/files/158/158-0.txt",
            "filename": "emma.txt"
        },
        {
            "title": "Persuasion",
            "author": "Jane Austen",
            "url": "https://www.gutenberg.org/files/105/105-0.txt",
            "filename": "persuasion.txt"
        },
        {
            "title": "The Tenant of Wildfell Hall",
            "author": "Anne Brontë",
            "url": "https://www.gutenberg.org/files/969/969-0.txt",
            "filename": "tenant_of_wildfell.txt"
        }
    ]
}

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def download_novel(url, save_path, retries=3, delay=1):
    for attempt in range(retries):
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Successfully downloaded to {save_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Failed to download {url} after {retries} attempts")
                return False

def clean_gutenberg_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT"
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "End of Project Gutenberg's"
    ]
    start_pos = 0
    for marker in start_markers:
        pos = content.find(marker)
        if pos != -1:
            start_pos = content.find("\n", pos) + 1
            break
    end_pos = len(content)
    for marker in end_markers:
        pos = content.find(marker)
        if pos != -1:
            end_pos = pos
            break
    cleaned_content = content[start_pos:end_pos].strip()
    cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    print(f"Cleaned text file: {file_path}")
    return True

def combine_text_files(directory, output_filename="input.txt"):
    output_path = os.path.join(directory, output_filename)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if filename.endswith('.txt') and filename != output_filename and os.path.isfile(filepath):
                print(f"Adding {filename} to {output_filename}")
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                outfile.write(content)
                outfile.write("\n\n")
    print(f"Combined text saved to {output_path}")
    return output_path

def print_stats(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    char_count = len(content)
    word_count = len(content.split())
    line_count = content.count('\n') + 1
    print(f"Statistics for {filepath}:")
    print(f"  - Characters: {char_count}")
    print(f"  - Words: {word_count}")
    print(f"  - Lines: {line_count}")
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  - File size: {size_kb:.2f} KB")

def main():
    print("Starting novel data collection for char-rnn...")
    data_dir = "data"
    create_directory(data_dir)
    for style, novels in NOVEL_COLLECTIONS.items():
        style_dir = os.path.join(data_dir, style)
        create_directory(style_dir)
        for novel in novels:
            save_path = os.path.join(style_dir, novel["filename"])
            if not os.path.exists(save_path):
                success = download_novel(novel["url"], save_path)
                if success:
                    print(f"Downloaded '{novel['title']}' by {novel['author']}")
                    clean_gutenberg_text(save_path)
                else:
                    print(f"Failed to download '{novel['title']}'")
            else:
                print(f"File already exists: {save_path}")
        combined_path = combine_text_files(style_dir)
        print_stats(combined_path)
    print("\nData collection complete!")
    print("Example training commands:")
    print("  Science Fiction: python train.py --data_dir=data/science_fiction --save_dir=save/scifi")
    print("  Romance: python train.py --data_dir=data/romance --save_dir=save/romance")

if __name__ == "__main__":
    main()
