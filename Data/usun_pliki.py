import os

# Ścieżka do katalogu wavs
base_dir = 'Dataset/wavs'

# Wczytaj ścieżki z pliku tekstowego
with open('combined_final.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Wyciągnij tylko ścieżki plików .wav z linii (przed pierwszym '|')
whitelist = set(line.strip().split('|')[0] for line in lines if '|' in line)

# Iteruj przez wszystkie podfoldery i pliki w Dataset/wavs/
for root, dirs, files in os.walk(base_dir):
    for file in files:
        full_path = os.path.join(root, file).replace('\\', '/')
        if full_path not in whitelist:
            print(f"Usuwam: {full_path}")
            os.remove(full_path)
