#!/usr/bin/env python3
import os
import numpy as np
import soundfile as sf

# Ścieżka do katalogu wavs
base_dir = 'Dataset/wavs'
# Próg amplitudy – pliki, które nie mają próbki > threshold, zostaną usunięte
threshold = 0.07

# Iteruj przez wszystkie podfoldery i pliki
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if not file.lower().endswith('.wav'):
            continue
        full_path = os.path.join(root, file).replace('\\', '/')
        # Wczytaj dane audio (może być wielokanałowe)
        data, sr = sf.read(full_path)
        # Jeśli nie ma próbki przekraczającej threshold → usuń
        if not np.any(np.abs(data) > threshold):
            print(f"Usuwam: {full_path}")
            os.remove(full_path)