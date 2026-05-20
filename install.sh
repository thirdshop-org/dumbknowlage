#!/usr/bin/env bash
set -euo pipefail

echo "=== Installation de whisper-nlp-graph ==="

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "Erreur: Python 3 n'est pas installé."
    exit 1
fi

# Dépendances système
echo "[0/6] Installation des dépendances système..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq portaudio19-dev python3-pyaudio ffmpeg 2>/dev/null && \
        echo "  ✓ paquets système installés" || echo "  → certain paquets manquants (optionnel)"
elif command -v brew &>/dev/null; then
    brew install portaudio ffmpeg 2>/dev/null && echo "  ✓ paquets Homebrew installés" || true
fi

echo "[1/6] Création de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
fi
source venv/bin/activate

echo "[2/6] Installation des dépendances Python..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "[3/6] Téléchargement des modèles spaCy..."
python -m spacy download fr_core_news_lg 2>/dev/null || echo "  → spaCy FR: déjà présent"
python -m spacy download en_core_web_lg 2>/dev/null || echo "  → spaCy EN: déjà présent"

echo "[4/6] Création des dossiers de données..."
mkdir -p data

echo "[5/6] Démarrage ArangoDB (Docker)..."
if command -v docker &>/dev/null; then
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q whisper-nlp-arango; then
        echo "  → ArangoDB déjà en cours d'exécution"
    else
        docker compose up -d 2>/dev/null || docker-compose up -d 2>/dev/null || \
            echo "  → Docker indisponible, démarrez ArangoDB manuellement"
    fi
else
    echo "  → Docker non trouvé, démarrez ArangoDB manuellement"
fi

echo "[6/6] Vérification..."
python -c "
import whisper, spacy, torch, sounddevice, arango, rich
print(f'  ✓ Whisper: ok')
print(f'  ✓ spaCy: {spacy.__version__}')
print(f'  ✓ Torch: {torch.__version__}')
print(f'  ✓ sounddevice: ok')
print(f'  ✓ Arango: ok')
print(f'  ✓ Rich: {rich.__version__}')
" 2>&1 | grep "  ✓"

echo ""
echo "=== Installation terminée ==="
echo ""
echo "Commandes rapides :"
echo "  source venv/bin/activate"
echo "  python main.py record --duration 60 --lang fr --build-graph"
echo "  python main.py pipeline fichier.mp3 --lang fr --build-graph"
echo ""
echo "ArangoDB : http://localhost:8529 (root / whispernlp)"
