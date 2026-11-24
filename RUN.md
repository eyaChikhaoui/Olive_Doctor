# Exécuter Olive Doctor en local

## Prérequis

Python 3.10 ou plus récent est recommandé. PyTorch et timm sont nécessaires pour charger les checkpoints réels ; l’installation peut être plus longue et plus volumineuse que celle d’une API web classique. Si une machine CUDA est disponible et compatible avec la version installée de PyTorch, elle sera utilisée automatiquement. Sinon, le serveur fonctionne sur CPU.

## Linux / macOS

```bash
cd Olive_Doctor_Final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Ouvrir ensuite `http://127.0.0.1:8000`.

## Windows

Double-cliquer sur `run_windows.bat`. Le script crée `.venv`, installe `requirements.txt`, puis démarre Uvicorn sur `http://127.0.0.1:8000`.

## Vérification rapide de l’API

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/models/status
curl -F "file=@/chemin/vers/une-feuille.jpg" http://127.0.0.1:8000/api/analyze/leaf
curl -F "file=@/chemin/vers/une-branche.jpg" http://127.0.0.1:8000/api/analyze/branch
```

Les deux endpoints d’analyse sont indépendants. Une analyse feuille ne charge ni n’appelle le checkpoint branche, et inversement. Le premier appel de chaque spécialiste peut être plus lent à cause du chargement du modèle ; les appels suivants réutilisent le modèle gardé en mémoire.

## Tests

```bash
pytest -q
```

Pour activer les tests d’intégration lourds qui chargent et exécutent réellement les deux checkpoints :

```bash
OLIVE_DOCTOR_RUN_MODEL_TESTS=1 pytest -q
```

Sous PowerShell :

```powershell
$env:OLIVE_DOCTOR_RUN_MODEL_TESTS = "1"
pytest -q
```

## Dépannage

- `503` avec « runtime PyTorch/timm indisponible » : installer les dépendances depuis `requirements.txt` dans l’environnement virtuel actif.
- `503` avec checkpoint introuvable : vérifier que les deux fichiers `.pth` sont bien dans `models/` et n’ont pas été renommés.
- `400` au dépôt : seuls JPG, PNG et WebP de 10 Mo maximum sont acceptés.
- La route `/api/models/status` expose la cause technique disponible sans générer de prédiction de remplacement.
