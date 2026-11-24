# Olive Doctor

Olive Doctor est une application web locale en français pour analyser séparément une **feuille** ou une **branche** d’olivier. L’utilisateur choisit manuellement le spécialiste, importe une image et reçoit une prédiction réelle avec sa confiance et la distribution complète des probabilités.

Cette version réutilise exclusivement les deux checkpoints fournis : `models/leaf_specialist_best.pth` et `models/branch_specialist_best.pth`. Le pipeline historique YOLO/YOLOE et la détection automatique d’organe ne sont pas exécutés ni intégrés.

## Architecture

Le backend Python est servi par FastAPI et Uvicorn. `app/main.py` expose les routes HTTP et sert le frontend statique. `app/services.py` valide les fichiers, prétraite l’image et appelle le spécialiste demandé. `app/model_loader.py` contient deux spécifications indépendantes et charge chaque checkpoint à la première requête qui le nécessite, une seule fois, sur CPU ou CUDA. Aucun endpoint ne déclenche l’autre spécialiste.

Le prétraitement suit le contrat vérifié dans les notebooks : transpose EXIF, composition des images transparentes sur `(128, 128, 128)`, redimensionnement du plus petit côté vers `int(384 × 1,12)`, recadrage central 384 × 384, puis normalisation ImageNet. Les noms et l’ordre des classes sont validés contre les métadonnées du checkpoint ; un mismatch est refusé, jamais réordonné silencieusement.

Le frontend est composé uniquement de HTML5, CSS3 et JavaScript Vanilla. Il propose une direction visuelle glassmorphism cinématique, un choix manuel feuille/branche, un import par sélecteur ou glisser-déposer, un aperçu, les états de chargement et d’erreur, puis un résultat avec barres de probabilité.

## Routes API

- `GET /api/health` — état du service et périphérique détecté.
- `GET /api/models/status` — disponibilité des checkpoints, état de chargement et runtime.
- `POST /api/analyze/leaf` — champ multipart `file`, spécialiste feuille uniquement.
- `POST /api/analyze/branch` — champ multipart `file`, spécialiste branche uniquement.

Une erreur de checkpoint ou de runtime retourne une erreur HTTP explicite. L’application ne fabrique pas de classe de secours et ne renvoie jamais de fausse prédiction.

## Classes affichées

Le spécialiste feuille conserve `Fumagina`, `Healthy_Leaf`, `Mite_Damage`, `Nutritional_Deficiency`, `Peacock_Spot`, `Virosis`, avec des libellés français dans l’interface. Le spécialiste branche conserve `Black_Scale`, `Fumagina`, `Healthy_Branch`, `Knot_Disease`.

## Démarrage

La procédure complète, incluant l’installation PyTorch/timm et le lancement Windows, se trouve dans [`RUN.md`](RUN.md). Sous macOS/Linux, depuis la racine :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Puis ouvrir `http://127.0.0.1:8000`.

## Vérifications

Les tests légers peuvent être lancés avec `pytest -q`. Les tests de contrat frontend vérifient la présence des comportements attendus ; les tests d’inférence réelle sont marqués et ne chargent les deux modèles lourds que si `OLIVE_DOCTOR_RUN_MODEL_TESTS=1` est défini.

L’archive finale doit conserver les deux checkpoints et exclure `.venv`, `__pycache__`, `.pytest_cache`, logs et fichiers temporaires.
