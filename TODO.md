# Olive Doctor — Suivi de livraison

## 1. Socle FastAPI et contrat d’inférence
- [x] Servir une application locale FastAPI avec un point d’entrée clair et un serveur Uvicorn documenté.
- [x] Exposer `GET /api/health`, `GET /api/models/status`, `POST /api/analyze/leaf` et `POST /api/analyze/branch`.
- [x] Retourner le nom technique du modèle, l’identifiant de classe prédit, le libellé français, la confiance et une probabilité pour chaque classe.
- [x] Gérer les erreurs de fichier, les extensions/MIME invalides, les fichiers trop volumineux et les checkpoints indisponibles avec des erreurs françaises explicites, sans fabriquer de prédiction.

## 2. Deux spécialistes indépendants
- [x] Réutiliser exactement `leaf_specialist_best.pth` et `branch_specialist_best.pth` sans réentraînement ni modification des poids.
- [x] Reconstituer `convnextv2_tiny.fcmae_ft_in22k_in1k_384` avec les paramètres d’entraînement nécessaires, `eval()` et `torch.inference_mode()`.
- [x] Préserver l’ordre exact des classes feuille : `Fumagina`, `Healthy_Leaf`, `Mite_Damage`, `Nutritional_Deficiency`, `Peacock_Spot`, `Virosis`.
- [x] Préserver l’ordre exact des classes branche : `Black_Scale`, `Fumagina`, `Healthy_Branch`, `Knot_Disease`.
- [x] Appliquer le prétraitement vérifié : transpose EXIF, composition RGBA sur gris `(128, 128, 128)`, `SmallestMaxSize(int(384 * 1.12))`, `CenterCrop(384)`, normalisation ImageNet.
- [x] Garantir que la route feuille charge/utilise uniquement le spécialiste feuille et que la route branche charge/utilise uniquement le spécialiste branche.
- [x] Ne pas inclure YOLO/YOLOE, détection automatique, bounding boxes, masques, crops, tuiles, agrégation arbre ou visualisations dérivées.
- [x] Supporter CPU par défaut et CUDA automatiquement lorsqu’il est disponible.

## 3. Interface française et expérience d’analyse
- [x] Fournir une interface visible entièrement en français avec la marque Olive Doctor.
- [x] Présenter deux cartes de sélection manuelle : `Leaf Specialist` et `Branch Specialist`, avec état sélectionné clairement identifiable.
- [x] Permettre le glisser-déposer et le sélecteur de fichier, avec validation des types et de la taille.
- [x] Afficher l’aperçu de l’image, le remplacement/la suppression, l’état de chargement, le résultat, la confiance, toutes les probabilités et l’action `Nouvelle analyse`.
- [x] Afficher des erreurs françaises lisibles et ne jamais masquer une erreur technique de chargement ou d’inférence.
- [x] Assurer une mise en page responsive desktop, tablette et mobile.
- [x] Utiliser la direction visuelle premium glassmorphism cinématique : oliviers en plein écran, panneaux givrés, palette olive/profonde/beige/terre, lumière chaude et animations discrètes.
- [x] Ne pas ajouter de boutons décoratifs non fonctionnels, de couleurs saturées ou d’interface générique de tableau de bord.

## 4. Documentation, tests et paquet local
- [x] Inclure `README.md`, `RUN.md`, `requirements.txt` et `run_windows.bat`.
- [x] Inclure des tests couvrant démarrage, health/status, chemins leaf/branch, fichiers invalides, checkpoint absent, CPU, indépendance des modèles et cohérence des probabilités.
- [x] Inclure un manifeste de routes frontend synchronisé avec la page servie.
- [x] Exclure environnements virtuels, caches Python, fichiers temporaires et anciennes fonctions YOLO/YOLOE de l’archive finale.
- [x] Produire et vérifier `/home/ubuntu/Olive_Doctor_Final.zip`.
