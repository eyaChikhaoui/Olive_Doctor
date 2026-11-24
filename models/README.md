# Checkpoints Olive Doctor

Les deux fichiers `.pth` de ce dossier sont les checkpoints fournis par le projet source et copiés sans modification.

- `leaf_specialist_best.pth` — `convnextv2_tiny.fcmae_ft_in22k_in1k_384`, 6 classes feuille.
- `branch_specialist_best.pth` — `convnextv2_tiny.fcmae_ft_in22k_in1k_384`, 4 classes branche.

Le backend vérifie les métadonnées `model_name`, `img_size` et `class_names` avant de charger les poids. Il refuse tout mismatch et ne réordonne pas les classes.
