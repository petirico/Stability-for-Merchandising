## Éditeur d’images (Stability AI Inpaint)

Ce projet montre comment éditer une image via l’API « inpaint » de Stability AI, avec prise en charge d’un masque de protection, d’options d’édition (visage/fond), et d’un système de dossiers de résultats horodatés contenant les images et les logs.

### Prérequis
- Python 3.10+
- Une clé API Stability AI active

### Installation
```bash
cd /Users/eric/Project_2/stability-AI-edit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si vous ne souhaitez pas utiliser `requirements.txt`:
```bash
python -m pip install requests pillow opencv-python numpy python-dotenv
```

### Configuration (.env)
Créez un fichier `.env` à la racine :
```
STABILITY_API_KEY=sk-xxxxxxxxxxxxxxxx

# Optionnel
EDIT_IMAGE_PATH=men.png          # image par défaut
EDIT_MASK_PATH=mask.png          # masque projet (noir = protégé, blanc = éditable)
RESULTS_DIR=results              # racine où stocker les runs
```

> Remarque: `.env`, `.venv/` et `__pycache__/` sont ignorés par Git (.gitignore inclus).

### Lancement rapide
```bash
source .venv/bin/activate
# Exemple simple: n’éditer que les zones blanches de mask.png
python -c "from edit import example_protect_areas as run; run()"

# Exemple avancé (Le Morne + visage), sans masques spécifiques (tout éditable), mais avec protections désactivées
python -c "from edit import example_morne_background_and_face as run; run(include_background_mask=False, include_face_mask=False, apply_project_protection=False, apply_polo_protection=False)"
```

### Où se trouvent les résultats ?
- À chaque exécution, un dossier `results/<timestamp>/` est créé.
- Il contient :
  - `men.png` (copie de l’image d’entrée)
  - `mask_initial.png` (si protection projet activée)
  - `debug_final_mask_<timestamp>.png` (si `save_debug_mask=True`)
  - `*_ <timestamp>.png` (image(s) de sortie)
  - `log.txt` et `log.json` (tous les paramètres utilisés)

### Signification du masque
- Noir (0) = protégé, aucune modification.
- Blanc (255) = zone éditable par le modèle.

### API et paramètres clés
Dans `edit.py`, la classe `StabilityImageEditor` appelle l’endpoint :
```
https://api.stability.ai/v2beta/stable-image/edit/inpaint
```
Headers :
- `authorization: Bearer <STABILITY_API_KEY>`
- `accept: image/*`

Paramètres envoyés (multipart/form-data) :
- `image` : l’image d’entrée
- `mask` : masque unique généré côté client
- `prompt` : description textuelle de ce qu’on veut générer
- `negative_prompt` (optionnel)
- `output_format` (par défaut `png`)
- `seed` (0 = aléatoire côté API)

### Fonctions d’exemple
- `example_protect_areas()` : applique le prompt en respectant `mask.png`.
- `example_change_background()` : demande un fond studio noir en respectant `mask.png`.
- `example_morne_background_and_face(...)` : scénario avancé Le Morne/visage.

Signature de `example_morne_background_and_face` :
```
include_background_mask: bool = False     # ajoute un masque de fond auto (GrabCut)
include_face_mask: bool = False           # ajoute un masque « visage » détecté
save_debug_mask: bool = True              # sauvegarde le masque final
apply_project_protection: bool = True     # applique le masque projet (EDIT_MASK_PATH)
apply_polo_protection: bool = True        # protège le polo blanc détecté
```

Cas utiles :
```bash
# 1) Sans aucun masque spécifique ni protection (tout éditable)
python -c "from edit import example_morne_background_and_face as run; run(False, False, True, False, False)"

# 2) Avec protection projet seulement (mask.png), sans masques auto
python -c "from edit import example_morne_background_and_face as run; run(False, False, True, True, False)"

# 3) Avec fond + visage + protections
python -c "from edit import example_morne_background_and_face as run; run(True, True, True, True, True)"
```

### Dépannage
- Masque tout noir => aucune zone éditable; vérifiez vos options et `debug_final_mask_*.png`.
- Aucune génération => assurez-vous que `STABILITY_API_KEY` est bien chargé (visible dans votre shell) et que l’API répond.
- Problèmes d’import `dotenv` dans l’IDE => sélectionnez l’interpréteur de `.venv`.

### Licence
Usage interne/démonstration. Vérifiez les conditions d’utilisation de l’API Stability AI pour l’usage et la distribution des contenus.


