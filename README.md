## Image Editor (Stability AI Inpaint)

This project demonstrates how to edit an image via Stability AI's "inpaint" API, with support for a protection mask, editing options (face/background), and timestamped result folders containing images and logs.

### Prerequisites
- Python 3.10+
- An active Stability AI API key

### Installation
```bash
cd /Users/eric/Project_2/stability-AI-edit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not want to use `requirements.txt`:
```bash
python -m pip install requests pillow opencv-python numpy python-dotenv
```

### Configuration (.env)
Create a `.env` file at the project root:
```
STABILITY_API_KEY=sk-xxxxxxxxxxxxxxxx

# Optional
EDIT_IMAGE_PATH=men.png          # default image
EDIT_MASK_PATH=mask.png          # project mask (black = protected, white = editable)
RESULTS_DIR=results              # root folder to store runs
```

> Note: `.env`, `.venv/`, and `__pycache__/` are ignored by Git (included in .gitignore).

### Quick Start
```bash
source .venv/bin/activate
# Simple example: only edit the white areas of mask.png
python -c "from edit import example_protect_areas as run; run()"

# Advanced example (Le Morne + face), with no specific masks (everything editable), protections disabled
python -c "from edit import example_morne_background_and_face as run; run(include_background_mask=False, include_face_mask=False, apply_project_protection=False, apply_polo_protection=False)"
```

### Where are the results saved?
- Each run creates a `results/<timestamp>/` folder.
- It contains:
  - `men.png` (a copy of the input image)
  - `mask_initial.png` (if project protection was enabled)
  - `debug_final_mask_<timestamp>.png` (if `save_debug_mask=True`)
  - `*_<timestamp>.png` (output image(s))
  - `log.txt` and `log.json` (all parameters used)

### Mask meaning
- Black (0) = protected, no modification.
- White (255) = area editable by the model.

### API and key parameters
In `edit.py`, the `StabilityImageEditor` class calls the endpoint:
```
https://api.stability.ai/v2beta/stable-image/edit/inpaint
```
Headers:
- `authorization: Bearer <STABILITY_API_KEY>`
- `accept: image/*`

Parameters sent (multipart/form-data):
- `image`: input image
- `mask`: single mask generated on the client side
- `prompt`: textual description of what to generate
- `negative_prompt` (optional)
- `output_format` (default `png`)
- `seed` (0 = random on the API side)

### Example functions
- `example_protect_areas()`: applies the prompt while respecting `mask.png`.
- `example_change_background()`: requests a black studio background while respecting `mask.png`.
- `example_morne_background_and_face(...)`: advanced Le Morne/face scenario.

Signature of `example_morne_background_and_face`:
```
include_background_mask: bool = False     # add an automatic background mask (GrabCut)
include_face_mask: bool = False           # add a detected "face" mask
save_debug_mask: bool = True              # save the final mask
apply_project_protection: bool = True     # apply the project mask (EDIT_MASK_PATH)
apply_polo_protection: bool = True        # protect the detected white polo shirt
```

Useful cases:
```bash
# 1) No specific masks or protections (everything editable)
python -c "from edit import example_morne_background_and_face as run; run(False, False, True, False, False)"

# 2) With project protection only (mask.png), no auto masks
python -c "from edit import example_morne_background_and_face as run; run(False, False, True, True, False)"

# 3) With background + face + protections
python -c "from edit import example_morne_background_and_face as run; run(True, True, True, True, True)"
```

### Troubleshooting
- All-black mask => no editable area; check your options and `debug_final_mask_*.png`.
- No generation => make sure `STABILITY_API_KEY` is loaded (visible in your shell) and the API is responsive.
- `dotenv` import issues in the IDE => select the `.venv` interpreter.

### License
Internal/demo usage. Check Stability AI's Terms of Use for usage and content distribution.


