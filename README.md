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
EDIT_IMAGE_PATH=images-src/let-me-cook.png   # default image used by let-me-cook-4.py
EDIT_MASK_PATH=mask/tshirt-1000.png         # project mask (black = protected, white = editable)
RESULTS_DIR=results                          # root folder to store runs
```

> Note: `.env`, `.venv/`, and `__pycache__/` are ignored by Git (included in .gitignore).

### Quick Start
```bash
source .venv/bin/activate
# Run the main "Let me cook" scenario
python let-me-cook-4.py
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
In `let-me-cook-4.py`, the `StabilityImageEditor` class calls the endpoint:
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

### Main function and parameters (let-me-cook-4.py)
- `image_edit_background_and_face_mask(...)`: orchestrates mask creation/combination, feathering, and the API call.

Actual signature in the script:
```
save_debug_mask: bool = True
apply_project_protection: bool = True
feather_radius: int = 8
grow_mask: int = 2
inner_clip: int = 15
gamma: float = 1.8
```

Tip: run `python let-me-cook-5.py` to use the default values, then adjust the arguments by editing the final call to `image_edit_background_and_face_mask(...)` in the file if you want to try other combinations.

### Troubleshooting
- All-black mask => no editable area; check your options and `debug_final_mask_*.png`.
- No generation => make sure `STABILITY_API_KEY` is loaded (visible in your shell) and the API is responsive.
- `dotenv` import issues in the IDE => select the `.venv` interpreter.

## Let me cook scenario (`let-me-cook-4.py`)

This demo-oriented script combines background generation, optional face replacement, and targeted protections (project mask and white polo protection). It automatically creates a `results/<timestamp>/` folder with images and logs.

### Run the example
```bash
source .venv/bin/activate
python let-me-cook-4.py
```

By default it uses:
- **source image**: `images-src/let-me-cook.png`
- **project mask**: `mask/tshirt-1000.png`
- **output**: a file `Cooked-Result_<timestamp>.png` in `results/<timestamp>/`

### Mask granularity: grow and feather

- **grow_mask**: expands the white (editable) area server‑side before inpainting.
  - **Effect**: fills tiny gaps and helps avoid dark rims when the boundary is too tight.
  - **Script default**: `grow_mask=2` pixels. Increase if seams are still visible.

- **feather**: softens the transition at mask edges client‑side without graying the protected area.
  - Implemented by `feather_mask(mask, radius, inner_clip, gamma)`:
    - **radius**: Gaussian blur strength (e.g., `8`).
    - **inner_clip**: clips low values near 0 back to 0 to keep protected regions fully black (prevents a halo on the protected side).
    - **gamma**: transition curve on the editable side (e.g., `1.8`) for a crisper falloff.
  - **Effect**: cleaner seam between the edit and the untouched area.

These two knobs are complementary: start with moderate feather (radius ~8), then slightly increase `grow_mask` if an edge persists.

#### Hard edge (no fading)

If you want a hard transition with no feathering at the mask boundary:
- Set `feather_radius = 0` (feathering is skipped automatically when radius <= 0)
- Set `inner_clip = 0`
- Set `gamma = 1.0`
- Set `grow_mask = 0` to avoid server-side expansion of the editable area

Example call:
```python
image_edit_background_and_face_mask(
    feather_radius=0,
    inner_clip=0,
    gamma=1.0,
    grow_mask=0,
)
```

### Before / After

Input:

![input let-me-cook](images-src/let-me-cook.png)

Result (example from the repo):

![result morne](results/Let%20me%20Cook%203/Cooked-Result_20250818_194123.png)

### License
Internal/demo usage. Check Stability AI's Terms of Use for usage and content distribution.


