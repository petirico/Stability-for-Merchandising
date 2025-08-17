import requests
import base64
from PIL import Image, ImageDraw
import io
import os
from typing import Optional, Tuple
import cv2
import numpy as np
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from pathlib import Path
import shutil
import json

# Load env variables from .env if present
load_dotenv(find_dotenv(usecwd=True), override=False)

# Defaults: ensure all tests/examples use these paths
DEFAULT_IMAGE_PATH = os.getenv("EDIT_IMAGE_PATH", "men.png")
DEFAULT_MASK_PATH = os.getenv("EDIT_MASK_PATH", "mask.png")
RESULTS_ROOT = os.getenv("RESULTS_DIR", "results")

class StabilityImageEditor:
    """
    A class to handle image editing using Stability AI's inpainting API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the editor with your Stability AI API key.
        
        Args:
            api_key: Your Stability AI API key. If None, will read from
                environment variable `STABILITY_API_KEY`.
        """
        if api_key is None:
            api_key = os.getenv("STABILITY_API_KEY")
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "Missing Stability API key. Set STABILITY_API_KEY in the environment or pass api_key explicitly."
            )
        self.base_url = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
        self.headers = {
            "authorization": f"Bearer {self.api_key}",
            "accept": "image/*"
        }
    
    def image_to_base64(self, image_path: str) -> str:
        """
        Convert an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def pil_to_base64(self, pil_image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Base64 encoded string of the image
        """
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def pil_to_png_bytes(self, pil_image: Image.Image) -> bytes:
        """Convert PIL Image to raw PNG bytes (for multipart uploads)."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return buffered.getvalue()

    def load_mask_from_path(self, mask_path: str) -> Image.Image:
        """
        Load a mask from disk, ensuring single-channel 'L' format.
        Black (0) areas will be protected; white (255) areas will be edited.
        """
        return Image.open(mask_path).convert('L')

    @staticmethod
    def timestamped_filename(path: str) -> str:
        """Append _YYYYMMDD_HHMMSS before extension to keep versions."""
        base, ext = os.path.splitext(path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}{ext}"

    @staticmethod
    def create_run_directory(prefix: str = "run") -> str:
        """Create results/<timestamp> directory and return its path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(RESULTS_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / f"{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return str(run_dir)

    @staticmethod
    def write_log(run_dir: str, info: dict) -> None:
        """Write a human-readable log and a json copy with all parameters."""
        path_txt = Path(run_dir) / "log.txt"
        path_json = Path(run_dir) / "log.json"
        lines = []
        for k, v in info.items():
            lines.append(f"{k}: {v}\n")
        with open(path_txt, "a", encoding="utf-8") as f:
            f.writelines(lines)
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def detect_white_polo_region(self, image_path: str) -> Optional[np.ndarray]:
        """
        Heuristic protection mask for a white polo on upper body.
        Returns a uint8 array (255 = protect) or None if not found.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        height, width = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # White: low saturation, high value
        lower = np.array([0, 0, 200], dtype=np.uint8)
        upper = np.array([179, 40, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower, upper)

        # Focus on upper-body region
        top = int(height * 0.35)
        bottom = int(height * 0.80)
        region = np.zeros((height, width), dtype=np.uint8)
        region[top:bottom, :] = 255
        white_mask = cv2.bitwise_and(white_mask, region)

        # Morphological cleanup
        kernel = np.ones((9, 9), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Keep largest component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask)
        if num_labels <= 1:
            return None
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        keep = (labels == largest_idx).astype('uint8') * 255
        return keep
    
    def create_inverse_mask(self, image_path: str, protected_areas: list) -> Image.Image:
        """
        Create a mask where specified areas are protected (black) and rest is white.
        Protected areas won't be modified.
        
        Args:
            image_path: Path to the original image
            protected_areas: List of tuples (x1, y1, x2, y2) defining rectangles to protect
            
        Returns:
            PIL Image mask
        """
        # Open the image to get dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Create white mask (everything editable by default)
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        # Draw black rectangles for protected areas
        for area in protected_areas:
            x1, y1, x2, y2 = area
            draw.rectangle([x1, y1, x2, y2], fill=0)
        
        return mask
    
    def detect_and_create_face_mask(self, image_path: str, expand_ratio: float = 1.2) -> Optional[Image.Image]:
        """
        Detect faces in the image and create a mask for face replacement.
        
        Args:
            image_path: Path to the image
            expand_ratio: Ratio to expand the face bounding box
            
        Returns:
            PIL Image mask with face area white (editable) or None if no face detected
        """
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return None
        
        # Create black mask
        height, width = img.shape[:2]
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw white areas for each detected face
        for (x, y, w, h) in faces:
            # Expand the face area slightly
            expand_w = int(w * (expand_ratio - 1) / 2)
            expand_h = int(h * (expand_ratio - 1) / 2)
            
            x1 = max(0, x - expand_w)
            y1 = max(0, y - expand_h)
            x2 = min(width, x + w + expand_w)
            y2 = min(height, y + h + expand_h)
            
            # Draw ellipse for more natural face mask
            draw.ellipse([x1, y1, x2, y2], fill=255)
        
        return mask
    
    def create_background_mask(self, image_path: str, use_grabcut: bool = True) -> Image.Image:
        """
        Create a mask for background replacement using GrabCut or simple edge detection.
        
        Args:
            image_path: Path to the image
            use_grabcut: Whether to use GrabCut algorithm (more accurate but slower)
            
        Returns:
            PIL Image mask with background white (editable)
        """
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        if use_grabcut:
            # Use GrabCut for better foreground/background separation
            mask = np.zeros((height, width), np.uint8)
            
            # Initialize rectangle for GrabCut (center 80% of image assumed as possible foreground)
            rect_x = int(width * 0.1)
            rect_y = int(height * 0.1)
            rect_w = int(width * 0.8)
            rect_h = int(height * 0.8)
            rect = (rect_x, rect_y, rect_w, rect_h)
            
            # Apply GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask (0 for foreground, 255 for background)
            mask2 = np.where((mask == 2) | (mask == 0), 255, 0).astype('uint8')
        else:
            # Simple edge-based approach
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to create foreground area
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Invert to make background white
            mask2 = cv2.bitwise_not(dilated)
        
        return Image.fromarray(mask2)
    
    def edit_image(self, 
                   image_path: str,
                   mask: Image.Image,
                   prompt: str,
                   output_path: str = "edited_image.png",
                   negative_prompt: Optional[str] = None,
                   seed: int = 0,
                   output_format: str = "png",
                   run_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        Send image editing request to Stability AI API.
        
        Args:
            image_path: Path to the original image
            mask: PIL Image mask (white areas will be edited)
            prompt: Text prompt for what to generate in masked areas
            output_path: Path to save the edited image
            negative_prompt: What not to include in the generation
            seed: Random seed for reproducibility
            output_format: Output format (png or webp)
            
        Returns:
            (success, saved_path)
        """
        # Prepare the multipart form data
        files = {
            "image": ("image.png", open(image_path, "rb"), "image/png"),
            # Send raw PNG bytes for the mask
            "mask": ("mask.png", io.BytesIO(self.pil_to_png_bytes(mask)), "image/png"),
        }
        
        data = {
            "prompt": prompt,
            "output_format": output_format,
            "seed": str(seed)
        }
        
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        
        # Make the API request
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                # Save the edited image
                target_dir = Path(run_dir) if run_dir else Path(".")
                target_dir.mkdir(parents=True, exist_ok=True)
                stamped_name = Path(self.timestamped_filename(output_path)).name
                stamped_path = target_dir / stamped_name
                with open(stamped_path, "wb") as f:
                    f.write(response.content)
                print(f"Edited image saved to {stamped_path}")
                return True, str(stamped_path)
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return False, ""
                
        except Exception as e:
            print(f"Exception occurred: {e}")
            return False, ""


# Example usage functions
def example_protect_areas():
    """
    Example: Edit `men.png` while respecting protection areas from `mask.png`.
    """
    editor = StabilityImageEditor()
    
    # Load default mask from project root (black = protect, white = edit)
    mask = editor.load_mask_from_path(DEFAULT_MASK_PATH)
    
    # Edit the image
    editor.edit_image(
        image_path=DEFAULT_IMAGE_PATH,
        mask=mask,
        prompt="refine t-shirt logo and typography, high quality",
        output_path="edited_with_protection.png",
        run_dir=StabilityImageEditor.create_run_directory()
    )


def example_replace_face():
    """
    Example adapted: still uses `men.png` and `mask.png` (no detection),
    demonstrating a different prompt while honoring the protection mask.
    """
    editor = StabilityImageEditor()
    mask = editor.load_mask_from_path(DEFAULT_MASK_PATH)
    editor.edit_image(
        image_path=DEFAULT_IMAGE_PATH,
        mask=mask,
        prompt="clean, crisp chest logo text 'BillionnAIre', realistic print",
        negative_prompt="blurry, deformed, extra limbs, bad text",
        output_path="face_replaced.png",
        run_dir=StabilityImageEditor.create_run_directory()
    )


def example_change_background():
    """
    Example adapted: still edits `men.png` using `mask.png` as protection
    while asking for a background change.
    """
    editor = StabilityImageEditor()
    mask = editor.load_mask_from_path(DEFAULT_MASK_PATH)
    editor.edit_image(
        image_path=DEFAULT_IMAGE_PATH,
        mask=mask,
        prompt="clean studio black background for the person, professional lighting",
        negative_prompt="crowds, artifacts, extra text",
        output_path="background_changed.png",
        run_dir=StabilityImageEditor.create_run_directory()
    )


def combined_example():
    """
    Example adapted: demonstrates reuse of `mask.png` with another prompt.
    """
    editor = StabilityImageEditor()
    mask = editor.load_mask_from_path(DEFAULT_MASK_PATH)
    editor.edit_image(
        image_path=DEFAULT_IMAGE_PATH,
        mask=mask,
        prompt="subtle enhancement of shirt print, sharp edges, natural fabric texture",
        output_path="complex_edit.png",
        run_dir=StabilityImageEditor.create_run_directory()
    )


def example_morne_background_and_face(
    include_background_mask: bool = False,
    include_face_mask: bool = False,
    save_debug_mask: bool = True,
    apply_project_protection: bool = True,
    apply_polo_protection: bool = False,
):
    """
    1) Replace background with summit view of Le Morne Mountain (Mauritius)
    2) Replace/enhance face to a handsome, realistic 35-year-old male
    3) Preserve top-body clothing as a WHITE polo (protected)

    Options:
    - include_background_mask: toggle background replacement guidance
    - include_face_mask: toggle face replacement guidance
    - save_debug_mask: save the final combined mask as 'debug_final_mask.png'
    """
    editor = StabilityImageEditor()

    image_path = DEFAULT_IMAGE_PATH
    img = Image.open(image_path)
    width, height = img.size

    # Create run directory and copy inputs
    run_dir = StabilityImageEditor.create_run_directory()
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(image_path, str(Path(run_dir) / Path(image_path).name))
    except Exception:
        pass

    # Editable regions: background + face (optional)
    combined = np.zeros((height, width), dtype=np.uint8)
    any_specific_mask = False

    if include_background_mask:
        bg_mask = editor.create_background_mask(image_path, use_grabcut=True)
        combined = np.maximum(combined, np.array(bg_mask, dtype=np.uint8))
        any_specific_mask = True

    if include_face_mask:
        face_mask = editor.detect_and_create_face_mask(image_path, expand_ratio=1.25)
        if face_mask is not None:
            combined = np.maximum(combined, np.array(face_mask, dtype=np.uint8))
            any_specific_mask = True

    # If no specific masks selected, default to editing the whole image,
    # protections will be applied below (mask.png, polo region)
    if not any_specific_mask:
        combined[:] = 255

    # Protect the white polo (if detected and requested)
    if apply_polo_protection:
        polo_protect = editor.detect_white_polo_region(image_path)
        if polo_protect is not None:
            combined[polo_protect == 255] = 0

    # Apply project protection mask (mask.png: black = protect)
    protection_img = None
    if apply_project_protection:
        protection_img = editor.load_mask_from_path(DEFAULT_MASK_PATH)
        protection = np.array(protection_img, dtype=np.uint8)
        combined[protection == 0] = 0

    final_mask = Image.fromarray(combined)
    if save_debug_mask:
        debug_name = StabilityImageEditor.timestamped_filename("debug_final_mask.png")
        final_mask.save(str(Path(run_dir) / Path(debug_name).name))

    # Save initial mask as well
    if protection_img is not None:
        try:
            protection_img.save(str(Path(run_dir) / "mask_initial.png"))
        except Exception:
            pass

    prompt = (
        "photorealistic summit of Le Morne Brabant, Mauritius, panoramic ocean view, natural daylight; "
        "handsome 35-year-old male model, realistic facial features, balanced skin tones; "
        "keep clothing as a clean white polo shirt"
    )

    success, saved_path = editor.edit_image(
        image_path=image_path,
        mask=final_mask,
        prompt=prompt,
        negative_prompt="crowds, artifacts, extra text",
        output_path="morne_background_and_face.png",
        run_dir=run_dir
    )

    # Write log
    editable_ratio = float(np.count_nonzero(np.array(final_mask)))/float(width*height)
    info = {
        "api_endpoint": editor.base_url,
        "accept": editor.headers.get("accept"),
        "output_format": "png",
        "seed": 0,
        "negative_prompt": "crowds, artifacts, extra text",
        "prompt": prompt,
        "image_path": image_path,
        "mask_from": DEFAULT_MASK_PATH,
        "include_background_mask": include_background_mask,
        "include_face_mask": include_face_mask,
        "apply_project_protection": apply_project_protection,
        "apply_polo_protection": apply_polo_protection,
        "save_debug_mask": save_debug_mask,
        "editable_ratio": round(editable_ratio, 6),
        "saved_result": saved_path if success else "",
    }
    StabilityImageEditor.write_log(run_dir, info)


if __name__ == "__main__":
    # Remember to install required packages:
    # pip install requests pillow opencv-python numpy
    print("Stability AI Image Editor initialized")
    print("Using default image:", DEFAULT_IMAGE_PATH)
    print("Using protection mask:", DEFAULT_MASK_PATH)
    if not os.getenv("STABILITY_API_KEY"):
        print("Warning: STABILITY_API_KEY not set in environment.")
        print("Export your key, e.g.: export STABILITY_API_KEY=\"YOUR_KEY\"")
    
    # Uncomment any example to run
    # example_protect_areas()
    # example_replace_face()
    # example_change_background()
    # combined_example()
    example_morne_background_and_face()