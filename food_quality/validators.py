import os
import uuid
from pathlib import Path
from PIL import Image
from werkzeug.utils import secure_filename
from food_quality.config import Config

class ValidationError(Exception):
    """Custom exception for file and request validation errors."""
    pass

def allowed_file(filename: str) -> bool:
    """Check if the filename has an allowed extension."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS

def generate_safe_filename(original_filename: str) -> str:
    """Generate a collision-proof unique filename with a validated extension."""
    sanitized = secure_filename(original_filename)
    if not sanitized or '.' not in sanitized:
        ext = 'jpg'
    else:
        ext = sanitized.rsplit('.', 1)[1].lower()
        if ext not in Config.ALLOWED_EXTENSIONS:
            ext = 'jpg'
    
    unique_id = uuid.uuid4().hex
    return f"{unique_id}.{ext}"

def validate_and_process_upload(file_storage) -> str:
    """
    Validate uploaded file:
    1. Check presence and non-empty filename
    2. Check file extension
    3. Read content and verify header integrity via PIL
    4. Save to upload directory safely
    Returns: generated safe filename
    """
    if not file_storage or file_storage.filename == '':
        raise ValidationError("No image file was provided or selected.")

    if not allowed_file(file_storage.filename):
        allowed = ", ".join(sorted(list(Config.ALLOWED_EXTENSIONS))).upper()
        raise ValidationError(f"Unsupported file format. Allowed formats are: {allowed}.")

    # Generate a unique safe filename
    safe_filename = generate_safe_filename(file_storage.filename)
    upload_path = Config.UPLOAD_FOLDER / safe_filename

    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    # Save to disk first then inspect content
    file_storage.save(str(upload_path))

    # Inspect the actual image content using Pillow
    try:
        with Image.open(upload_path) as img:
            img.verify()  # Verifies file integrity
            img_format = (img.format or '').lower()
            valid_formats = {'jpeg', 'png', 'webp'}
            if img_format not in valid_formats:
                # Remove invalid file
                if upload_path.exists():
                    os.remove(upload_path)
                raise ValidationError(f"Invalid image format ({img_format}). Only JPEG, PNG, and WebP are accepted.")
    except Exception as e:
        if upload_path.exists():
            os.remove(upload_path)
        raise ValidationError(f"The uploaded file is corrupt or not a valid image: {str(e)}")

    return safe_filename

def safe_delete_upload(filename: str) -> bool:
    """Safely delete an uploaded image file without path traversal vulnerabilities."""
    if not filename:
        return False
    
    # Strip any directory path components
    safe_name = os.path.basename(filename)
    target_path = (Config.UPLOAD_FOLDER / safe_name).resolve()
    upload_dir_resolved = Config.UPLOAD_FOLDER.resolve()

    # Guard against path traversal
    if not str(target_path).startswith(str(upload_dir_resolved)):
        return False

    if target_path.exists() and target_path.is_file():
        try:
            os.remove(target_path)
            return True
        except OSError:
            return False
    return False
