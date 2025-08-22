from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Load the image
image_path = r"img/7.jpg"
image = Image.open(image_path)

# Enhance the edges to make fold lines more obvious
# Convert to grayscale
image_gray = image.convert("L")

# Apply edge enhancement filter
image_edges = image_gray.filter(ImageFilter.FIND_EDGES)

# Enhance contrast
enhancer = ImageEnhance.Contrast(image_edges)
image_enhanced = enhancer.enhance(3)  # Increase contrast by a factor of 3

# Convert back to RGB
final_image = image_enhanced.convert("RGB")

# Save the processed image
final_image_path = "7_enhanced.jpg"
final_image.save(final_image_path)
