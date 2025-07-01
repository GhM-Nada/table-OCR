# Required imports for PDF processing, image manipulation, and OCR
import os
import cv2
import numpy as np
import pdf2image

from PIL import Image
from deskew import determine_skew
from scipy import ndimage
from tqdm import tqdm
import logging
# YOLO model for table detection
from ultralytics import YOLO

# Configure logging for tracking progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    PDF processing pipeline for OCR preparation.
    This class handles:
    1. PDF to image conversion
    2. Image quality enhancement
    3. Deskewing and denoising
    4. Table detection and extraction
    5. In-memory processing without file saving
    """
    def __init__(self, input_dir, output_dir=None, dpi=300):
        """
        Initialize the PDF processor with input/output directories and processing parameters.
        
        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str): Directory for saving processed images and box files (optional for in-memory processing)
            dpi (int): DPI for PDF to image conversion (higher DPI = better quality)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dpi = dpi

        # Create directory structure for outputs only if output_dir is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)  # Original images
            os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)  # Extracted tables
            os.makedirs(os.path.join(output_dir, "box_files"), exist_ok=True)  # Tesseract box files

        # Initialize YOLO model for table detection
        self.yolo_model = YOLO('./src/models/best.pt')
        self.yolo_model.overrides['conf'] = 0.25  # Minimum confidence for detection
        self.yolo_model.overrides['iou'] = 0.45   # IoU threshold for non-maximum suppression
        self.yolo_model.overrides['agnostic_nms'] = False
        self.yolo_model.overrides['max_det'] = 1000

    def convert_pdf_to_images(self, pdf_path):
        """
        Convert PDF pages to high-quality images.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of PIL Image objects, one per page
        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=self.dpi)
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []

    def correct_skew(self, image):
        """
        Detect and correct skewed images using angle detection and rotation.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            numpy.ndarray: Deskewed image
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Detect skew angle
        angle = determine_skew(gray)

        # Rotate if angle is significant
        if abs(angle) > 0.1:
            rotated = ndimage.rotate(image_np, angle, reshape=True, mode='constant', cval=255)
            return rotated

        return image_np

    def remove_noise(self, image):
        """
        Apply noise reduction using Gaussian and median filtering.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            numpy.ndarray: Denoised image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply noise reduction filters
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Gaussian blur for noise reduction
        denoised = cv2.medianBlur(blurred, 3)  # Median blur for salt-and-pepper noise
        return denoised

    def apply_thresholding(self, image):
        """
        Convert grayscale image to binary using adaptive thresholding.
        
        Args:
            image: Input grayscale image
            
        Returns:
            numpy.ndarray: Binary image
        """
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def thinning_skeletonization(self, image):
        """
        Optimize text for OCR by performing skeletonization.
        
        Args:
            image: Input binary image
            
        Returns:
            numpy.ndarray: Skeletonized image
        """
        inverted = cv2.bitwise_not(image)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(inverted, kernel, iterations=1)
        thinned = cv2.dilate(eroded, kernel, iterations=1)
        result = cv2.bitwise_not(thinned)
        return result

    def scale_image(self, image, scale_factor=1.5):
        """
        Scale image to improve OCR accuracy.
        
        Args:
            image: Input image
            scale_factor (float): Scaling factor (default: 1.5)
            
        Returns:
            numpy.ndarray: Scaled image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def crop_table_region(self, image_np, page_name="page"):
        """
        Detect and crop tables from images using YOLO.
        
        Args:
            image_np: Input image as numpy array
            page_name (str): Name of the page for logging
            
        Returns:
            list: List of cropped table images
        """
        # Detect tables using YOLO
        results = self.yolo_model.predict(image_np)
        boxes = results[0].boxes.data.numpy()
        
        padding = 2  # Padding around detected tables
        cropped_images = []

        if boxes.shape[0] == 0:
            logger.warning(f"No tables detected on {page_name}. Returning full image.")
            return [Image.fromarray(image_np)]

        # Keep only the highest confidence detection
        if boxes.shape[0] > 0:
            best_box_idx = np.argmax(boxes[:, 4])
            boxes = boxes[best_box_idx:best_box_idx+1]
            logger.info(f"Keeping only the highest confidence table detection on {page_name}.")
        
        # Process the best detection
        x1, y1, x2, y2, _, _ = map(int, boxes[0])
        
        # Apply padding while staying within image boundaries
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_np.shape[1], x2 + padding)
        y2 = min(image_np.shape[0], y2 + padding)
        
        cropped = image_np[y1:y2, x1:x2]
        if cropped.size > 0:
            cropped_pil = Image.fromarray(cropped)
            cropped_images.append(cropped_pil)
            logger.info(f"Cropped table on {page_name} with padding: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        if not cropped_images:
            logger.warning(f"No valid cropped regions on {page_name}. Returning full image.")
            return [Image.fromarray(image_np)]
            
        return cropped_images

    def preprocess_image(self, image, scale_factor=1.5):
        """
        Apply complete preprocessing pipeline to an image.
        
        Args:
            image: Input image
            scale_factor (float): Scaling factor for image enlargement
            
        Returns:
            numpy.ndarray: Preprocessed image ready for OCR
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply preprocessing steps
        corrected = self.correct_skew(gray)  # Fix skewed images
        denoised = self.remove_noise(corrected)  # Remove noise
        binary = self.apply_thresholding(denoised)  # Convert to binary
        thinned = self.thinning_skeletonization(binary)  # Optimize for OCR
        scaled = self.scale_image(thinned, scale_factor)  # Scale up for better recognition
        return scaled

    def process_pdf_in_memory(self, pdf_path):
        """
        Process a single PDF file and return the first preprocessed table image in memory.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            numpy.ndarray or PIL.Image: Preprocessed table image, or None if no table found
        """
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        logger.info(f"Processing PDF in memory: {pdf_name}")
        
        # Convert PDF to images
        images = self.convert_pdf_to_images(pdf_path)
        
        if not images:
            logger.error(f"Failed to convert PDF to images: {pdf_path}")
            return None

        # Process each page until we find a table
        for i, image in enumerate(images):
            page_name = f"{pdf_name}_page_{i+1}"
            logger.info(f"Processing page: {page_name}")

            # Convert to RGB for YOLO detection
            image_np = np.array(image.convert("RGB"))

            # Detect and crop tables
            cropped_tables = self.crop_table_region(image_np, page_name=page_name)

            # Process the first detected table
            if cropped_tables:
                cropped_img = cropped_tables[0]
                logger.info(f"Processing table from {page_name}")

                # Preprocess the cropped table
                preprocessed_crop = self.preprocess_image(cropped_img)
                
                logger.info(f"Successfully preprocessed table from {page_name}")
                return preprocessed_crop

        logger.warning(f"No tables found in PDF: {pdf_path}")
        return None

    def process_pdf(self, pdf_path):
        """
        Process a single PDF file through the complete pipeline (original method with file saving).
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        if not self.output_dir:
            logger.error("Output directory not specified. Use process_pdf_in_memory() for in-memory processing.")
            return
            
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        logger.info(f"Processing PDF: {pdf_name}")
        
        # Convert PDF to images
        images = self.convert_pdf_to_images(pdf_path)

        # Process each page
        for i, image in enumerate(images):
            page_name = f"{pdf_name}_page_{i+1}"
            image_path = os.path.join(self.output_dir, "images", f"{page_name}.png")
            image.save(image_path)
            logger.info(f"Saved original image: {image_path}")

            # Convert to RGB for YOLO detection
            image_np = np.array(image.convert("RGB"))

            # Detect and crop tables
            cropped_tables = self.crop_table_region(image_np, page_name=page_name)

            # Process each detected table
            for j, cropped_img in enumerate(cropped_tables):
                out_name = f"{page_name}_table_{j+1}.png"
                tables_path = os.path.join(self.output_dir, "tables", out_name)

                # Preprocess the cropped table
                preprocessed_crop = self.preprocess_image(cropped_img)

                # Save preprocessed result
                if isinstance(preprocessed_crop, np.ndarray):
                    cv2.imwrite(tables_path, preprocessed_crop)
                else:
                    preprocessed_crop.save(tables_path)

                logger.info(f"Saved preprocessed table image: {tables_path}")

    def process_all_pdfs(self):
        """
        Process all PDF files in the input directory.
        """
        if not self.output_dir:
            logger.error("Output directory not specified for batch processing.")
            return
            
        pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.input_dir, pdf_file)
            self.process_pdf(pdf_path)

        logger.info("All PDFs processed successfully")