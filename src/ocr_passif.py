import argparse
import os
import sys
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import traceback
from datetime import datetime
import re
from difflib import SequenceMatcher

# Import your table extractor module
from table_creator.table_extractor import TableExtraction
from preprocessing import PDFProcessor

# Predefined structure for PASSIF balance sheet
BALANCE_SHEET_STRUCTURE_PASSIF = {
    "columns": [
        "PASSIF",
        "N", 
        "N-1"
    ],
    "sections": {
        "CAPITAUX PROPRES": [
            "Capital émis",
            "Capital non appelé",
            "Primes et réserves- Réserves consolidées",
            "Ecarts de réévaluation", 
            "Ecart d'équivalence",
            "Résultat net - Résultat net part du groupe",
            "Autres capitaux propres I Report à nouveau",
            "Part de la société consolidante",
            "Part des minoritaires",
            "TOTAL I"
        ],
        "PASSIFS NON-COURANTS": [
            "Emprunts et dettes financières",
            "Impôts (différés et provisionnés)",
            "Autres dettes non courantes", 
            "Provisions et produits constatés d'avance",
            "TOTAL II"
        ],
        "PASSIFS COURANTS": [
            "Fournisseurs et comptes rattachés",
            "Impôts",
            "Autres dettes",
            "Trésorerie Passif", 
            "TOTAL III",
            "TOTAL PASSIF (I+II+III)"
        ]
    }
}

def clean_number_value(value):
    """Clean numerical values for PASSIF data"""
    if value is None or pd.isna(value):
        return [None]
    
    value_str = str(value).strip()
    
    if not value_str or value_str.lower() == 'nan':
        return [None]
    
    # Check if it looks like a date
    date_pattern = r'\d+\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)'
    if re.search(date_pattern, value_str.lower()):
        return [None]
    
    # Replace C and O with 0
    value_str = value_str.replace('C', '0').replace('O', '0')
    
    # Split by various patterns
    split_patterns = ['[', ']', '\n', '\r']
    parts = [value_str]
    for pattern in split_patterns:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(pattern))
        parts = new_parts
    
    cleaned_parts = []
    for part in parts:
        if not part.strip():
            continue
        
        # Remove unwanted characters, keep spaces for French number format
        cleaned_part = re.sub(r'[\]\[\\/\.]', '', part)
        cleaned_part = re.sub(r'[^0-9,\s]', '', cleaned_part)
        
        if cleaned_part and re.match(r'^[0-9,\s]+$', cleaned_part):
            cleaned_part = cleaned_part.strip(',')
            if cleaned_part:
                cleaned_parts.append(cleaned_part)
    
    return cleaned_parts if cleaned_parts else [None]

def convert_to_numeric(value):
    """Convert cleaned string value to numeric, handling French number format"""
    if value is None or value == '' or str(value).lower() == 'nan':
        return None
    try:
        # Handle French number format (spaces as thousand separators)
        numeric_str = str(value).replace(' ', '').replace(',', '')
        return float(numeric_str)
    except (ValueError, TypeError):
        return None

def apply_number_cleaning_to_dataframe_passif(df):
    """Apply number cleaning to PASSIF DataFrame (3 columns)"""
    if df is None or df.empty:
        return df
    
    cleaned_df = df.copy()
    
    # Clean only the numerical columns (columns 1 and 2 for N and N-1)
    for col_idx in range(1, min(3, len(cleaned_df.columns))):
        for row_idx in range(len(cleaned_df)):
            value = cleaned_df.iloc[row_idx, col_idx]
            cleaned_values = clean_number_value(value)
            cleaned_df.iloc[row_idx, col_idx] = cleaned_values[0] if cleaned_values[0] is not None else None
    
    return cleaned_df

def similarity_score(a, b):
    """Calculate similarity between two strings after removing all spaces"""
    a = re.sub(r'\s+', '', str(a).lower().strip())
    b = re.sub(r'\s+', '', str(b).lower().strip())
    
    if not a or not b:
        return 0
    return SequenceMatcher(None, a, b).ratio()

def clean_text(text):
    """Clean text for better matching"""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.lower().strip())

def find_best_match(target_text, candidate_texts):
    """Find the best matching text from a list of candidates"""
    if not target_text or not candidate_texts:
        return -1, 0
    
    target_text = clean_text(target_text)
    best_score = 0
    best_idx = -1
    
    for idx, candidate in enumerate(candidate_texts):
        candidate = clean_text(candidate)
        score = similarity_score(target_text, candidate)
        if score > best_score and score > 0.3:
            best_score = score
            best_idx = idx
    
    return best_idx, best_score

def find_year_or_n_columns(raw_df):
    """
    Enhanced year/N column detection with:
    - Year range 2000-2050
    - Special character cleaning
    - Adjacent column checking for None values
    """
    n_col = -1
    n1_col = -1
    found_year = None
    
    def clean_year_text(text):
        """Clean text and extract year if present"""
        if not text or pd.isna(text):
            return None
        
        # Remove special characters and clean
        cleaned = re.sub(r'[^\d\w\s]', '', str(text).strip())
        
        # Look for 4-digit numbers in the range 2000-2050
        year_matches = re.findall(r'\b(20[0-4][0-9]|2050)\b', cleaned)
        
        if year_matches:
            try:
                year = int(year_matches[0])
                if 2000 <= year <= 2050:
                    return year
            except ValueError:
                pass
        
        # Check for N or N-1 patterns
        cleaned_upper = cleaned.upper()
        if cleaned_upper == 'N':
            return 'N'
        elif 'N-1' in cleaned_upper or 'N1' in cleaned_upper:
            return 'N-1'
            
        return None

    print("\n----- Scanning for Year/N Columns -----")
    
    # Scan first 5 rows for year/N indicators
    year_columns = []
    
    for row_idx in range(min(5, len(raw_df))):
        for col_idx in range(len(raw_df.columns)):
            cell_val = raw_df.iloc[row_idx, col_idx]
            year_info = clean_year_text(cell_val)
            
            if year_info:
                year_columns.append((col_idx, year_info, row_idx))
                print(f"Found '{year_info}' in column {col_idx}, row {row_idx}: '{cell_val}'")
    
    # Sort by column index for consistent processing
    year_columns.sort(key=lambda x: x[0])
    
    # Step 1: Look for N and N-1 pairs first
    for col_idx, year_info, row_idx in year_columns:
        if year_info == 'N' and n_col == -1:
            n_col = col_idx
            print(f"Assigned 'N' to column {n_col}")
            
            # Look for N-1 in nearby columns
            for other_col, other_info, other_row in year_columns:
                if other_info == 'N-1' and other_col != col_idx:
                    n1_col = other_col
                    print(f"Assigned 'N-1' to column {n1_col}")
                    break
            break
    
    # Step 2: If no N/N-1 found, look for year pairs
    if n_col == -1:
        year_pairs = [(col, year, row) for col, year, row in year_columns if isinstance(year, int)]
        year_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by year descending
        
        if len(year_pairs) >= 2:
            # Take the two most recent years
            current_year_col = year_pairs[0]
            prev_year_col = year_pairs[1]
            
            n_col = current_year_col[0]
            n1_col = prev_year_col[0]
            found_year = current_year_col[1]
            
            print(f"Assigned year {current_year_col[1]} to column {n_col}")
            print(f"Assigned year {prev_year_col[1]} to column {n1_col}")
        
        elif len(year_pairs) == 1:
            # Only one year found, use it as N and find adjacent column
            n_col = year_pairs[0][0]
            found_year = year_pairs[0][1]
            print(f"Assigned year {found_year} to column {n_col}")
            
            # Look for adjacent column with data
            for adjacent_col in [n_col + 1, n_col - 1]:
                if 0 <= adjacent_col < len(raw_df.columns) and adjacent_col != n_col:
                    if has_meaningful_data(raw_df, adjacent_col):
                        n1_col = adjacent_col
                        print(f"Assigned adjacent column {n1_col} as N-1")
                        break
    
    # Step 3: Enhanced fallback with adjacent column checking
    if n_col == -1 or n1_col == -1:
        print("Applying enhanced fallback logic...")
        
        # Find columns with meaningful numerical data
        data_columns = []
        for col_idx in range(len(raw_df.columns)):
            if has_meaningful_data(raw_df, col_idx):
                data_columns.append(col_idx)
        
        print(f"Found {len(data_columns)} columns with meaningful data: {data_columns}")
        
        if len(data_columns) >= 2:
            if n_col == -1:
                n_col = data_columns[-2]  # Second to last column with data
                print(f"Fallback: Using column {n_col} for N")
            
            if n1_col == -1:
                n1_col = data_columns[-1]  # Last column with data
                print(f"Fallback: Using column {n1_col} for N-1")
    
    return n_col, n1_col, found_year

def has_meaningful_data(df, col_idx):
    """Check if a column contains meaningful numerical data"""
    if col_idx < 0 or col_idx >= len(df.columns):
        return False
    
    meaningful_count = 0
    total_count = 0
    
    for row_idx in range(len(df)):
        val = df.iloc[row_idx, col_idx]
        total_count += 1
        
        if val is not None and str(val).strip() and str(val).strip().lower() != 'nan':
            # Check if it contains digits (potential numerical data)
            if any(c.isdigit() for c in str(val)):
                # Skip if it looks like a year in wrong context
                cleaned_val = re.sub(r'[^\d]', '', str(val))
                if cleaned_val and not (2000 <= int(cleaned_val[:4]) <= 2050 if len(cleaned_val) >= 4 else False):
                    meaningful_count += 1
    
    # Column has meaningful data if at least 20% of cells contain numerical values
    return meaningful_count > 0 and (meaningful_count / total_count) >= 0.2

def extract_value_from_column_enhanced(row, col_idx):
    """Enhanced value extraction with adjacent column checking and simple text filtering"""
    if col_idx < 0 or col_idx >= len(row):
        return None
    
    # Try the primary column first
    val = extract_value_from_column(row, col_idx)  # This now includes text filtering
    if val is not None:
        return val
    
    # If primary column is empty or filtered out, check adjacent columns
    for adjacent_offset in [-1, 1]:
        adjacent_col = col_idx + adjacent_offset
        if 0 <= adjacent_col < len(row):
            adj_val = extract_value_from_column(row, adjacent_col)  # This also includes text filtering
            if adj_val is not None:
                print(f"Using adjacent column {adjacent_col} value instead of column {col_idx}")
                return adj_val
    
    return None



def extract_value_from_column(row, col_idx):
    """
    Extract a value from a specific column in a row
    Simple logic: only extract if it's mostly numbers, reject if too much text
    """
    if col_idx < 0 or col_idx >= len(row):
        return None
    
    val = row[col_idx]
    if val is None or not str(val).strip() or str(val).strip().lower() == 'nan':
        return None
    
    val_str = str(val).strip()
    
    # Simple text filter: reject if it contains too much text
    if is_too_much_text(val_str):
        return None
    
    # Only accept if it contains digits
    if isinstance(val, (int, float)) or (isinstance(val, str) and any(c.isdigit() for c in val)):
        return val
    
    return None

def is_too_much_text(text):
    """
    Simple logic to detect if a value contains too much text
    Returns True if the value should be rejected
    """
    if not text or len(text.strip()) == 0:
        return True
    
    text = text.strip()
    
    # Rule 1: If it has more than 3 words, it's probably descriptive text
    words = text.split()
    if len(words) > 3:
        return True
    
    # Rule 2: If it contains common French words, it's probably text
    french_words = ['de', 'la', 'le', 'du', 'des', 'et', 'ou', 'part', 'société', 'societe', 
                    'consolidante', 'groupe', 'capital', 'résultat', 'resultat', 'total']
    text_lower = text.lower()
    for word in french_words:
        if word in text_lower:
            return True
    
    # Rule 3: If it has more letters than numbers, it's probably text
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    
    if letters > digits and letters > 2:  # More than 2 letters and more letters than digits
        return True
    
    # Rule 4: If it contains parentheses with numbers like (1), it's probably a footnote
    if re.search(r'\([0-9]+\)', text):
        return True
    
    return False


def create_structured_passif_table(raw_df, structure=None):
    """Create structured PASSIF table with enhanced column detection"""
    if raw_df is None or raw_df.empty or not structure:
        return pd.DataFrame(columns=structure["columns"] if structure else ["PASSIF", "N", "N-1"])

    # Get all structured rows
    all_structured_rows = []
    for section, rows in structure["sections"].items():
        all_structured_rows.extend(rows)

    print("\n----- Enhanced PASSIF Column Detection -----")
    
    # Use enhanced year detection
    n_col, n1_col, found_year = find_year_or_n_columns(raw_df)
    
    print(f"Final column assignments:")
    print(f"  N column: {n_col}")
    print(f"  N-1 column: {n1_col}")
    if found_year:
        print(f"  Detected year: {found_year}")

    print(f"Using column {n_col} for current year values")
    print(f"Using column {n1_col} for previous year values")

    # Extract descriptions for matching
    print("\n----- Matching PASSIF Row Descriptions -----")
    raw_descriptions = []
    for idx, row in raw_df.iterrows():
        for col_idx in range(min(n_col if n_col > 0 else 2, len(row))):
            val = row[col_idx]
            if val is not None and str(val).strip() and str(val).strip().lower() != 'nan':
                if not str(val).replace(' ', '').replace('-', '').isdigit():
                    raw_descriptions.append((idx, val))
                    break

    print(f"Found {len(raw_descriptions)} raw descriptions:")
    for i, (idx, desc) in enumerate(raw_descriptions):
        print(f"  Row {idx}: '{str(desc)[:60]}...'")

    # Step 1: Find all rows containing 'Total' (case insensitive)
    total_rows = []
    for idx, desc in raw_descriptions:
        if 'total' in str(desc).lower():
            total_rows.append((idx, desc))
    
    print(f"\nFound {len(total_rows)} rows containing 'Total':")
    for i, (idx, desc) in enumerate(total_rows):
        print(f"  {i+1}. Row {idx}: '{desc}'")

    # Step 2: Sequential TOTAL assignment
    assigned_matches = {}
    used_raw_rows = set()
    
    # Assign TOTAL I, II, III sequentially
    total_structured_rows = ['TOTAL I', 'TOTAL II', 'TOTAL III']
    for i, total_structured in enumerate(total_structured_rows):
        if i < len(total_rows):
            raw_idx, raw_desc = total_rows[i]
            assigned_matches[total_structured] = raw_idx
            used_raw_rows.add(raw_idx)
            print(f"Sequential matching: '{total_structured}' -> Row {raw_idx}: '{raw_desc}'")

    # Step 3: Match non-TOTAL rows using similarity
    non_total_structured = [row for row in all_structured_rows 
                           if not any(total in row for total in ['TOTAL I', 'TOTAL II', 'TOTAL III', 'TOTAL PASSIF'])]
    
    # Get non-total raw descriptions (excluding already used rows)
    non_total_raw = [(idx, desc) for idx, desc in raw_descriptions 
                     if idx not in used_raw_rows and 'total' not in str(desc).lower()]

    print(f"\nMatching {len(non_total_structured)} non-TOTAL structured rows with {len(non_total_raw)} available raw rows:")

    # Match non-TOTAL rows
    row_matches = []
    for structured_row_desc in non_total_structured:
        candidate_texts = [desc for _, desc in non_total_raw]
        if candidate_texts:
            best_idx, score = find_best_match(structured_row_desc, candidate_texts)
            if best_idx >= 0 and score > 0.4:  # Lower threshold for PASSIF matching
                raw_idx = non_total_raw[best_idx][0]
                raw_desc = non_total_raw[best_idx][1]
                row_matches.append((structured_row_desc, raw_idx, score, raw_desc))
                print(f"Best match for '{structured_row_desc}' -> '{raw_desc}' (score: {score:.2f})")
            else:
                row_matches.append((structured_row_desc, -1, 0, ""))

    # Sort by score and assign without overlap
    row_matches.sort(key=lambda x: x[2], reverse=True)
    
    for structured_row_desc, raw_idx, score, raw_desc in row_matches:
        if raw_idx < 0 or score <= 0.4 or raw_idx in used_raw_rows:
            continue
        
        assigned_matches[structured_row_desc] = raw_idx
        used_raw_rows.add(raw_idx)
        print(f"Matching '{structured_row_desc}' -> Row {raw_idx}: '{raw_desc}' (score: {score:.2f})")

    # Step 4: Handle TOTAL PASSIF (I+II+III) - assign to last row if not already assigned
    total_passif_key = 'TOTAL PASSIF (I+II+III)'
    if total_passif_key not in assigned_matches:
        # Find the last row (highest index) that hasn't been used
        available_rows = [idx for idx, desc in raw_descriptions if idx not in used_raw_rows]
        if available_rows:
            last_row_idx = max(available_rows)
            last_row_desc = next(desc for idx, desc in raw_descriptions if idx == last_row_idx)
            assigned_matches[total_passif_key] = last_row_idx
            used_raw_rows.add(last_row_idx)
            print(f"Assigning last available row to '{total_passif_key}' -> Row {last_row_idx}: '{last_row_desc}'")

    print(f"\n----- Final Assignment Summary -----")
    print(f"Total assigned: {len(assigned_matches)} out of {len(all_structured_rows)} structured rows")
    for structured_row, raw_idx in assigned_matches.items():
        raw_desc = next(desc for idx, desc in raw_descriptions if idx == raw_idx)
        print(f"  '{structured_row}' -> Row {raw_idx}: '{raw_desc[:50]}...'")

    # Step 5: Create structured data
    structured_data = []
    for structured_row_desc in all_structured_rows:
        new_row = [structured_row_desc, None, None]
        
        if structured_row_desc in assigned_matches:
            raw_idx = assigned_matches[structured_row_desc]
            
            if raw_idx < len(raw_df):
                raw_row = raw_df.iloc[raw_idx].tolist()
                
                # Use enhanced extraction
                new_row[1] = extract_value_from_column_enhanced(raw_row, n_col)
                new_row[2] = extract_value_from_column_enhanced(raw_row, n1_col)
        
        structured_data.append(new_row)

    # Create DataFrame
    result_df = pd.DataFrame(structured_data, columns=structure["columns"])
    
    # Apply number cleaning
    print("\n----- Applying Number Cleaning for PASSIF -----")
    result_df = apply_number_cleaning_to_dataframe_passif(result_df)
    
    return result_df


def draw_bounding_box(image_path, bbox, output_path):
    """Draw a bounding box on the image and save it"""
    image = Image.open(image_path)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img_array, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    result_img = Image.fromarray(img_array)
    result_img.save(output_path)
    print(f"Saved annotated image to: {output_path}")

def process_image(imgpath, tab_ext, bbox_output=None, use_structure=True):
    """Process an image and extract PASSIF table data"""
    try:
        if imgpath.lower().endswith('.pdf'):
            pdf_processor = PDFProcessor(input_dir=None, output_dir=None)
            preprocessed_image = pdf_processor.process_pdf_in_memory(imgpath)
            
            if preprocessed_image is None:
                print(f"❌ No tables found in PDF: {imgpath}")
                return None, None, None
            
            if isinstance(preprocessed_image, np.ndarray):
                from PIL import Image
                preprocessed_image = Image.fromarray(preprocessed_image)
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                preprocessed_image.save(temp_path)
            
            try:
                (raw_df, cleaned_df), bbox = tab_ext.detect(temp_path)
            finally:
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            (raw_df, cleaned_df), bbox = tab_ext.detect(imgpath)
        
        print(f"✅ Successfully extracted PASSIF table from {imgpath}")
        
        if bbox_output and bbox and len(bbox) > 0 and not imgpath.lower().endswith('.pdf'):
            draw_bounding_box(imgpath, bbox[0], bbox_output)
        
        if use_structure and raw_df is not None and not raw_df.empty:
            structured_df = create_structured_passif_table(raw_df, BALANCE_SHEET_STRUCTURE_PASSIF)
            
            print("\n----- Raw PASSIF Table Data Columns -----")
            for col_idx in range(min(10, raw_df.shape[1])):
                sample_values = [str(raw_df.iloc[row_idx, col_idx])[:30] for row_idx in range(min(3, raw_df.shape[0]))]
                print(f"Column {col_idx}: {sample_values}")
            
            return raw_df, cleaned_df, structured_df
        
        return raw_df, cleaned_df, None
        
    except Exception as e:
        print(f"❌ Error processing PASSIF image: {imgpath}")
        print(traceback.format_exc())
        return None, None, None

def main():
    default_input = "./src/input/passif14.pdf"  
    default_output_dir = "./src/output/passif"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parser = argparse.ArgumentParser(description='PASSIF Table Extraction Tool - CLI Version')
    parser.add_argument('--input', type=str, default=default_input,
                       help=f'Path to the PASSIF file (PDF or image) (default: {default_input})')
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                       help=f'Directory to save output files (default: {default_output_dir})')
    parser.add_argument('--temp-dir', type=str, default='./temp_processed',
                       help='Temporary directory for PDF processing')
    parser.add_argument('--no-bbox', action='store_true',
                       help='Skip saving annotated image with bounding box')
    parser.add_argument('--no-html', action='store_true',
                       help='Skip saving HTML representation')
    parser.add_argument('--no-structure', action='store_true',
                       help='Skip applying PASSIF structure')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    
    _, ext = os.path.splitext(args.input)
    if ext.lower() not in ['.png', '.jpg', '.jpeg', '.pdf']:
        print(f"Error: Unsupported file format '{ext}'.")
        return 1
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    image_basename = os.path.basename(args.input).split('.')[0]
    raw_output = os.path.join(args.output_dir, f"{image_basename}_passif_raw_{timestamp}.csv")
    structured_output = os.path.join(args.output_dir, f"{image_basename}_passif_structured_{timestamp}.csv")
    bbox_output = None if args.no_bbox else os.path.join(args.output_dir, f"{image_basename}_passif_annotated_{timestamp}{ext}")
    html_output = None if args.no_html else os.path.join(args.output_dir, f"{image_basename}_passif_tables_{timestamp}.html")
    
    print(f"Loading table extraction models...")
    tab_ext = TableExtraction()
    print("Models loaded successfully.")
    
    print(f"Processing PASSIF image: {args.input}")
    raw_df, enhanced_df, structured_df = process_image(args.input, tab_ext, bbox_output, not args.no_structure)
    
    if raw_df is not None and not raw_df.empty:
        print("\n----- Raw PASSIF Table Data (Preview) -----")
        print(raw_df.head().to_string())
        
        if structured_df is not None:
            print("\n----- Structured PASSIF Table Data (Preview) -----")
            print(structured_df.head(15).to_string())  # Show more rows for PASSIF
        
        if html_output:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PASSIF Balance Sheet Extraction Results</title>
    <style>
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .section {{ font-weight: bold; background-color: #e6f3ff; }}
        .number {{ text-align: right; }}
    </style>
</head>
<body>
    <h1>PASSIF Balance Sheet Extraction Results</h1>
    <p>Processed on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Raw Table Data</h2>
    <table>
"""
            
            # Add raw table
            if not raw_df.empty:
                html_content += "<tr>"
                for col in raw_df.columns:
                    html_content += f"<th>{col}</th>"
                html_content += "</tr>"
                
                for _, row in raw_df.iterrows():
                    html_content += "<tr>"
                    for val in row:
                        html_content += f"<td>{val if val is not None else ''}</td>"
                    html_content += "</tr>"
            
            html_content += "</table>"
            
            # Add structured table
            if structured_df is not None and not structured_df.empty:
                html_content += "<h2>Structured PASSIF Data</h2><table>"
                html_content += "<tr>"
                for col in structured_df.columns:
                    html_content += f"<th>{col}</th>"
                html_content += "</tr>"
              
                current_section = ""
                for _, row in structured_df.iterrows():
                    desc = str(row['PASSIF'])
                    
                    # Check if we're entering a new section
                    section_found = False
                    for section_name in BALANCE_SHEET_STRUCTURE_PASSIF["sections"].keys():
                        if any(item.lower() in desc.lower() for item in BALANCE_SHEET_STRUCTURE_PASSIF["sections"][section_name]):
                            if current_section != section_name:
                                current_section = section_name
                                html_content += f'<tr class="section-header"><td colspan="3"><strong>{section_name}</strong></td></tr>'
                            break
                    
                    # Create the data row
                    html_content += "<tr>"
                    for val in row:
                        html_content += f"<td>{val if val is not None else ''}</td>"
                    html_content += "</tr>"
        

            else:
                html_content += "<p>No structured PASSIF data available</p>"
            
            html_content += """
    </table>
</body>
</html>
"""
            
            with open(html_output, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ PASSIF HTML output saved to: {html_output}")
        
        return 0
    else:
        print("❌ Failed to extract PASSIF table data from the image.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
