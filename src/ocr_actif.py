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
# Predefined structure for balance sheet tables (ACTIF section)
BALANCE_SHEET_STRUCTURE = {
    "columns": [
        "Description",
        "Montants Bruts",
        "Amortissements Provisions et pertes de valeurs",
        "Net N",
        "Net N-1"
    ],
    "sections": {
        "ACTIFS NON COURANTS": [
            "Ecart d'acquisition-goodwill positif ou négatif",
            "Immobilisations Incorporelles",
            "Immobilisations corporelles",
            "Terrains",
            "Bâtiments",
            "Autres immobilisations corporelles",
            "Immobilisations en concession",
            "Immobilisations encours",
            "Immobilisations financières",
            "Titres mis en équivalence",
            "Autres participations et créances rattachées",
            "Autres titres immobilisés",
            "Prêts et autres actifs financiers non courants",
            "Impôts différés actif",
            "TOTAL ACTIF NON COURANT"
        ],
        "ACTIF COURANT": [
            "Stocks et encours",
            "Créances et emplois assimilés",
            "Clients",
            "Autres débiteurs",
            "Impôts et assimilés",
            "Autres créances et emplois assimilés",
            "Disponibilités et assimilés",
            "Placements et autres actifs financiers courants",
            "Trésorerie",
            "TOTAL ACTIF COURANT",
            "TOTAL GENERAL ACTIF"
        ]
    }
}

def clean_number_value(value):
    """
    Clean numerical values according to specified rules:
    - Delete ], [, \, / and spaces
    - Delete strings within numbers
    - Delete . (don't consider it as comma)
    - Replace C, O with 0
    - Split numbers containing [ or ] and return list of parts
    - Delete cells that look like dates (24mai, 12decembre, etc.)
    """
    if value is None or pd.isna(value):
        return [None]
    
    # Convert to string
    value_str = str(value).strip()
    
    # If empty or 'nan', return None
    if not value_str or value_str.lower() == 'nan':
        return [None]
    
    # Check if it looks like a date (number followed by month name)
    date_pattern = r'\d+\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)'
    if re.search(date_pattern, value_str.lower()):
        return [None]
    
    # Replace C and O with 0
    value_str = value_str.replace('C', '0').replace('O', '0')
    
    # Split by [ and ] and newlines to handle multiple numbers in one cell
    # First, let's identify all splitting patterns
    split_patterns = ['[', ']', '\n', '\r']
    
    # Split the string by these patterns
    parts = [value_str]
    for pattern in split_patterns:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(pattern))
        parts = new_parts
    
    # Clean each part
    cleaned_parts = []
    for part in parts:
        if not part.strip():
            continue
            
        # Remove unwanted characters: ], [, \, /, spaces, dots
        cleaned_part = re.sub(r'[\]\[\\/\s\.]', '', part)
        
        # Remove non-numeric characters except for digits and basic separators
        # Keep only digits and common thousand separators
        cleaned_part = re.sub(r'[^0-9,\s]', '', cleaned_part)
        
        # Remove any remaining spaces
        cleaned_part = cleaned_part.replace(' ', '')
        
        # If the result contains only digits and commas, it's a valid number
        if cleaned_part and re.match(r'^[0-9,]+$', cleaned_part):
            # Remove leading/trailing commas
            cleaned_part = cleaned_part.strip(',')
            if cleaned_part:
                cleaned_parts.append(cleaned_part)
    
    # Return the cleaned parts, or [None] if no valid numbers found
    return cleaned_parts if cleaned_parts else [None]
def convert_to_numeric(value):
    """Convert cleaned string value to numeric, handling comma-separated thousands"""
    if value is None or value == '' or str(value).lower() == 'nan':
        return None
    
    try:
        # Remove commas (thousand separators) and convert to float
        numeric_str = str(value).replace(',', '')
        return float(numeric_str)
    except (ValueError, TypeError):
        return None
def calculate_missing_values(df):
    """
    Calculate missing values based on the formula:
    - If Net N is empty and both Montant Brut and Amortissement are not empty: Net N = Montant Brut - Amortissement
    - If Amortissement is empty and both Montant Brut and Net N are not empty: Amortissement = Montant Brut - Net N
    - If Montant Brut is empty and both Net N and Amortissement are not empty: Montant Brut = Amortissement + Net N
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    calculated_df = df.copy()
    
    # Define column indices based on the structure
    montant_brut_col = 1  # "Montants Bruts"
    amortissement_col = 2  # "Amortissements Provisions et pertes de valeurs"
    net_n_col = 3  # "Net N"
    
    for idx in range(len(calculated_df)):
        # Get current values and convert to numeric
        montant_brut = convert_to_numeric(calculated_df.iloc[idx, montant_brut_col])
        amortissement = convert_to_numeric(calculated_df.iloc[idx, amortissement_col])
        net_n = convert_to_numeric(calculated_df.iloc[idx, net_n_col])
        
        # Calculate Net N if missing
        if net_n is None and montant_brut is not None and amortissement is not None:
            calculated_value = montant_brut - amortissement
            if calculated_value > 0:
                calculated_df.iloc[idx, net_n_col] = str(int(calculated_value)) if calculated_value == int(calculated_value) else str(calculated_value)
                print(f"Calculated Net N for row {idx}: {montant_brut} - {amortissement} = {calculated_value}")
            
        
    
    return calculated_df

def shift_data_right(df, start_row, start_col, positions_to_shift):
    """
    Shift data to the right by the specified number of positions
    """
    # Get the maximum column index we need to consider
    max_col = len(df.columns) - 1
    
    # Work backwards from the rightmost columns to avoid overwriting
    for col_idx in range(max_col, start_col - 1, -1):
        target_col = col_idx + positions_to_shift
        
        # Only shift if target column exists and source has data
        if target_col < len(df.columns) and not pd.isna(df.iloc[start_row, col_idx]) and df.iloc[start_row, col_idx] is not None:
            # If target cell is empty, move the data
            if pd.isna(df.iloc[start_row, target_col]) or df.iloc[start_row, target_col] is None:
                df.iloc[start_row, target_col] = df.iloc[start_row, col_idx]
                df.iloc[start_row, col_idx] = None

def apply_number_cleaning_to_dataframe(df):
    """
    Apply number cleaning to all numerical columns in the DataFrame
    Handle cases where one cell contains multiple numbers by expanding columns
    and shifting existing data to make room for split numbers
    Then calculate missing values based on financial formulas
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # First pass: determine how many additional columns we need
    max_additional_cols = 0
    split_info = []  # Store information about splits: (row, col, split_count)
    
    for col_idx in range(1, len(cleaned_df.columns)):  # Skip Description column
        for row_idx in range(len(cleaned_df)):
            value = cleaned_df.iloc[row_idx, col_idx]
            cleaned_values = clean_number_value(value)
            if len(cleaned_values) > 1:
                split_count = len(cleaned_values) - 1
                split_info.append((row_idx, col_idx, split_count))
                max_additional_cols = max(max_additional_cols, split_count)
    
    # Process splits in reverse order (rightmost columns first) to avoid conflicts
    split_info.sort(key=lambda x: (x[1], x[0]), reverse=True)
    
    for row_idx, col_idx, split_count in split_info:
        value = cleaned_df.iloc[row_idx, col_idx]
        cleaned_values = clean_number_value(value)
        
        if len(cleaned_values) > 1:
            # Check if we need to shift existing data to the right
            data_to_shift = []
            for check_col in range(col_idx + 1, min(col_idx + len(cleaned_values), len(cleaned_df.columns))):
                current_value = cleaned_df.iloc[row_idx, check_col]
                if not pd.isna(current_value) and current_value is not None:
                    data_to_shift.append((check_col, current_value))
            
            # Shift existing data to the right
            if data_to_shift:
                # Calculate how far we need to shift
                shift_distance = len(cleaned_values) - 1
                
                # Shift data starting from the rightmost affected column
                for shift_col, shift_value in reversed(data_to_shift):
                    target_col = shift_col + shift_distance
                    if target_col < len(cleaned_df.columns):
                        # Find the next available column
                        while target_col < len(cleaned_df.columns) and not pd.isna(cleaned_df.iloc[row_idx, target_col]) and cleaned_df.iloc[row_idx, target_col] is not None:
                            target_col += 1
                        
                        if target_col < len(cleaned_df.columns):
                            cleaned_df.iloc[row_idx, target_col] = shift_value
                            cleaned_df.iloc[row_idx, shift_col] = None
            
            # Place the split numbers in consecutive columns
            for i, cleaned_value in enumerate(cleaned_values):
                target_col = col_idx + i
                if target_col < len(cleaned_df.columns):
                    cleaned_df.iloc[row_idx, target_col] = cleaned_value if cleaned_value is not None else None
    
    # Clean up any remaining single-cell numbers that weren't part of splits
    original_col_count = len(cleaned_df.columns)
    for col_idx in range(1, original_col_count):  # Skip Description column
        for row_idx in range(len(cleaned_df)):
            value = cleaned_df.iloc[row_idx, col_idx]
            # Only process if this cell wasn't already handled by the split logic
            if value is not None and not any(row_idx == r and col_idx == c for r, c, _ in split_info):
                cleaned_values = clean_number_value(value)
                cleaned_df.iloc[row_idx, col_idx] = cleaned_values[0] if cleaned_values[0] is not None else None
    
    # Apply financial calculations to fill missing values
    cleaned_df = calculate_missing_values(cleaned_df)
    
    return cleaned_df

def similarity_score(a, b):
    """
    Calculate similarity between two strings after removing all spaces
    """
    # Ensure we're working with strings and remove all spaces
    a = re.sub(r'\s+', '', str(a).lower().strip())
    b = re.sub(r'\s+', '', str(b).lower().strip())
    
    # Handle None values or empty strings
    if not a or not b:
        return 0
        
    return SequenceMatcher(None, a, b).ratio()

def clean_text(text):
    """Clean text for better matching"""
    if not isinstance(text, str):
        return ""
    # Remove extra spaces, normalize case
    return re.sub(r'\s+', ' ', text.lower().strip())
   

def find_best_match(target_text, candidate_texts):
    """
    Find the best matching text from a list of candidates
    Returns the index of the best match and the similarity score
    """
    if not target_text or not candidate_texts:
        return -1, 0
        
    target_text = clean_text(target_text)
    
    best_score = 0
    best_idx = -1
    
    for idx, candidate in enumerate(candidate_texts):
        candidate = clean_text(candidate)
        score = similarity_score(target_text, candidate)
       
        if score > best_score and score>0.3:
            best_score = score
            best_idx = idx
            
    return best_idx, best_score

def find_column_by_header(raw_df, header_terms, def_index, existing_columns=None ):
    """
    Find column index that best matches any of the header terms
    Returns the best matching column index and next column
    """
    if existing_columns is None:
        existing_columns = set()
    
    best_col_idx = -1
    best_score = 0
    
    # First two rows are likely headers
    for row_idx in range(min(5, len(raw_df))):
        for col_idx in range(len(raw_df.columns)):
            # Skip columns already used
            if col_idx in existing_columns:
                continue
           
            cell_val = str(raw_df.iloc[row_idx, col_idx]).lower().strip()
           
            # Check against each term
            for term in header_terms:
                if cell_val is not None and str(cell_val).strip() and str(cell_val).strip().lower() != 'nan' :
                    score = similarity_score(term, cell_val)
                    if score > best_score and score > 0.7 and col_idx >def_index:  # Minimum threshold
                        best_score = score
                        best_col_idx = col_idx
    
    # If found, consider the next column too (as specified in rules)
    next_col_idx = best_col_idx + 1 if best_col_idx >= 0 and best_col_idx + 1 < len(raw_df.columns) else -1
    
    # Return both the best column and its adjacent column
    return best_col_idx, next_col_idx, best_score

def extract_value_from_column(row, col_idx):
    """Extract a value from a specific column in a row"""
    if col_idx < 0 or col_idx >= len(row):
        return None
        
    val = row[col_idx]
    if val is not None and str(val).strip() and str(val).strip().lower() != 'nan' :
        # Check if it's a number or looks like a number with spaces
        if isinstance(val, (int, float)) or (isinstance(val, str) and any(c.isdigit() for c in val)):
            # Skip values that look like years (e.g., 2021, 2020)
            #if str(val) not in ["2021", "2020", "2019"]:
                return val
    return None

def create_structured_table_data(raw_df, structure=None):
    """
    Create structured table data strictly following the specified rules:
    1. Include all rows from BALANCE_SHEET_STRUCTURE
    2. For each row, find the best match in raw data using text similarity
    3. Extract values using specific column identification logic:
       - Montants Bruts: column where "Montants" or "Bruts" is found
       - Amortissements et Provisions: column where part of this text is found
       - Net 2021: the first column where "Net" is found
       - Net 2020: the last or before last column
    """
    if raw_df is None or raw_df.empty or not structure:
        # Return empty DataFrame with structure columns
        return pd.DataFrame(columns=structure["columns"])
    
    # Get all structured rows from the balance sheet structure
    all_structured_rows = []
    for section, rows in structure["sections"].items():
        all_structured_rows.extend(rows)
    
    # First, let's identify the columns in raw_df:
    print("\n----- Identifying Raw Data Columns -----")
    
    # Define search terms for each column
    montants_terms = ["montants", "bruts", "montants bruts"]
    amortissements_terms = ["amortissements", "amort", "prov", "provisions", "pertes", "valeurs", "et"]
    net2021_terms = ["net", "net 20", "N" , "2020"]
    net2020_terms = ["net", "net 20", "N-1"]
    
    # Column tracking to avoid overlaps
    used_columns = set()
    
    # Find Montants Bruts column
    montants_col, montants_next_col, montants_score = find_column_by_header(raw_df, montants_terms, 0, used_columns)
    if montants_col >= 0:
        used_columns.add(montants_col)
        print(f"Found 'Montants Bruts' column at index {montants_col} (score: {montants_score:.2f})")
    
    # Find Amortissements et Provisions column
    amort_col, amort_next_col, amort_score = find_column_by_header(raw_df, amortissements_terms, montants_col, used_columns)
    if amort_col >= 0 and amort_col not in used_columns:
        used_columns.add(amort_col)
        print(f"Found 'Amortissements et Provisions' column at index {amort_col} (score: {amort_score:.2f})")
    
    last_col = len(raw_df.columns) - 1  
    # Find Net 2021 column 
    net2021_col, net2021_next_col, net2021_score = find_column_by_header(raw_df, net2021_terms, amort_col, used_columns)
   
    if net2021_col >= 0 and net2021_col != last_col and net2021_col not in used_columns:
        used_columns.add(net2021_col)
        print(f"Found 'Net 2021' column at index {net2021_col} (score: {net2021_score:.2f})")
    
    
    # For Net 2020, use the last or before last column that's not already used
    net2020_col, net2020_next_col, net2020_score = find_column_by_header(raw_df, net2020_terms, net2021_col, used_columns)
    
    if net2020_col > net2021_col and net2020_col > 0 and net2020_col not in used_columns:
        used_columns.add(net2020_col)
        print(f"Found 'Net 2020' column at index {net2020_col} (score: {net2020_score:.2f})")
    else:
        last_col = len(raw_df.columns) - 1    
        net2020_col = last_col 
        if net2020_col not in used_columns:
            used_columns.add(net2020_col)
            print(f"Using column at index {net2020_col} for 'Net N-1'")
            if (net2020_col - 1) not in used_columns:
                net2020_next_col = net2020_col-1
            else: 
                net2020_next_col = 0
        else:
            # Find any available column near the end
            for i in range(last_col, max(0, last_col - 1), -1):
                if i not in used_columns:
                    net2020_col = i
                    used_columns.add(i)
                    print(f"Using alternate column at index {net2020_col} for 'Net N-1'")
                    break
            else:
                net2020_col = -1
                print("Could not find a suitable column for 'Net N-1'")

    if net2020_col == net2021_col and net2020_col-1 not in used_columns : 
        net2021_col=net2020_col-1
        used_columns.add(net2021_col)
        print(f"Found 'Net N' column at index {net2021_col} (score: {net2021_score:.2f})")
    
    # Now identify rows by matching descriptions
    print("\n----- Matching Row Descriptions -----")
    
    # Extract all potential description rows from raw data
    # Focus on first two columns for descriptions (based on rule 1)
    raw_descriptions = []
   
    for idx, row in raw_df.iterrows():
        # Look specifically in columns 1-2 for descriptions as specified     
       
        for col_idx in range(min(used_columns)):  # Columns 1-2 in 0-based indexing
          
            if col_idx < len(row):
                val = row[col_idx]
               
                if val is not None and str(val).strip() and str(val).strip().lower() != 'nan':
                    # If it's not a pure number, it might be a description
                    if not str(val).replace(' ', '').isdigit():
                        raw_descriptions.append((idx, val))
                        break
 
    # Initialize structured DataFrame with all rows from the structure
    structured_data = []
    
    # Keep track of which raw rows have been used to prevent overlap
    used_raw_rows = set()
    
    # First pass: Match each structured row with the best raw row and track scores
    row_matches = []
    for structured_row_desc in all_structured_rows:
        # Compare structured description with all raw descriptions to find the best match
        candidate_texts = [desc for _, desc in raw_descriptions]
        best_idx, score = find_best_match(structured_row_desc, candidate_texts)
        
        if best_idx >= 0 and score > 0.5:
            raw_idx = raw_descriptions[best_idx][0]  # Get the actual row index in raw_df
            row_matches.append((structured_row_desc, raw_idx, score))
        else:
            # No good match found, add with a placeholder index and low score
            row_matches.append((structured_row_desc, -1, 0))
    
    # Sort matches by score in descending order to prioritize higher quality matches
    row_matches.sort(key=lambda x: x[2], reverse=True)
    
    # Second pass: Assign raw rows to structured rows without overlap
    assigned_matches = {}
    for structured_row_desc, raw_idx, score in row_matches:
        # Skip if no match was found
        if raw_idx < 0 or score <= 0.5:
            continue
            
        # Skip if this raw row has already been used
        if raw_idx in used_raw_rows:
            continue
            
        # Assign this raw row to the structured description
        assigned_matches[structured_row_desc] = raw_idx
        used_raw_rows.add(raw_idx)
        
        # Debug information
        print(f"Matching '{structured_row_desc}' -> '{raw_descriptions[[i for i, (idx, _) in enumerate(raw_descriptions) if idx == raw_idx][0]][1]}' (score: {score:.2f})")
    
    # Now create the final structured data
    for structured_row_desc in all_structured_rows:
        # Initialize a row with the structure description and None values
        new_row = [structured_row_desc, None, None, None, None]
        
        # If we have a match for this structured row
        if structured_row_desc in assigned_matches:
            raw_idx = assigned_matches[structured_row_desc]
            raw_row = raw_df.iloc[raw_idx].tolist()
            used_columns2 = used_columns.copy()
            
            # Extract financial values based on identified columns
            # 1. Montants Bruts
            if montants_col >= 0:
                new_row[1] = extract_value_from_column(raw_row, montants_col) 
                  
                if new_row[1] is None and  montants_col+1 not in used_columns2:
                    new_row[1] = extract_value_from_column(raw_row, montants_col+1)
                    
                    if new_row[1] is not None: 
                        used_columns2.add(montants_col+1)
                else: 
                        if new_row[1] is None and montants_col-1 > 0 and montants_col-1 not in used_columns2:
                            new_row[1] = extract_value_from_column(raw_row, montants_col-1)
                            
                            if new_row[1] is not None: 
                                used_columns2.add(montants_col-1)
            
            # 2. Amortissements et Provisions
            if amort_col >= 0:
                new_row[2] = extract_value_from_column(raw_row, amort_col)
                
                if new_row[2] is None and amort_col+1 not in used_columns2:
                    new_row[2] = extract_value_from_column(raw_row, amort_col+1)
                    
                    if new_row[2] is not None: 
                        used_columns2.add(amort_col+1)
                else :
                    if new_row[2] is None and amort_col-1 > 0 and amort_col-1 not in used_columns2:
                            new_row[2] = extract_value_from_column(raw_row, amort_col-1)
                            
                            if new_row[2] is not None: 
                                used_columns2.add(amort_col-1)
            
            # 3. Net 2021
            if net2021_col >= 0:
                new_row[3] = extract_value_from_column(raw_row, net2021_col)
                if new_row[3] is None and (net2021_col+1 > 0 and net2021_col+1 not in used_columns2):
                    new_row[3] = extract_value_from_column(raw_row, net2021_col+1)
                    if new_row[3] is not None: 
                        used_columns2.add(net2021_col+1)
                else: 
                        if new_row[3] is None and net2021_col-1 > 0 and net2021_col-1 not in used_columns2:
                            new_row[3] = extract_value_from_column(raw_row, net2021_col-1)
                            if new_row[3] is not None: 
                                used_columns2.add(net2021_col-1)
            
            # 4. Net 2020
            if net2020_col >= 0:
                new_row[4] = extract_value_from_column(raw_row, net2020_col)
                if new_row[4] is None and (net2020_next_col > 0 and net2020_next_col not in used_columns2):
                    new_row[4] = extract_value_from_column(raw_row, net2020_next_col)
                    if new_row[4] is None: 
                        if net2020_col-1 > 0 and net2020_col-1 not in used_columns2:
                            new_row[4] = extract_value_from_column(raw_row, net2020_col-1)
        else:
            # No match found for this structured row
            print(f"No good match found for '{structured_row_desc}'")
        
        # Add the row to our structured data
        structured_data.append(new_row)
    
    # Create DataFrame with all structured rows
    result_df = pd.DataFrame(structured_data, columns=structure["columns"])
    
    # Apply number cleaning to the structured DataFrame
    print("\n----- Applying Number Cleaning -----")
    result_df = apply_number_cleaning_to_dataframe(result_df)
    
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
    """Process an image and extract table data"""
    try:
        # Check if input is a PDF file
        if imgpath.lower().endswith('.pdf'):
            # Use in-memory PDF processing
            pdf_processor = PDFProcessor(input_dir=None, output_dir=None)  # No output directory
            preprocessed_image = pdf_processor.process_pdf_in_memory(imgpath)
            
            if preprocessed_image is None:
                print(f"❌ No tables found in PDF: {imgpath}")
                return None, None, None
            
            # Convert numpy array to PIL Image and save temporarily
            if isinstance(preprocessed_image, np.ndarray):
                from PIL import Image
                preprocessed_image = Image.fromarray(preprocessed_image)
            
            # Save temporarily for table extraction (since tab_ext.detect expects a file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                preprocessed_image.save(temp_path)
            
            try:
                # Extract table data from the temporary file
                (raw_df, cleaned_df), bbox = tab_ext.detect(temp_path)
            finally:
                # Clean up the temporary file
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            # Original image processing for non-PDF files
            (raw_df, cleaned_df), bbox = tab_ext.detect(imgpath)
        
        print(f"✅ Successfully extracted table from {imgpath}")
        
        # Save image with bounding box if requested (only for non-PDF or if you have the original image)
        if bbox_output and bbox and len(bbox) > 0 and not imgpath.lower().endswith('.pdf'):
            draw_bounding_box(imgpath, bbox[0], bbox_output)
        
        # Apply structure normalization if enabled
        if use_structure and raw_df is not None and not raw_df.empty:
            # Use the improved structured table data creation function
            structured_df = create_structured_table_data(raw_df, BALANCE_SHEET_STRUCTURE)
            
            # Debug: Print raw data column indices to help with debugging
            print("\n----- Raw Table Data Columns -----")
            for col_idx in range(min(15, raw_df.shape[1])):  # Show first 15 columns
                sample_values = [str(raw_df.iloc[row_idx, col_idx])[:30] for row_idx in range(min(3, raw_df.shape[0]))]
                print(f"Column {col_idx}: {sample_values}")
            
            return raw_df, cleaned_df, structured_df
        
        return raw_df, cleaned_df, None
        
    except Exception as e:
        print(f"❌ Error processing image: {imgpath}")
        print(traceback.format_exc())
        return None, None, None
def main():
    # Default input and output paths
    default_input = "./src/input/actif11.pdf"
    default_output_dir = "./src/output/actif"
    
       
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parser = argparse.ArgumentParser(description='Table Extraction Tool - CLI Version (Supports PDF and Images)')
    
    # Arguments with default values
    parser.add_argument('--input', type=str, default=default_input,
                        help=f'Path to the input file (PDF or image) containing a table (default: {default_input})')
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                        help=f'Directory to save all output files (default: {default_output_dir})')
    parser.add_argument('--temp-dir', type=str, default='./temp_processed',
                        help='Temporary directory for PDF processing (default: ./temp_processed)')
    parser.add_argument('--no-bbox', action='store_true',
                        help='Skip saving the annotated image with bounding box')
    parser.add_argument('--no-html', action='store_true',
                        help='Skip saving the HTML representation')
    parser.add_argument('--no-structure', action='store_true',
                        help='Skip applying predefined table structure')
    
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    
    # Check if file is of supported type
    _, ext = os.path.splitext(args.input)
    if ext.lower() not in ['.png', '.jpg', '.jpeg', '.pdf']:  # Added PDF support
        print(f"Error: Unsupported file format '{ext}'. Only PNG, JPG, JPEG, and PDF formats are supported.")
        return 1
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    
    # Define output file paths
    image_basename = os.path.basename(args.input).split('.')[0]
    raw_output = os.path.join(args.output_dir, f"{image_basename}_raw_{timestamp}.csv")
    enhanced_output = os.path.join(args.output_dir, f"{image_basename}_enhanced_{timestamp}.csv")
    structured_output = os.path.join(args.output_dir, f"{image_basename}_structured_{timestamp}.csv")
    bbox_output = None if args.no_bbox else os.path.join(args.output_dir, f"{image_basename}_annotated_{timestamp}{ext}")
    html_output = None if args.no_html else os.path.join(args.output_dir, f"{image_basename}_tables_{timestamp}.html")
    
    print(f"Loading table extraction models...")
    tab_ext = TableExtraction()
    print("Models loaded successfully.")
    
    print(f"Processing image: {args.input}")
    raw_df, enhanced_df, structured_df = process_image(args.input, tab_ext, bbox_output, not args.no_structure)
    
    if raw_df is not None and not raw_df.empty:
        # Save the raw data to CSV
        #raw_df.to_csv(raw_output, index=False)
        #print(f"✅ Raw data saved to: {raw_output}")
        
        
        
        # Save structured data if available
        #if structured_df is not None:
            #structured_df.to_csv(structured_output, index=False)
            #print(f"✅ Structured data saved to: {structured_output}")
        
        # Display sample of the data
        print("\n----- Raw Table Data (Preview) -----")
        print(raw_df.head().to_string())
        
      
        
        if structured_df is not None:
            print("\n----- Structured Table Data (Preview) -----")
            print(structured_df.head().to_string())
        
        # Save HTML representation if requested
        if html_output:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Extracted Table: {image_basename}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h2 {{ color: #2563eb; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .section-header {{ background-color: #e9e9e9; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Table Extraction Results: {image_basename}</h1>
                <p>Processed on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Raw Table Data</h2>
                {raw_df.to_html(index=False)}
                
              
                
                <h2>Structured Table Data (Complete)</h2>
                """
            
            # Add formatted structured data with section headers
            if structured_df is not None:
                # Convert DataFrame to HTML with custom formatting for sections
                html_rows = []
                html_rows.append("<table border='1' class='dataframe'>")
                
                # Add header row
                html_rows.append("<thead>")
                html_rows.append("<tr style='text-align: right;'>")
                for col in structured_df.columns:
                    html_rows.append(f"<th>{col}</th>")
                html_rows.append("</tr>")
                html_rows.append("</thead>")
                
                # Add data rows with section headers
                html_rows.append("<tbody>")
                
                # Add section headers and data rows
                section_added = set()
                current_section = ""
                
                for desc in structured_df['Description']:
                    # Determine which section this row belongs to
                    section_found = None
                    for section, rows in BALANCE_SHEET_STRUCTURE["sections"].items():
                        if desc in rows:
                            section_found = section
                            break
                    
                    # Add section header if we're entering a new section
                    if section_found and section_found != current_section and section_found not in section_added:
                        html_rows.append(f"<tr class='section-header'><td colspan='{len(structured_df.columns)}'>{section_found}</td></tr>")
                        section_added.add(section_found)
                        current_section = section_found
                    
                    # Add data row
                    row_data = structured_df[structured_df['Description'] == desc].iloc[0]
                    html_rows.append("<tr>")
                    for val in row_data:
                        html_rows.append(f"<td>{val if val is not None else ''}</td>")
                    html_rows.append("</tr>")
                
                html_rows.append("</tbody>")
                html_rows.append("</table>")
                
                html_content += "\n".join(html_rows)
            else:
                html_content += "<p>No structured data available</p>"
            
            html_content += """
            </body>
            </html>
            """
            
            with open(html_output, 'w') as f:
                f.write(html_content)
            print(f"✅ HTML output saved to: {html_output}")
            
        return 0
    else:
        print("❌ Failed to extract table data from the image.")
        return 1

if __name__ == "__main__":
    sys.exit(main())