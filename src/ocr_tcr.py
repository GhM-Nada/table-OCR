import os
os.environ['CCACHE_DISABLE'] = '1'
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import tempfile
import traceback
from datetime import datetime
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings

# Suppress PaddleOCR warnings
warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", message="No ccache found")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your custom modules AFTER setting environment variables
try:
    from table_creator.table_extractor import TableExtraction
    from preprocessing import PDFProcessor
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure table_creator and preprocessing modules are available")
    exit(1)

TCR_FINANCIAL_STRUCTURE = {
    "columns": [
        "Rubriques",
        "N_DEBIT", 
        "N_CREDIT",
        "N-1_DEBIT",
        "N-1_CREDIT"
    ],
    "sections": {
        "CHIFFRE_AFFAIRES": [
            "Ventes de marchandises",
            "Produits fabriqués",
            "Production vendue Prestations de services",
            "Vente de travaux",
            "Produits annexes",
            "Rabais, remises, ristournes accordés",
            "Chiffre d'affaires net des Rabais, remises, ristournes"
        ],
        "PRODUCTION": [
            "Production stockée ou déstockée",
            "Production immobilisée",
            "Subventions d'exploitation",
            "I-Production de l'exercice"
        ],
        "CONSOMMATIONS": [
            "Achats de marchandises vendues",
            "Matières premières",
            "Autres approvisionnements",
            "Variations des stocks",
            "Achats d'études et de prestations de services",
            "Autres consommations",
            "Rabais, remises, ristournes obtenus sur achats",
            "Sous-traitance générale",
            "Locations",
            "Entretien, réparations et maintenance",
            "Primes d'assurances",
            "Personnel extérieur à l'entreprise",
            "Rémunération d'intermédiaires et honoraires",
            "Publicité",
            "Déplacements, missions et réceptions",
            "Autres services",
            "Rabais, remises, ristournes obtenus sur services extérieurs",
            "II-Consommations de l'exercice"
        ],
        "VALEUR_AJOUTEE": [
            "III-Valeur ajoutée d'exploitation (I-II)"
        ],
        "CHARGES_PERSONNEL": [
            "Charges de personnel",
            "Impôts et taxes et versements assimilés"
        ],
        "RESULTAT_EXPLOITATION": [
            "IV-Excédent brut d'exploitation"
        ],
        "AUTRES_PRODUITS_CHARGES": [
            "Autres produits opérationnels",
            "Autres charges opérationnelles",
            "Dotations aux amortissements",
            "Provision",
            "Pertes de valeur",
            "Reprises sur pertes de valeur et provisions"
        ],
        "RESULTAT_OPERATIONNEL": [
            "V-Résultat opérationnel"
        ],
        "RESULTAT_FINANCIER": [
            "Produits financiers",
            "Charges financières",
            "VI-Résultat financier"
        ],
        "RESULTAT_ORDINAIRE": [
            "VII-Résultat ordinaire (V+VI)"
        ],
        "ELEMENTS_EXTRAORDINAIRES": [
            "Eléments extraordinaires (produits)",
            "Eléments extraordinaires (Charges)"
        ],
        "RESULTAT_EXTRAORDINAIRE": [
            "VIII-Résultat extraordinaire"
        ],
        "IMPOTS": [
            "Impôts exigibles sur résultats",
            "Impôts différés (variations) sur résultats"
        ],
        "RESULTAT_NET": [
            "IX - RESULTAT NET DE L'EXERCICE"
        ]
    }
}

def clean_number_value(value):
    """Enhanced numerical value cleaning with specific rules"""
    if value is None or pd.isna(value):
        return [None]
    
    # Convert to string
    value_str = str(value).strip()
    
    # If empty or 'nan', return None
    if not value_str or value_str.lower() == 'nan':
        return [None]
    
    # Check if it looks like a date
    date_pattern = r'\d+\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)'
    if re.search(date_pattern, value_str.lower()):
        return [None]
    
    # Split by various patterns
    split_patterns = ['[', ']', '\n', '\r', '|']
    parts = [value_str]
    for pattern in split_patterns:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(pattern))
        parts = new_parts
    
    # Clean each part with enhanced rules
    cleaned_parts = []
    for part in parts:
        if not part.strip():
            continue
        
        cleaned_part = clean_single_number(part)
        if cleaned_part is not None:
            cleaned_parts.append(cleaned_part)
    
    return cleaned_parts if cleaned_parts else [None]

def clean_single_number(value_str):
    """Clean a single number string according to specific rules"""
    if not value_str or not value_str.strip():
        return None
    
    original = value_str.strip()
    
    # Handle specific patterns based on your examples:
    
    # Pattern 1: Numbers with spaces (1 338.911 861 -> 1338911861)
    # Remove all spaces first
    cleaned = re.sub(r'\s+', '', original)
    
    # Pattern 2: Remove ALL dots since decimal separator is comma, not dot
    # (.27.091 831 -> 27091831, 1.338.911 -> 1338911)
    cleaned = cleaned.replace('.', '')
    
    # Pattern 3: Handle negative numbers
    # Single minus: - 280 118 -> -280118
    if cleaned.startswith('-') and not cleaned.startswith('---'):
        # Remove the minus, clean the number, then add minus back
        number_part = cleaned[1:].strip()
        number_part = re.sub(r'[^0-9,]', '', number_part)
        if number_part:
            cleaned = '-' + number_part
    
    # Pattern 4: Triple minus becomes positive (--- 280 118 -> 280118)
    elif cleaned.startswith('---'):
        # Remove all minuses and clean
        cleaned = re.sub(r'^-+', '', cleaned)
        cleaned = re.sub(r'[^0-9,]', '', cleaned)
    
    # Pattern 5: Keep comma-separated numbers as is (298,767 -> 298,767)
    # This is already handled by keeping commas
    
    # Replace common OCR errors
    cleaned = cleaned.replace('C', '0').replace('O', '0').replace('l', '1')
    
    # Final cleaning: remove unwanted characters but keep numbers, commas, and minus
    # Note: dots are excluded since decimal separator is comma
    cleaned = re.sub(r'[^0-9,+-]', '', cleaned)
    
    # Remove multiple consecutive commas
    cleaned = re.sub(r',{2,}', ',', cleaned)
    
    # Remove leading/trailing commas
    cleaned = cleaned.strip(',')
    
    # Validate the result
    if cleaned and (re.match(r'^-?[\d,]+$', cleaned) or re.match(r'^-?\d+$', cleaned)):
        # Additional validation: should contain at least one digit
        if any(c.isdigit() for c in cleaned):
            return cleaned
    
    return None


def apply_number_cleaning_to_dataframe(df):
    """Apply enhanced number cleaning to all numeric columns in the dataframe"""
    if df is None or df.empty:
        return df
    
    logger.info("Applying enhanced number cleaning to dataframe...")
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Identify numeric columns (skip the first column which is descriptions)
    numeric_columns = []
    for col_idx in range(1, len(cleaned_df.columns)):  # Skip first column
        col_name = cleaned_df.columns[col_idx]
        if any(keyword in str(col_name).lower() for keyword in ['debit', 'credit', 'n', 'n-1']) or col_idx > 0:
            numeric_columns.append(col_idx)
    
    # Apply cleaning to identified numeric columns
    for col_idx in numeric_columns:
        col_name = cleaned_df.columns[col_idx]
        logger.info(f"Cleaning column {col_idx}: {col_name}")
        
        for row_idx in range(len(cleaned_df)):
            value = cleaned_df.iloc[row_idx, col_idx]
            if pd.notna(value) and str(value).strip():
                cleaned_values = clean_number_value(value)
                if cleaned_values and cleaned_values[0] is not None:
                    cleaned_df.iloc[row_idx, col_idx] = cleaned_values[0]
                else:
                    cleaned_df.iloc[row_idx, col_idx] = None
    
    logger.info("Enhanced number cleaning completed")
    return cleaned_df

def similarity_score(a, b):
    """Calculate similarity between two strings after removing all spaces"""
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

class DynamicTCRColumnDetector:
    """Dynamic column detector for TCR tables with enhanced header detection"""
    
    def __init__(self):
        self.year_patterns = [
            r'20\d{2}',  # 2020, 2021, etc.
            r'19\d{2}',  # 1999, etc.
        ]
        self.debit_patterns = [
            'debit', 'débit', 'doit'
        ]
        self.credit_patterns = [
            'credit', 'crédit', 'avoir'
        ]
        self.n_patterns = [
            'n', 'exercice', 'courant'
        ]
        self.n1_patterns = [
            'n-1', 'précédent', 'precedent', 'antérieur'
        ]
    
    def is_numeric_column(self, df: pd.DataFrame, col_idx: int, min_numeric_ratio: float = 0.6) -> bool:
        """Check if a column contains primarily numeric data"""
        if col_idx >= len(df.columns):
            return False
        
        column_data = df.iloc[:, col_idx].dropna()
        if len(column_data) == 0:
            return False
        
        numeric_count = 0
        for cell in column_data:
            cell_str = str(cell).replace(',', '').replace(' ', '').strip()
            # Check for numeric patterns
            if re.search(r'\d+', cell_str):
                numeric_count += 1
        
        return numeric_count / len(column_data) >= min_numeric_ratio
    
    def is_description_column(self, df: pd.DataFrame, col_idx: int) -> bool:
        """Check if a column contains description text"""
        if col_idx >= len(df.columns):
            return False
        
        column_data = df.iloc[:, col_idx].dropna()
        if len(column_data) == 0:
            return False
        
        text_count = 0
        for cell in column_data:
            cell_str = str(cell).strip()
            # Check if it contains meaningful text (not just numbers)
            if (len(cell_str) > 3 and 
                re.search(r'[a-zA-Zàâäéèêëïîôöùûüÿç]', cell_str) and
                not cell_str.replace(' ', '').replace(',', '').isdigit()):
                text_count += 1
        
        return text_count / len(column_data) >= 0.5
    
    def find_header_zones(self, df: pd.DataFrame, max_header_rows: int = 5) -> Dict[str, List[Tuple[int, int]]]:
        """Find zones where specific headers appear"""
        zones = {
            'debit': [],
            'credit': [],
            'n': [],
            'n-1': [],
            'years': []
        }
        
        # Search in first few rows for headers
        for row_idx in range(min(max_header_rows, len(df))):
            for col_idx in range(len(df.columns)):
                cell_value = str(df.iloc[row_idx, col_idx]).lower().strip()
                
                # Check for DEBIT
                if any(pattern in cell_value for pattern in self.debit_patterns):
                    zones['debit'].append((row_idx, col_idx))
                
                # Check for CREDIT
                if any(pattern in cell_value for pattern in self.credit_patterns):
                    zones['credit'].append((row_idx, col_idx))
                
                # Check for N (current year indicators)
                if any(pattern in cell_value for pattern in self.n_patterns):
                    zones['n'].append((row_idx, col_idx))
                
                # Check for N-1 (previous year indicators)
                if any(pattern in cell_value for pattern in self.n1_patterns):
                    zones['n-1'].append((row_idx, col_idx))
                
                # Check for years
                if any(re.search(pattern, cell_value) for pattern in self.year_patterns):
                    zones['years'].append((row_idx, col_idx))
        
        return zones
    
    def detect_column_structure(self, df: pd.DataFrame) -> Dict[str, int]:
        """Dynamically detect column structure"""
        logger.info("=== Starting Dynamic Column Detection ===")
        
        # Step 1: Find description column (first text column)
        description_col = -1
        for col_idx in range(min(3, len(df.columns))):  # Check first 3 columns
            if self.is_description_column(df, col_idx):
                description_col = col_idx
                logger.info(f"Found description column at index: {description_col}")
                break
        
        # Step 2: Find numeric columns
        numeric_columns = []
        for col_idx in range(len(df.columns)):
            if self.is_numeric_column(df, col_idx):
                numeric_columns.append(col_idx)
        
        logger.info(f"Found numeric columns: {numeric_columns}")
        
        # Step 3: Find header zones
        header_zones = self.find_header_zones(df)
        logger.info(f"Header zones found: {header_zones}")
        
        # Step 4: Map headers to numeric columns
        column_mapping = {
            'description': description_col,
            'n_debit': -1,
            'n_credit': -1,
            'n1_debit': -1,
            'n1_credit': -1
        }
        
        # Determine N and N-1 zones based on headers and years
        n_zone_cols = set()
        n1_zone_cols = set()
        
        # Use year information to determine zones
        for row_idx, col_idx in header_zones['years']:
            year_value = str(df.iloc[row_idx, col_idx]).strip()
            if re.search(r'202[0-9]', year_value):  # Current years (2020+)
                n_zone_cols.add(col_idx)
                # Also check adjacent columns
                for adjacent in [col_idx-1, col_idx+1]:
                    if adjacent >= 0 and adjacent < len(df.columns):
                        n_zone_cols.add(adjacent)
            elif re.search(r'201[0-9]', year_value):  # Previous years (2010-2019)
                n1_zone_cols.add(col_idx)
                # Also check adjacent columns
                for adjacent in [col_idx-1, col_idx+1]:
                    if adjacent >= 0 and adjacent < len(df.columns):
                        n1_zone_cols.add(adjacent)
        
        # Use N/N-1 indicators to determine zones
        for row_idx, col_idx in header_zones['n']:
            n_zone_cols.add(col_idx)
            # Check nearby columns
            for adjacent in [col_idx-1, col_idx+1, col_idx+2, col_idx+3]:
                if adjacent >= 0 and adjacent < len(df.columns):
                    n_zone_cols.add(adjacent)
        
        for row_idx, col_idx in header_zones['n-1']:
            n1_zone_cols.add(col_idx)
            # Check nearby columns
            for adjacent in [col_idx-1, col_idx+1, col_idx+2, col_idx+3]:
                if adjacent >= 0 and adjacent < len(df.columns):
                    n1_zone_cols.add(adjacent)
        
        logger.info(f"N zone columns: {n_zone_cols}")
        logger.info(f"N-1 zone columns: {n1_zone_cols}")
        
        # Step 5: Assign DEBIT and CREDIT for N zone
        n_numeric_cols = [col for col in numeric_columns if col in n_zone_cols]
        n_numeric_cols.sort()
        
        for col_idx in n_numeric_cols:
            # Check if DEBIT header is near this column
            debit_near = False
            credit_near = False
            
            for header_row, header_col in header_zones['debit']:
                if abs(header_col - col_idx) <= 2:  # Within 2 columns
                    debit_near = True
                    break
            
            for header_row, header_col in header_zones['credit']:
                if abs(header_col - col_idx) <= 2:  # Within 2 columns
                    credit_near = True
                    break
            
            if debit_near and column_mapping['n_debit'] == -1:
                column_mapping['n_debit'] = col_idx
                logger.info(f"Assigned N DEBIT to column {col_idx}")
            elif credit_near and column_mapping['n_credit'] == -1:
                column_mapping['n_credit'] = col_idx
                logger.info(f"Assigned N CREDIT to column {col_idx}")
            elif column_mapping['n_debit'] == -1:  # First available numeric column
                column_mapping['n_debit'] = col_idx
                logger.info(f"Assigned N DEBIT to column {col_idx} (first available)")
            elif column_mapping['n_credit'] == -1:  # Second available numeric column
                column_mapping['n_credit'] = col_idx
                logger.info(f"Assigned N CREDIT to column {col_idx} (second available)")
        
        # Step 6: Assign DEBIT and CREDIT for N-1 zone
        n1_numeric_cols = [col for col in numeric_columns if col in n1_zone_cols]
        n1_numeric_cols.sort()
        
        for col_idx in n1_numeric_cols:
            # Check if DEBIT header is near this column
            debit_near = False
            credit_near = False
            
            for header_row, header_col in header_zones['debit']:
                if abs(header_col - col_idx) <= 2:  # Within 2 columns
                    debit_near = True
                    break
            
            for header_row, header_col in header_zones['credit']:
                if abs(header_col - col_idx) <= 2:  # Within 2 columns
                    credit_near = True
                    break
            
            if debit_near and column_mapping['n1_debit'] == -1:
                column_mapping['n1_debit'] = col_idx
                logger.info(f"Assigned N-1 DEBIT to column {col_idx}")
            elif credit_near and column_mapping['n1_credit'] == -1:
                column_mapping['n1_credit'] = col_idx
                logger.info(f"Assigned N-1 CREDIT to column {col_idx}")
            elif column_mapping['n1_debit'] == -1:  # First available numeric column
                column_mapping['n1_debit'] = col_idx
                logger.info(f"Assigned N-1 DEBIT to column {col_idx} (first available)")
            elif column_mapping['n1_credit'] == -1:  # Second available numeric column
                column_mapping['n1_credit'] = col_idx
                logger.info(f"Assigned N-1 CREDIT to column {col_idx} (second available)")
        
        # Step 7: Fallback assignment for unassigned columns
        remaining_numeric = [col for col in numeric_columns if col not in 
                           [column_mapping['n_debit'], column_mapping['n_credit'], 
                            column_mapping['n1_debit'], column_mapping['n1_credit']] 
                           and col != -1]
        
        # Fill any missing assignments
        if column_mapping['n_debit'] == -1 and remaining_numeric:
            column_mapping['n_debit'] = remaining_numeric.pop(0)
            logger.info(f"Fallback: Assigned N DEBIT to column {column_mapping['n_debit']}")
        
        if column_mapping['n_credit'] == -1 and remaining_numeric:
            column_mapping['n_credit'] = remaining_numeric.pop(0)
            logger.info(f"Fallback: Assigned N CREDIT to column {column_mapping['n_credit']}")
        
        if column_mapping['n1_debit'] == -1 and remaining_numeric:
            column_mapping['n1_debit'] = remaining_numeric.pop(0)
            logger.info(f"Fallback: Assigned N-1 DEBIT to column {column_mapping['n1_debit']}")
        
        if column_mapping['n1_credit'] == -1 and remaining_numeric:
            column_mapping['n1_credit'] = remaining_numeric.pop(0)
            logger.info(f"Fallback: Assigned N-1 CREDIT to column {column_mapping['n1_credit']}")
        
        logger.info(f"Final column mapping: {column_mapping}")
        return column_mapping

class EnhancedTCRPostProcessor:
    """Enhanced TCR post-processor with dynamic column detection and ocr_actif row matching"""
    
    def __init__(self):
        self.template_rows = self._flatten_template_rows()
        self.column_detector = DynamicTCRColumnDetector()
        
    def _flatten_template_rows(self) -> List[str]:
        """Flatten all template rows into a single list"""
        rows = []
        for section in TCR_FINANCIAL_STRUCTURE["sections"].values():
            rows.extend(section)
        return rows
    
    def extract_value_from_column(self, row, col_idx):
        """Extract a value from a specific column in a row"""
        if col_idx < 0 or col_idx >= len(row):
            return None
        
        val = row[col_idx]
        if val is not None and str(val).strip() and str(val).strip().lower() != 'nan':
            # Check if it's a number or looks like a number with spaces
            if isinstance(val, (int, float)) or (isinstance(val, str) and any(c.isdigit() for c in val)):
                # Skip values that look like years
                if str(val) not in ["2021", "2020", "2019", "2022", "2023", "2024"]:
                    return val
        return None
    
    def create_structured_tcr_data(self, raw_df):
        """Create structured TCR data using dynamic column detection and ocr_actif row matching"""
        if raw_df is None or raw_df.empty:
            return pd.DataFrame(columns=TCR_FINANCIAL_STRUCTURE["columns"])
        
        logger.info("Creating structured TCR data using dynamic column detection and enhanced row matching...")
        
        # Step 1: Detect column structure dynamically
        column_mapping = self.column_detector.detect_column_structure(raw_df)
        
        # Step 2: Extract descriptions from the description column (ocr_actif approach)
        description_col = column_mapping['description']
        raw_descriptions = []
        
        if description_col >= 0:
            for idx, row in raw_df.iterrows():
                # Look for descriptions in the first few columns, prioritizing the detected description column
                for col_check in [description_col] + [i for i in range(min(3, len(raw_df.columns))) if i != description_col]:
                    if col_check < len(row):
                        val = row[col_check]
                        if val is not None and str(val).strip() and str(val).strip().lower() != 'nan':
                            # If it's not a pure number, it might be a description
                            if not str(val).replace(' ', '').replace(',', '').isdigit():
                                raw_descriptions.append((idx, val))
                                break  # Found description for this row, move to next row
        else:
            # Fallback: look in first column if no description column detected
            for idx, row in raw_df.iterrows():
                val = row[0] if len(row) > 0 else None
                if val is not None and str(val).strip() and str(val).strip().lower() != 'nan':
                    if not str(val).replace(' ', '').replace(',', '').isdigit():
                        raw_descriptions.append((idx, val))
        
        logger.info(f"Found {len(raw_descriptions)} description rows")
        
        # Step 3: Match template rows with extracted descriptions (ocr_actif approach)
        # Get all structured rows from the TCR structure
        all_structured_rows = self.template_rows
        
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
            
            if best_idx >= 0 and score > 0.3:  # Lowered threshold slightly
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
            if raw_idx < 0 or score <= 0.3:
                continue
            
            # Skip if this raw row has already been used
            if raw_idx in used_raw_rows:
                continue
            
            # Assign this raw row to the structured description
            assigned_matches[structured_row_desc] = raw_idx
            used_raw_rows.add(raw_idx)
            
            # Debug information
            raw_desc = raw_descriptions[[i for i, (idx, _) in enumerate(raw_descriptions) if idx == raw_idx][0]][1]
            logger.info(f"Matching '{structured_row_desc}' -> '{raw_desc}' (score: {score:.2f})")
        
        # Step 4: Build final structured data
        for structured_row_desc in all_structured_rows:
            # Initialize a row with the structure description and None values
            new_row = [structured_row_desc, None, None, None, None]
            
            # If we have a match for this structured row
            if structured_row_desc in assigned_matches:
                raw_idx = assigned_matches[structured_row_desc]
                raw_row = raw_df.iloc[raw_idx].tolist()
                
                # Extract values using column mapping
                if column_mapping['n_debit'] >= 0:
                    new_row[1] = self.extract_value_from_column(raw_row, column_mapping['n_debit'])
                
                if column_mapping['n_credit'] >= 0:
                    new_row[2] = self.extract_value_from_column(raw_row, column_mapping['n_credit'])
                
                if column_mapping['n1_debit'] >= 0:
                    new_row[3] = self.extract_value_from_column(raw_row, column_mapping['n1_debit'])
                
                if column_mapping['n1_credit'] >= 0:
                    new_row[4] = self.extract_value_from_column(raw_row, column_mapping['n1_credit'])
            else:
                # No match found for this structured row
                logger.info(f"No good match found for '{structured_row_desc}'")
            
            # Add the row to our structured data
            structured_data.append(new_row)
        
        # Create DataFrame
        result_df = pd.DataFrame(structured_data, columns=TCR_FINANCIAL_STRUCTURE["columns"])
        
        # Apply enhanced number cleaning
        result_df = apply_number_cleaning_to_dataframe(result_df)
        
        logger.info(f"Structured TCR processing complete. Processed {len(result_df)} rows.")
        return result_df

def combine_tcr_documents(results_list: List[Dict]) -> pd.DataFrame:
    """Combine multiple TCR documents into a single structured document"""
    logger.info("Creating combined TCR document...")
    
    if not results_list:
        return pd.DataFrame(columns=TCR_FINANCIAL_STRUCTURE["columns"])
    
    # Initialize combined data with template structure
    combined_data = []
    template_rows = []
    for section in TCR_FINANCIAL_STRUCTURE["sections"].values():
        template_rows.extend(section)
    
    # Create combined structure
    for template_row in template_rows:
        combined_row = {
            'Rubriques': template_row,
            'N_DEBIT': None,
            'N_CREDIT': None,
            'N-1_DEBIT': None,
            'N-1_CREDIT': None
        }
        
        # Look for this row in each document and merge values
        for result in results_list:
            if 'structured_df' in result and result['structured_df'] is not None:
                structured_df = result['structured_df']
                
                # Find matching row in this document
                for _, row in structured_df.iterrows():
                    if pd.notna(row['Rubriques']) and str(row['Rubriques']).strip():
                        # Use exact match or high similarity
                        if (row['Rubriques'] == template_row or 
                            SequenceMatcher(None, str(row['Rubriques']).lower(), 
                                          template_row.lower()).ratio() > 0.8):
                            
                            # Merge values (prioritize non-null values)
                            for col in ['N_DEBIT', 'N_CREDIT', 'N-1_DEBIT', 'N-1_CREDIT']:
                                if pd.notna(row[col]) and str(row[col]).strip() and str(row[col]).strip().lower() != 'none':
                                    if combined_row[col] is None:
                                        combined_row[col] = row[col]
                                    else:
                                        # If both documents have values, prefer the non-empty one
                                        current_val = str(combined_row[col]).strip()
                                        new_val = str(row[col]).strip()
                                        if not current_val or current_val.lower() in ['none', 'nan']:
                                            combined_row[col] = row[col]
                            break
        
        combined_data.append(combined_row)
    
    # Create combined DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Apply number cleaning to the combined dataframe
    combined_df = apply_number_cleaning_to_dataframe(combined_df)
    
    logger.info(f"Combined document created with {len(combined_df)} rows")
    return combined_df

def process_image(imgpath, tab_ext, bbox_output=None, use_structure=True):
    """Process an image and extract table data"""
    try:
        # Check if input is a PDF file
        if imgpath.lower().endswith('.pdf'):
            # Use in-memory PDF processing
            pdf_processor = PDFProcessor(input_dir=None, output_dir=None)
            preprocessed_image = pdf_processor.process_pdf_in_memory(imgpath)
            
            if preprocessed_image is None:
                logger.error(f"❌ No tables found in PDF: {imgpath}")
                return None, None, None
            
            # Convert numpy array to PIL Image and save temporarily
            if isinstance(preprocessed_image, np.ndarray):
                preprocessed_image = Image.fromarray(preprocessed_image)
            
            # Save temporarily for table extraction
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                preprocessed_image.save(temp_path)
            
            try:
                # Extract table data from the temporary file
                result = tab_ext.detect(temp_path)
                if result and len(result) == 2:
                    (raw_df, cleaned_df), bbox = result
                else:
                    return None, None, None
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            # Original image processing for non-PDF files
            result = tab_ext.detect(imgpath)
            if result and len(result) == 2:
                (raw_df, cleaned_df), bbox = result
            else:
                return None, None, None
        
        logger.info(f"✅ Successfully extracted table from {imgpath}")
        
        # Apply structure normalization if enabled
        if use_structure and raw_df is not None and not raw_df.empty:
            tcr_processor = EnhancedTCRPostProcessor()
            structured_df = tcr_processor.create_structured_tcr_data(raw_df)
            return raw_df, cleaned_df, structured_df
        
        return raw_df, cleaned_df, None
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {imgpath}")
        logger.error(traceback.format_exc())
        return None, None, None

def process_pdf_with_enhanced_matching(pdf_path: str, output_dir: str = "output"):
    """Process PDF using enhanced dynamic column detection and row matching"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize processors
        logger.info("Initializing processors...")
        
        # Initialize TableExtraction
        try:
            tab_ext = TableExtraction()
            logger.info("Table extractor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize table extractor: {e}")
            return None
        
        logger.info(f"Processing file: {pdf_path}")
        
        # Process the file using dynamic detection
        raw_df, cleaned_df, structured_df = process_image(pdf_path, tab_ext, use_structure=True)
        
        if raw_df is not None and not raw_df.empty:
            logger.info(f"Extracted {len(raw_df)} rows from {pdf_path}")
            
            if structured_df is not None and not structured_df.empty:
                # Generate output files
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Raw output
                raw_html_path = os.path.join(output_dir, f"{base_name}_raw_{timestamp}.html")
                dataframe_to_html_report(raw_df, raw_html_path, f"Raw OCR - {base_name}")
                
                # Structured output
                structured_html_path = os.path.join(output_dir, f"{base_name}_enhanced_structured_{timestamp}.html")
                dataframe_to_html_report(structured_df, structured_html_path, 
                                       f"Enhanced TCR with Row Matching - {base_name}")
                
                logger.info(f"Generated reports: {raw_html_path}, {structured_html_path}")
                
                return [{
                    'page': 1,
                    'file_path': pdf_path,
                    'raw_df': raw_df,
                    'structured_df': structured_df,
                    'raw_html': raw_html_path,
                    'structured_html': structured_html_path
                }]
            else:
                logger.warning("Enhanced structured processing returned empty result")
        else:
            logger.warning("No table data extracted")
        
        return None
        
    except Exception as e:
        logger.error(f"Error in enhanced processing: {e}")
        logger.error(traceback.format_exc())
        return None

def dataframe_to_html_report(df: pd.DataFrame, output_html_path: str, title="Enhanced TCR Financial Report"):
    """Generate HTML report with enhanced styling"""
    
    style = """
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        h1 { 
            color: #2c3e50; 
            text-align: center; 
            margin-bottom: 30px; 
            font-size: 28px; 
        }
        .enhanced-features {
            background: #e8f5e8;
            padding: 15px;
            border-left: 4px solid #27ae60;
            margin-bottom: 20px;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
            font-size: 14px; 
        }
        th, td { 
            padding: 12px 8px; 
            text-align: left; 
            border: 1px solid #ddd; 
        }
        th { 
            background-color: #34495e; 
            color: white; 
            font-weight: bold; 
            position: sticky; 
            top: 0; 
        }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f5f5f5; }
        .numeric { text-align: right; font-family: 'Courier New', monospace; }
        .description { font-weight: 500; }
        .empty-cell { color: #bdc3c7; font-style: italic; }
        .negative { color: #e74c3c; }
    </style>
    """
    
    def format_cell(value, col_name):
        if pd.isna(value) or str(value).strip() in ['', '-', 'nan']:
            return '<span class="empty-cell">-</span>'
        
        if col_name in ['N_DEBIT', 'N_CREDIT', 'N-1_DEBIT', 'N-1_CREDIT']:
            try:
                # Handle different number formats
                value_str = str(value).replace(',', '.')
                if value_str.startswith('-'):
                    num_val = float(value_str)
                    return f'<span class="numeric negative">{num_val:,.0f}</span>'
                else:
                    num_val = float(value_str)
                    return f'<span class="numeric">{num_val:,.0f}</span>'
            except:
                return f'<span class="numeric">{str(value)}</span>'
        elif col_name == 'Rubriques':
            return f'<span class="description">{value}</span>'
        
        return str(value)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        {style}
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <div class="enhanced-features">
                <strong>🎯 Enhanced Features:</strong> 
                • Dynamic Column Detection 
                • OCR_Actif Row Matching Algorithm 
                • Advanced Number Cleaning (handles spaces, dots, negative numbers)
                • Template-based Structure Mapping
            </div>
            <table>
                <thead>
                    <tr>
                        {''.join([f'<th>{col}</th>' for col in df.columns])}
                    </tr>
                </thead>
                <tbody>
    """
    
    for idx, row in df.iterrows():
        html_content += '<tr>\n'
        for col in df.columns:
            formatted_cell = format_cell(row[col], col)
            html_content += f'<td>{formatted_cell}</td>\n'
        html_content += '</tr>\n'
    
    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Enhanced TCR HTML report generated: {output_html_path}")

# Main execution function
def main():
    # Default configuration
    default_files = ["./src/input/TCR9.pdf"]
    default_output_dir = "./src/output/tcr_enhanced"
    
    parser = argparse.ArgumentParser(description='Enhanced TCR Financial Statement Extraction Tool - With Advanced Number Cleaning & Row Matching')
    parser.add_argument('--input', type=str, nargs='+', default=default_files,
                        help='Path(s) to the TCR financial file(s) (PDF or image). Can specify 1 or 2 files.')
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                        help=f'Directory to save output files (default: {default_output_dir})')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    logger.info("=== Enhanced OCR-TCR Processing with Advanced Features ===")
    logger.info(f"Input files: {args.input}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Process each input file
    all_results = []
    for input_path in args.input:
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            continue
        
        logger.info(f"Processing file: {input_path}")
        results = process_pdf_with_enhanced_matching(input_path, args.output_dir)
        
        if results:
            all_results.extend(results)
            logger.info(f"✅ Successfully processed: {input_path}")
            for result in results:
                if 'structured_html' in result:
                    logger.info(f"📄 Enhanced Report: {result['structured_html']}")
        else:
            logger.error(f"❌ Failed to process: {input_path}")
    
    # If multiple files were processed, create a combined report
    if len(all_results) > 1:
        logger.info("Creating combined report from multiple files...")
        combined_df = combine_tcr_documents(all_results)
        
        if not combined_df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_html_path = os.path.join(args.output_dir, f"combined_enhanced_tcr_{timestamp}.html")
            dataframe_to_html_report(combined_df, combined_html_path, "Combined Enhanced TCR Report")
            logger.info(f"📄 Combined Enhanced Report: {combined_html_path}")
    
    if all_results:
        logger.info(f"✅ Enhanced processing completed successfully! Processed {len(all_results)} file(s).")
    else:
        logger.error("❌ No files were successfully processed!")

if __name__ == "__main__":
    main()
