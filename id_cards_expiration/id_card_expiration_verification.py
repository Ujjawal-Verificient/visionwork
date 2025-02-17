import re
import logging
from datetime import datetime


class ExpiryDateExtractor:
    """
    A class that extracts and validates expiry dates from OCR-extracted text.

    Methods:
        - extract_all_dates(ocr_text): Extracts all dates from OCR text.
        - exclude_15yrs_old_dates(all_dates): Removes dates older than 15 years.
        - exclude_issue_dates(ocr_text, all_dates): Filters out issued dates.
        - select_future_date_as_expiry(all_dates): Selects a future date (from today onward) as the expiry date.
        - find_expiry_using_keywords(ocr_text, all_dates): Identifies expiry dates using keywords.
        - select_latest_date_as_expiry(all_dates): Selects the latest date as expiry.
        - apply_expiry_selection_rules(all_dates): Applies priority rules to determine the expiry date.
        - extract_expiry_date(ocr_text): Extracts and selects the most relevant expiry date.
        - is_card_expired(ocr_text): Checks if the extracted expiry date has passed.
    """

    MONTHS = [
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
    ]
    MONTHS_REGEX = "|".join(MONTHS)

    EXPIRY_KEYWORDS = ["EXP", "Expiry", "Expiry Date", "Expires", "Expiration", "Date of Expiry"]
    ISSUE_KEYWORDS = ["Iss", "Issue", "Issued", "Issue Date", "Date of Issue", "Until"]

    DATE_PATTERNS = [
        # (r"(\d{1,2}/\d{1,2}/\d{4})", "%d/%m/%Y"),  # MM/DD/YYYY or DD/MM/YYYY
        (r"(\d{1,2}-\d{1,2}-\d{4})", "%d-%m-%Y"),  # MM-DD-YYYY or DD-MM-YYYY
        # (r"(\d{4}/\d{1,2}/\d{1,2})", "%Y/%m/%d"),  # YYYY/MM/DD
        (r"(\d{4}-\d{1,2}-\d{1,2})", "%Y-%m-%d"),  # YYYY-MM-DD

        (r"(\d{1,2}[/ ]\d{1,2}[/ ]\d{4})", "%d/%m/%Y"),
        (r"(\d{4}[/ ]\d{1,2}[/ ]\d{1,2})", "%Y/%m/%d"),


        (fr"(\d{{1,2}}\s*(?:{MONTHS_REGEX})\s*,?\s*\d{{4}})", "%d %B %Y"),  # DD Month YYYY
    ]

    def __init__(self):
        # self.today = datetime.today()
        self.today = datetime(2021, 1, 1)
        self.found_issue_date = False

    def extract_year_from_date(self, match):
        """
        Extracts the year from a date string.

        Args:
            match (str): A date string.

        Returns:
            str or None: Standardized date format (YYYY-12-31) or None if extraction fails or date is not in between 2000-2099.
        """
        try:
            year_match = re.search(r"\b(20\d{2})\b", match)  # Extract years between 2000-2099
            if year_match:
                return f"{year_match.group()}-12-31"  # Convert to YYYY-12-31 format
        except Exception as e:
            logging.error(f"Error extracting year from match '{match}': {e}")
        return None

    def extract_all_dates(self, ocr_text):
        """
        Extracts all date from OCR text.

        Args:
            ocr_text (str): Text extracted from OCR.

        Returns:
            list: A sorted list of extracted date tuples [(original_date, standardized_date), ...].
        """
        found_dates = []
        seen_dates = set()

        try:
            ocr_text = str(ocr_text).strip().lower() if isinstance(ocr_text, str) else ""
            if ocr_text:
                for pattern, _ in self.DATE_PATTERNS:
                    matches = re.findall(pattern, ocr_text, re.IGNORECASE)
                    for match in matches:
                        standardized_date = self.extract_year_from_date(match)
                        if standardized_date and standardized_date not in seen_dates:
                            found_dates.append((match, standardized_date))
                            seen_dates.add(standardized_date)
        except Exception as e:
            logging.error(f"Error extracting dates: {e}")

        return sorted(found_dates, key=lambda x: x[1])

    def exclude_15yrs_old_dates(self, all_dates):
        """
        Removes dates that are more than 15 years old.

        Args:
            all_dates (list): List of (original_date, standardized_date) tuples.

        Returns:
            list: Filtered list of dates within the last 15 years.
        """

        recent_dates = [
            (original_date, standardized_date)
            for original_date, standardized_date in all_dates
            if datetime.strptime(standardized_date, "%Y-%m-%d").year >= (self.today.year - 15)
        ]

        return recent_dates

    def exclude_issue_dates(self, ocr_text, all_dates):
        """
        Filters out card issued dates on basis of keyword.

        Args:
            ocr_text (str): OCR text.
            all_dates (list): List of (original_date, standardized_date) tuples.

        Returns:
            list: List of dates that are NOT linked to issue-related keywords.
        """
        self.found_issue_date = False
        filtered_dates = []
        for original_date, standardized_date in all_dates:
            if any(re.search(rf"{kw.lower()}[:\s]*{original_date}", ocr_text, re.IGNORECASE) for kw in self.ISSUE_KEYWORDS):
                self.found_issue_date = True
            else:
                filtered_dates.append((original_date, standardized_date))
        return filtered_dates

    def find_expiry_using_keywords(self, ocr_text, all_dates):
        """
        Finds expiry date based on expiry-related keywords.

        Args:
            ocr_text (str): OCR text.
            all_dates (list): List of (original_date, standardized_date) tuples.

        Returns:
            str or None: Standardized expiry date (YYYY-MM-DD) or None if not found.
        """
        for original_date, standardized_date in all_dates:
            if any(re.search(rf"{kw.lower()}[:\s]*{original_date}", ocr_text, re.IGNORECASE) for kw in self.EXPIRY_KEYWORDS):
                return standardized_date
        return None

    def select_future_date_as_expiry(self, all_dates):
        """
        Selects a future date (from today onward) as the expiry date.

        Args:
            all_dates (list): List of (original_date, standardized_date) tuples.

        Returns:
            str or None: The future expiry date (YYYY-MM-DD) or None if no future dates are found.
        """
        future_dates = [d[1] for d in all_dates if datetime.strptime(d[1], "%Y-%m-%d") > self.today]

        if future_dates:
            return max(future_dates, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

        return None
    
    def select_latest_date_as_expiry(self, all_dates):
        """
        Selects the latest date from the remaining expiry candidates.

        This method is called only if multiple dates are still present after applying other expiry selection rules.

        Args:
            all_dates (list): List of (original_date, standardized_date) tuples.

        Returns:
            str or None: The latest expiry date (YYYY-MM-DD) or None if no dates are available.
        """
        if not all_dates:
            return None

        standardized_dates = [standardized_date for _, standardized_date in all_dates]
        latest_date = max(standardized_dates, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

        return latest_date

    def apply_expiry_selection_rules(self, ocr_text, all_dates):
        """
        Applies rules in priority order to determine the expiry date.

        Selection Process:
        1. First, attempts to find a future date as expiry. 
        2. If no future date is found, searches for an expiry date using expiry-related keywords.
        3. If no expiry date is found after applying the above rules, selects the latest date from the remaining dates as expiry date.

        Args:
            all_dates (list): List of (original_date, standardized_date) tuples.

        Returns:
            str or None: The selected expiry date (YYYY-MM-DD) or None if no valid expiry date found.
        """
        # Step 1: Try selecting a future expiry date
        expiry_date = self.select_future_date_as_expiry(all_dates)
        if expiry_date:
            return expiry_date

        # Step 2: Try finding an expiry date using keywords
        expiry_date = self.find_expiry_using_keywords(ocr_text, all_dates)
        if expiry_date:
            return expiry_date

        # Step 3: If multiple valid dates are still present, select the latest one
        if len(all_dates) > 1 or (self.found_issue_date and all_dates):
            return self.select_latest_date_as_expiry(all_dates)

        return None

    def extract_expiry_date(self, ocr_text):
        """
        Extracts and determines the expiry date from OCR text.

        Args:
            ocr_text (str): OCR-extracted text.

        Returns:
            str or None: The determined expiry date (YYYY-MM-DD) or None if not found.
        """
        try:
            all_dates = self.extract_all_dates(ocr_text)
            if not all_dates:
                return None

            all_dates = self.exclude_15yrs_old_dates(all_dates)
            all_dates = self.exclude_issue_dates(ocr_text, all_dates)

            expiry_date = self.apply_expiry_selection_rules(ocr_text, all_dates)
            return expiry_date
        except Exception as e:
            logging.error(f"Error extracting expiry date: {e}")
        return None

    def is_card_expired(self, ocr_text):
        """
        Determines if the extracted expiry date has passed.

        Args:
            ocr_text (str): OCR-extracted text.

        Returns:
            bool or None:
                - True  → if the card is expired
                - False → if the card is still valid
                - None  → Could not verify.
        """
        try:
            expiry_date_str = self.extract_expiry_date(ocr_text)
            if expiry_date_str:
                return datetime.strptime(expiry_date_str, "%Y-%m-%d") < self.today
        except Exception as e:
            logging.error(f"Error checking card expiry: {e}")
        return None

text = """REPUBLICA DEL PERU REGISTRO NACIONAL DE IDENTIFICACION Y ESTADO CIVIL
73150832
QUISPE
CUI
DOCUMENTO NACIONAL DE IDENTIDAD DNI 73150832-7
Primer Apellido
QUISPE
Segundo Apellido
SOLORZANO
Pre Nombres
BELEN ALEXA
Nacimiento: Fecha y Ubigeo
07 04 2004
Sexo
140108
Fecha Inscripción
20 02 2009
Fecha Emisión
19 01 2019
Fecha Caducidad
07 04 2021
F
I<PER73150832<9<<<<<<<<<<<<<<<
0404077F2104072PER<<<<<<<<<<<4
QUISPE<<BELEN<ALEXA<<<<<<<<<<<"""

extractor = ExpiryDateExtractor()

extracted_dates = extractor.is_card_expired(text)
print(extracted_dates)