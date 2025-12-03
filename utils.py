from pathlib import Path
import csv

def load_prompts_from_file(filepath, column_name: str = "prompt"):
    """
    Loads prompts from a text file, with each line being a prompt.
    Args:
        filepath (str): The path to the text file.
    Returns:
        list[str]: A list of prompts.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found at {filepath}")
    
    if path.suffix.lower() == ".csv":
        prompts = []
        with open(path, newline='', encoding="utf-8") as f:
            try:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    # Fallback: treat as plain text lines
                    return []
                # Map header to lowercase for matching
                lower_header = [h.lower() for h in header]
                if column_name.lower() not in lower_header:
                    raise ValueError(
                        f"Column '{column_name}' not found in CSV {filepath}. "
                        f"Available columns: {header}"
                    )
                col_idx = lower_header.index(column_name.lower())
                for row in reader:
                    if col_idx < len(row):
                        val = row[col_idx].strip()
                        if val:
                            prompts.append(val)
            except csv.Error as e:
                raise ValueError(f"Error parsing CSV '{filepath}': {e}")
        return prompts
    
    with open(path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts