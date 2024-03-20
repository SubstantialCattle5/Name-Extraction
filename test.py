from functools import wraps
import json
import random
import re
import time

import requests

from process_report import extract_text_from_image_with_textract

API_TOKEN = ""
API_URL = "https://api-inference.huggingface.co/models/Jean-Baptiste/roberta-large-ner-english"


data = list()


def calculate_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        print(f"Execution time for {func.__name__}: {execution_time} ms")
        return result

    return wrapper

def standardize_name(name):
    """Converts name to lowercase and removes middle initials."""
    return re.sub(r'\b[a-z]\b\s*', '', name.lower()).strip()
    

def fetch_name(groups: list, ocr_text: str, type: str = "DOC") -> str | None:
    """
    Extracts names from OCR text, considering context window and honorifics.

    Args:
        groups: List of entity groups from the OCR result.
        ocr_text: The OCR-processed text for context.
        type: Type of document (DOC for doctor names, otherwise for general names).

    Returns:
        A standardized name (for non-DOC types) or a set of standardized names (for DOC type),
        or None if no names are found.
    """

    WINDOW = 20
    names = set()
    seen_names = set()

    honorifics = {
        "DOC": ["Dr", "MBBS"],
        "DEFAULT": ["Name", "Mr", "Ms", "Mrs", "Master", "Patient:", "Patient Name:"]
    }

    for group in groups:
        if group.get('entity_group') != 'PER' or group.get('score') <= 0.60 or len(group.get('word')) < 3:
            continue  # Efficiently skip non-matching groups

        start_index = max(0, group['start'] - WINDOW)
        end_index = min(len(ocr_text), group["end"] + WINDOW)
        context_window = ocr_text[start_index:end_index]

        honorific_pattern = r"\b" + "|".join(re.escape(h) for h in honorifics[type])
        regex_match = re.search(honorific_pattern, context_window, re.IGNORECASE)

        if not regex_match:
            continue  # Early exit if no honorific is found

        candidate_name = ocr_text[group['start']:group['end']].strip()
        fixed_name = standardize_name(candidate_name)
        
        # remove all the non alphanumeric characters
        check_name = ''.join(e for e in fixed_name if e.isalnum())
        
        if check_name not in seen_names:
            if type == "DOC":
                names.add(fixed_name)
            else:
                return fixed_name  # Return the first name for non-DOC types

        seen_names.add(check_name)

    return names if type == "DOC" and len(names) else None  # Return names for DOC or None



# * NER call
@calculate_execution_time
def model_query(ocr) -> None:
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = None
    response = query({"inputs": ocr})

    while 'error' in response and response[
            'error'] == 'Model Jean-Baptiste/roberta-large-ner-english is currently loading':
        print(
            f"Model is still loading. Waiting for {response['estimated_time']} seconds...")
        time.sleep(response['estimated_time'])
        response = query({"inputs": data['ocr_text'].tolist()})

    if 'error' not in response:
        output = response
    else:
        print("Error: Model failed to load.")

    return output


def fetch_image():
    for num in range(1, 28):
        image_path = f"dev/{num}.jpg"
        print(image_path)
        with open(image_path, "rb") as image_path:
            image_bytes = image_path.read()
            res = extract_text_from_image_with_textract(
                user_id=random.randrange(50),
                message_id=random.randrange(50),
                process_type='Name Extraction',
                file='file',
                image_bytes=image_bytes
            )
            ner_response = model_query(res)
            json_body = {
                "doc_ans": fetch_name(groups=ner_response, ocr_text=res, honorefics=DOC_HONER),
                "user_ans": fetch_name(groups=ner_response, ocr_text=res, honorefics=USER_HONER),
                "ocr_text": res,
                "ner_output": ner_response,
            }

            print(f"JSON BODY WITH INDEX {num} - {json_body}")
            data.append(json_body)

    return data


if __name__ == "__main__":
    # data = fetch_image()
    # # Serializing json
    # json_object = json.dumps(data, indent=4)
    # # Writing to sample.json
    # with open("data.json", "w") as outfile:
    #     outfile.write(json_object)

    # Read the data.json file
    with open('data.json', 'r') as f:
        datas = json.load(f)

    for index,data in enumerate(datas):
        print(f' index - {index+1} -> {fetch_name(groups=data["ner_output"],ocr_text=data["ocr_text"], type="DOC")}' )
