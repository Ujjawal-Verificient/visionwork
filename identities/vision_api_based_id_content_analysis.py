import ast
import base64
import io
import logging
import os
import shlex
import sys
import tempfile
import time
import uuid
from subprocess import call

import numpy as np
from django.conf import settings
from fuzzywuzzy import fuzz
from google.auth.transport.requests import AuthorizedSession
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.oauth2 import service_account
from PIL import Image

client = vision.ImageAnnotatorClient.from_service_account_json(settings.GS_CREDENTIALS_FILE_LOCATION)
logger = logging.getLogger("identities")


g = 1
alpha = 1.2  # Simple contrast control
beta = 20  # Simple brightness control
confThreshold = 0.1
nmsThreshold = 0.4
inpWidth = 640
inpHeight = 640


class OnboardingScore:
    def __init__(self):
        self.associated_name_status = {}
        self.associated_name_fuzzy_score = {}
        self.frame = None
        self.boxFace = None
        self.fuzzy_threshold = 90
        self.face_on_facescan_detected = {}

    def generate_text_matching_score(self, str1, str2):
        Ratio = fuzz.ratio(str1.lower(), str2.lower())
        return Ratio

    def match_name_with_fuzzy_score(self, det_text, match_list):
        score = []
        matched_text = []

        def split(s):
            temp_s = ""
            for ch in s:
                if ch.isspace():
                    if temp_s:
                        yield temp_s
                        temp_s = ""
                else:
                    temp_s += ch
            if temp_s:
                yield temp_s

        det_text = list(split(det_text))
        det_filter_txt = []
        for txt in det_text:
            det_filter_txt += txt.split("\n")
            det_filter_txt += txt.split("-")
            det_filter_txt += txt.split("/")
            det_filter_txt += txt.split("_")

        for idx, text2 in enumerate(match_list):
            if not self.associated_name_status[text2]:
                for text1 in det_filter_txt:
                    ratio = self.generate_text_matching_score(text1, text2)
                    self.associated_name_fuzzy_score[text2].append(ratio)
                    if ratio > self.fuzzy_threshold:
                        self.associated_name_status[text2] = True

                        matched_text.append(text1)
                        score.append(ratio)

    def get_onboarding_score(self, json_data):
        """This function returns: score of id card content analysis and face verification."""
        st = time.time()
        id_card_detected = False
        face_on_id_detected = False
        id_detected_confidence = []
        face_on_facescan_detected_dict = {}
        fv_score = -1
        text_on_idcard = None

        try:
            self.frame = readbase64_using_pil(json_data["id_card_image"])
        except Exception as e:
            logger.error("Error while reading base64 image using pil : {}".format(e))
            return id_card_detected, face_on_id_detected, self.associated_name_status, {}, {}, {}, fv_score

        match_name_list = json_data["string_to_search"]

        for i in match_name_list:
            self.associated_name_status[i] = False
            self.associated_name_fuzzy_score[i] = []

        try:
            self.boxID, self.boxFace, face_on_facescan_detected_dict, fv_score, id_detected_confidence = detect_idcard(
                json_data
            )
            if len(self.boxID) > 0:
                id_card_detected = True
            if len(self.boxFace) > 0:
                face_on_id_detected = True

            if id_card_detected and face_on_id_detected:
                """ID card content analysis"""
                text_on_idcard = get_text_from_image(self.frame)
                # Checking exact names are recognised by OCR
                for word in self.associated_name_status:
                    self.associated_name_status[word] = is_text_available(word, str(text_on_idcard))
                    if self.associated_name_status[word]:
                        self.associated_name_fuzzy_score[word].append(100)
                # If failed to recognised exact names in text.
                if False in self.associated_name_status.values():
                    # Check names by matching every word in text with name by fuzzy score
                    self.match_name_with_fuzzy_score(text_on_idcard, match_name_list)

        except Exception as e:
            self.boxID = None
            self.boxFace = None
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.exception("ID verification error: {0}, {1}, {2}, {3}".format(e, exc_type, fname, exc_tb.tb_lineno))

        id_text_score = {}
        for key, values in self.associated_name_fuzzy_score.items():
            if values:
                id_text_score[key] = max(values)
            else:
                id_text_score[key] = 0

        id_content_result = (
            id_card_detected,
            face_on_id_detected,
            self.associated_name_status,
            id_text_score,
            id_detected_confidence,
            face_on_facescan_detected_dict,
            fv_score,
        )
        data_tobe_save = (self.boxID, self.boxFace, text_on_idcard)

        logger.info(
            "Time Taken by IDV Vision-API based service : {0} secs with Response | {1} ".format(
                round(time.time() - st, 2), id_content_result
            )
        )
        return id_content_result, data_tobe_save


def is_text_available(string, text):
    if string.lower() in text.lower():
        return True
    else:
        return False


def readbase64_using_pil(base64_image):
    try:
        base64_image = base64_image.split(",")[1]
    except Exception as e:
        pass
    nparr = np.fromstring(base64.b64decode(base64_image), np.uint8)
    bytes_io = bytearray(nparr)
    img = Image.open(io.BytesIO(bytes_io))
    return np.array(img.convert("RGB"))


def get_text_from_image(image):
    # The name of the image file to annotate
    file_name = tempfile.gettempdir() + "/" + str(uuid.uuid4()) + ".jpg"
    im = Image.fromarray(image)
    im.save(file_name)

    # Loads the image into memory
    with io.open(file_name, "rb") as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    st = time.time()
    response = client.text_detection(image=image)  # returns TextAnnotation
    logger.info("Time taken by Vision API: {} secs.".format(time.time() - st))

    annotations = response.text_annotations
    if len(annotations) > 0:
        text_result = annotations[0].description
    else:
        text_result = ""
    call(shlex.split("rm -rf {}".format(file_name)), shell=False)
    return text_result


def detect_idcard(json_data):
    st = time.time()
    """This Function calls cloud run API to verify id card.
       :type json_data: Dict with this format:
        {'id_card_image': base64_idcard_image, 'mode': 'idcard', 'facescan': base64_facescan, 'string_to_search': ['text']}
       :return Dict with this format: {'approved_status': False, 'log': result, 'reason': 'Name did not match with ID'}

       """
    api_url = settings.IDV_SERVICE_URL

    credentials = service_account.IDTokenCredentials.from_service_account_file(
        settings.GS_CREDENTIALS_FILE_LOCATION, target_audience=api_url
    )

    session = AuthorizedSession(credentials)
    service_response = session.post(api_url, json=json_data)

    logger.info(
        "IDV Service Response : {0} | Time taken to detect ID card : {1} secs.".format(
            service_response.text, time.time() - st
        )
    )

    detection_result = ast.literal_eval(service_response.text)

    if detection_result["status"]:
        (
            id_card_boxes,
            face_on_id_detected_boxes,
            face_on_facescan_detected_dict,
            fv_score,
            id_detected_confidence,
        ) = detection_result["result"]

        return (
            id_card_boxes,
            face_on_id_detected_boxes,
            face_on_facescan_detected_dict,
            fv_score,
            id_detected_confidence,
        )
    else:
        return [], [], [], [], []


def test_api():
    idcard_path = "../ui/static/img/policy3.png"
    facescan = idcard_path
    all_info = ["ANIKA", "MANDHANIA"]

    with open(idcard_path, mode="rb") as file:
        idcard = file.read()
    with open(facescan, mode="rb") as file:
        facescan = file.read()

    idcard = base64.encodebytes(idcard).decode("utf-8")
    facescan = base64.encodebytes(facescan).decode("utf-8")
    payload = {"id_card_image": idcard, "mode": "idcard", "facescan": facescan, "string_to_search": all_info}

    result = OnboardingScore().get_onboarding_score(json_data=payload)
