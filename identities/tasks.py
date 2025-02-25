import ast
import base64
import glob
import json
import logging
import os
import shlex
import time
import uuid as UUID
from subprocess import call

import boto
import numpy as np
import waffle
from celery import Celery
from celery.exceptions import MaxRetriesExceededError
from django.conf import settings
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from identities.utils import form_task_response
from identities.vision_api_based_id_content_analysis import OnboardingScore
from media.cassandra_models import (
    CSnapshotImage,
    CSnapshotImageByVideoId,
    CVideoByTestsessionId,
)
from media.utils import download_object, get_object_name, upload_processing_image
from proctoring.models import Incident
from testsessions.decorators import testsession_task
from testsessions.redis_processing import RedisVideo
from testsessions.utils import get_processing_configs
from visiontasks.facerec.util import extract_face_features
from visiontasks.utils import extract_and_match_faces, is_face_detected
from workflows.tasks import CeleryTaskResult

celery = Celery("identities", backend=settings.CELERY_RESULT_BACKEND, broker=settings.IDENT_TASK_BROKER)
logger = logging.getLogger("tasks")


@celery.task(name="identities.validate_face_img", queue="offline_tasks_finalized")
def validate_face_img(image):
    return _validate_face_img(image)


def _validate_face_img(image):
    """
    Yet to consider: Testsession will not be there. We will get a id-doc as input argument
    """
    logger.info("Started detecting face")

    result = is_face_detected(image, is_base64=True)
    logger.info("Ended detecting face. Result: {0}".format(result))

    return result


def issue_validate_face_img(image):
    func_with_params = validate_face_img.s(image).set(task_id=UUID.uuid4().hex)
    result = func_with_params.apply_async()

    logger.info("Validating face within 60secs. Task id: {0}".format(func_with_params.id))
    return result.get(timeout=60)


@celery.task(name="identities.verify_face_img", queue="offline_tasks_finalized")
def verify_face_img(image, student_id):
    return _verify_face_img(image, student_id)


def _verify_face_img(image, student_id):
    from students.models import Student

    logger.info("Started verifying face")
    stud = Student.objects.get(id=student_id)
    if (
        stud.user.lti_profile.institution.requires_veripass
        and not stud.user.lti_profile.institution.is_session_based_veripass_flow
    ):
        veripass_prof = stud.user.profile.get_veripass_data

        if veripass_prof["profile_status"] == "Approved":
            # perform face verification
            img_url = veripass_prof["scans"]["fs_scan"]["image_url"]
            result = extract_and_match_faces(
                img1_data=image, img1_type="base64", img2_data=img_url, img2_type="url", return_score=False
            )
            logger.info("Result of User VeriPass profile for student_id : {0} is {1}.".format(student_id, result))
            return result
        else:
            logger.info("User VeriPass profile is not yet approved!")
            return False  # , -1
    else:
        if stud.approved_ts:
            try:
                img_url = (
                    stud.approved_ts.test_session_meta.identity.face_scan_video.snapshots.order_by("timeindex")
                    .first()
                    .image.url
                )
                result = extract_and_match_faces(
                    img1_data=image, img1_type="base64", img2_data=img_url, img2_type="url", return_score=False
                )
                logger.info("Result of User VeriPass profile for student_id : {0} is {1}.".format(student_id, result))
                return result
            except Exception:
                return False
        return False


def face_verification_service_api(json_data, testsession_id=None):
    try:
        api_url = settings.FACE_VERIFICATION_SERVICE_URL

        credentials = service_account.IDTokenCredentials.from_service_account_file(
            settings.GS_CREDENTIALS_FILE_LOCATION, target_audience=api_url
        )

        session = AuthorizedSession(credentials)
        service_response = session.post(api_url, json=json_data)
        logger.info(
            "TestSession: {0} | Face verification service response | {1} ".format(
                testsession_id, service_response.text
            )
        )
        detection_result = ast.literal_eval(service_response.text)
        if not detection_result["status"]:
            logger.exception(
                "TestSession: {0} | Face verification service error | {1}".format(
                    testsession_id, detection_result["error"]
                )
            )
            return False

        return detection_result["result"].split("_")[0] == "matched"
    except Exception as e:
        logger.exception("TestSession: {0} | Face verification service error | {1}".format(testsession_id, e))
        return False


def verify_face_img_with_api(image, student_id, test_session):
    from students.models import Student

    logger.info("Started verifying face")
    stud = Student.objects.get(id=student_id)
    if (
        stud.user.lti_profile.institution.requires_veripass
        and not stud.user.lti_profile.institution.is_session_based_veripass_flow
    ):
        veripass_prof = stud.user.profile.get_veripass_data

        if veripass_prof["profile_status"] == "Approved":
            # perform face verification
            img_url = veripass_prof["scans"]["fs_scan"]["image_url"]
            # call CR API
            result = face_verification_service_api(
                {"facescan_base64": image, "onboarding_url": img_url}, testsession_id=test_session.id
            )
            logger.info("Result of User VeriPass profile for student_id : {0} is {1}.".format(student_id, result))
            return result
        else:
            logger.info("User VeriPass profile is not yet approved!")
            return False  # , -1
    else:
        if stud.approved_ts:
            try:
                img_url = (
                    stud.approved_ts.test_session_meta.identity.face_scan_video.snapshots.order_by("timeindex")
                    .first()
                    .image.url
                )
                result = face_verification_service_api({"facescan_base64": image, "onboarding_url": img_url})
                logger.info("Result of User VeriPass profile for student_id : {0} is {1}.".format(student_id, result))
                return result
            except Exception:
                return False
        return False


def issue_verify_face_img(image, student_id):
    func_with_params = verify_face_img.s(image, student_id).set(task_id=UUID.uuid4().hex)
    result = func_with_params.apply_async()

    logger.info("Verifying face within 60secs. Task id: {0}".format(func_with_params.id))
    return result.get(timeout=60)


def free_service_api(json_data):
    st = time.time()
    api_url = settings.IDV_OWN_SERVICE_URL

    credentials = service_account.IDTokenCredentials.from_service_account_file(
        settings.GS_CREDENTIALS_FILE_LOCATION, target_audience=api_url
    )

    session = AuthorizedSession(credentials)
    service_response = session.post(api_url, json=json_data)

    id_content_result = ast.literal_eval(service_response.text)

    return id_content_result


def test_free_service_api():
    idcard_path = "../ui/static/img/policy3.png"
    facescan = idcard_path
    all_info = ["ANIKA", "MANDHANIA"]

    with open(idcard_path, mode="rb") as file:
        idcard = file.read()
    with open(facescan, mode="rb") as file:
        facescan = file.read()

    idcard = base64.encodebytes(idcard).decode("utf-8")
    facescan = base64.encodebytes(facescan).decode("utf-8")
    json_data = {"id_card_image": idcard, "mode": "idcard", "facescan": facescan, "string_to_search": all_info}
    return free_service_api(json_data=json_data)


def get_name_list(id_names):
    si = []
    for i in id_names:
        si = si + [j for j in i.split(" ") if len(j) > 1]
    return si


def idv_service_api(json_data, test_session):
    """This Function calls cloud run API to verify id card.
    :param json_data: Dict with this format:
     {'id_card_image': base64_idcard_image, 'mode': 'idcard', 'facescan': base64_facescan, 'string_to_search': ['text']}
    :return Dict with this format: {'approved_status': False, 'log': result, 'reason': 'Name did not match with ID'}

    """
    test_id = test_session.test.id

    # Note: It will pass ID scans and will save respective facescans and ID-scans if this switch is enabled.
    # test_session.test.config.is_real_time_idv_face_match_required, this should be enable in order to
    # call idv_service_api.
    # Case 1: test_session.test.config.is_real_time_idv_face_match_required = True and
    # switch is actived for 'str(test_id) + "_save_scans_without_id_verification"' then it will save scans and
    # will not verify ID scans.
    # Case 2: test_session.test.config.is_real_time_idv_face_match_required = True and
    # switch is deactivated for 'str(test_id) + "_save_scans_without_id_verification"' then
    # it will it will save scans and will also verify ID scans.
    if waffle.switch_is_active(str(test_id) + "_save_scans_without_id_verification"):
        return {"approved_status": True, "log": "Saving ID scans", "reason": "Saved ID scans", "data_tobe_save": ""}

    if True:

        try:
            json_data["string_to_search"] = get_name_list(json_data["string_to_search"])
            id_content_result = OnboardingScore().get_onboarding_score(json_data=json_data)
            return analysis_result(id_content_result[0], json_data, test_session, data_tobe_save=id_content_result[1])
        except Exception as e:
            logger.exception("TestSession: {0} Error in Vision-based IDV Service | {1}".format(test_session.id, e))
            return {"approved_status": False, "log": e, "reason": "Name Does not Match"}
    else:
        id_content_result = free_service_api(json_data=json_data)
        if id_content_result["status"]:
            try:
                return analysis_result(id_content_result["result"], json_data, test_session)
            except Exception as e:
                logger.exception("TestSession: {0} Error in IDV own Service | {1}".format(test_session.id, e))
                return {"approved_status": False, "log": id_content_result["error"], "reason": "Name Does not Match"}

        else:
            logger.exception(
                "TestSession: {0} Error in IDV own Service | {1}".format(test_session.id, id_content_result["error"])
            )
            return {"approved_status": False, "log": id_content_result["error"], "reason": "Name Does not Match"}


def analysis_result(result_id_content_analysis, json_data, test_session, data_tobe_save=None):
    fv_threshold = 1.13
    max_fv_threshold = 1.5
    end_result = {}
    all_info = json_data["string_to_search"]

    (
        id_card_detected,
        face_on_id_detected,
        associated_name_status,
        id_text_score,
        id_detected_confidence,
        face_on_facescan_detected_dict,
        fv_score,
        is_card_expired,
    ) = result_id_content_analysis
    face_on_facescan_flag = [value for value in face_on_facescan_detected_dict.values()].count(True) > 0

    # consider name is not matched even if first name or last name is not matched
    # also the name-string should have more than 1 char
    is_text_not_matched = [associated_name_status[i] for i in all_info if len(i)>1].count(False) > 0
    # hence, name is considered as matched only if both first and last name matches
    is_text_matched = not is_text_not_matched

    if face_on_facescan_flag and face_on_id_detected and fv_score is not None:
        is_face_matched = fv_score[0] < fv_threshold
        is_face_matched_worst_case = fv_score[0] < max_fv_threshold
    else:
        is_face_matched = False
        is_face_matched_worst_case = False

    logger.info(
        "TestSession: {0} ID card verification result | {1}".format(test_session.id, result_id_content_analysis)
    )

    logger.info(
        "TestSession: {0} is_real_time_idv_face_match_required Status | {1}".format(
            test_session.id, test_session.test.config.is_real_time_idv_face_match_required
        )
    )

    # Todo: We can restrict imposter with high confidence using: is_face_matched_worst_case and not is_face_matched
    if not test_session.test.config.is_real_time_idv_face_match_required or json_data["facescan"] is None:

        end_result["approved_status"] = is_text_matched
    else:
        end_result["approved_status"] = is_text_matched and is_face_matched  # face_on_id_detected

    is_expiry_check_required = True # Replace is_expiry_check_required with test_session.test.config.is_expiry_check_required
    if is_expiry_check_required and is_card_expired:
        end_result["approved_status"] = False

    end_result["log"] = result_id_content_analysis
    end_result["data_tobe_save"] = data_tobe_save

    if not id_card_detected:
        end_result["reason"] = "No ID capture"
    elif not face_on_id_detected:
        end_result["reason"] = "face not clear"
    elif not is_text_matched:
        end_result["reason"] = "Name Does not Match"
    elif not is_face_matched:
        end_result["reason"] = "Photo matching score is below threshold"
    elif is_expiry_check_required and is_card_expired:  # Replace is_expiry_check_required with test_session.test.config.is_expiry_check_required
        end_result["reason"] = "ID card is expired"
    else:
        end_result["reason"] = "Name Does Match"

    # Need to disable for human in the loop functionality
    if True:
        incident_mapping = {
            "No ID capture": "Photo ID scan not captured",
            "face not clear": "Face was not clear in ID",
            "Name Does not Match": "Name did not match with ID",
            "ID card is expired": "Expired ID card detected",
        }

        violation_list = list(incident_mapping.values()) + ["Face did not match with ID", "Invalid Photo ID"]
        test_session.notifications_test_sessions.filter(
            violation_type__in=violation_list, is_real_time_incident=True
        ).update(is_valid=False)

        if not id_card_detected and not face_on_id_detected:
            incident, _ = Incident.objects.get_or_create(
                violation_type="Invalid Photo ID", test_session=test_session, is_real_time_incident=True, start_at=None
            )
            if not incident.is_valid:
                incident.is_valid = True
                incident.save()
        else:
            if end_result["reason"] in incident_mapping.keys():
                incident, _ = Incident.objects.get_or_create(
                    violation_type=incident_mapping[end_result["reason"]],
                    test_session=test_session,
                    is_real_time_incident=True,
                    start_at=None,
                )
                if not incident.is_valid:
                    incident.is_valid = True
                    incident.save()
            if not is_face_matched and id_card_detected and face_on_id_detected and json_data["facescan"] is not None:
                incident, _ = Incident.objects.get_or_create(
                    violation_type="Face did not match with ID",
                    test_session=test_session,
                    is_real_time_incident=True,
                    start_at=None,
                )
                if not incident.is_valid:
                    incident.is_valid = True
                    incident.save()
    try:
        end_result_string = json.dumps(end_result)
        if len(end_result_string) > 1000:
            end_result["data_tobe_save"] = data_tobe_save[:500]
    except Exception as ex:
        logger.info("Error in end_result truncation for session {0} | {1}".format(test_session.uuid, ex))
    return end_result


@celery.task(
    name="identities.extract_snapshots",
    queue="kickoff",
    max_retries=settings.MAX_RETRIES,
    default_retry_delay=60,
    base=CeleryTaskResult,
    bind=True,
)
@testsession_task
def extract_snapshots(self, testsession_id, test_session=None, task_name=None):

    if test_session.test_session_meta.is_curated or test_session.test_session_meta.is_offline_analysis_completed:
        logger.warning(
            "TestSession: {0} | This session is already curated or processed. Why extract_snapshots called!".format(
                testsession_id
            )
        )
        return form_task_response(test_session, "ok", task_name)
    if test_session.is_facial_scans():
        logger.warning(
            "TestSession: {0} | Face scan images exist. Why extract_snapshots called!".format(testsession_id)
        )
        return form_task_response(test_session, "ok", task_name)

    _extract_snapshots(testsession_id, test_session, task_name)

    return form_task_response(test_session, "ok", task_name)


def _extract_snapshots(testsession_id, test_session=None, task_name=None, save_features=False):
    is_redis_implementation = get_processing_configs(testsession_id, "is_redis_implementation")
    redis_snapshots_list = []
    if is_redis_implementation:
        redis_video_obj = RedisVideo(testsession_id)
        fscan_metadata = redis_video_obj.fetch_video_object("fscan")
        if not fscan_metadata:
            return form_task_response(test_session, "failed", task_name, reason="face scan video is missing")

        if test_session.is_facial_scans():
            return
        # TODO check if is_served_via_cloudfront has to be implemented
        stream_name = fscan_metadata["stream"]["stream_name"]
        download_object(stream_name, test_session, path=settings.SCREENCAST_PROCESS_FOLDER)
        object_name = get_object_name(stream_name)
    else:
        try:
            cvideos = CVideoByTestsessionId.objects.filter(testsession_id=testsession_id)
            cvideo_fscan_list = []
            # Now filter and collect all fscan videos whether morphed or non-morphed
            cvideo_fscan_list = [v for v in cvideos if "fscan" in v.title.lower()]
        except Exception as e:
            logger.error("TestSession: {0} | {1}".format(testsession_id, e))

        image_files = []
        for video in cvideo_fscan_list:
            stream = video.stream
            video_id = video.id
            # Think this is not required as we are serving all videos from cloudfront
            if not stream.is_served_via_cloudfront():
                try:
                    stream.convert_to_cloudfront()
                except MaxRetriesExceededError as e:
                    logger.error("TestSession: {0} | {1}".format(testsession_id, e))
                    return form_task_response(
                        test_session,
                        "failed",
                        task_name,
                        reason="{0}-->face scan video failed to convert to cloudfront. video_id: {1}".format(
                            task_name, video_id
                        ),
                    )

            # download streaming video
            try:
                stream.download_video(settings.SCREENCAST_PROCESS_FOLDER, testsession_id)
            except boto.exception.S3ResponseError as e:
                logger.error("TestSession: {0} | {1}".format(testsession_id, e))
                return form_task_response(
                    test_session,
                    "failed",
                    task_name,
                    reason="{0}-->s3responseerror thrown while extracting snapshots for video id: {1}".format(
                        task_name, video_id
                    ),
                )
        object_name = stream.get_object_name()

    video_filename = os.path.join(settings.SCREENCAST_PROCESS_FOLDER, os.path.basename(object_name))

    resolution = "scale=320:-1"
    snapshot_filename = os.path.join(settings.SCREENCAST_PROCESS_FOLDER, "%s_%s.jpg" % (testsession_id, "%d"))
    # call ffmpeg
    _extract_images(video_filename, resolution, snapshot_filename, testsession_id)
    image_files = glob.glob("%s/%s_*.jpg" % (settings.SCREENCAST_PROCESS_FOLDER, testsession_id))

    # Retry if not images
    _retry_count = 0
    while not image_files:
        _retry_count += 1
        msg = "TestSession: {0} | Face Scan image extration failed retrying: retry count {1}".format(
            testsession_id, _retry_count
        )
        # logger.info (msg)

        _extract_images(video_filename, resolution, snapshot_filename, testsession_id)
        image_files = glob.glob("%s/%s_*.jpg" % (settings.SCREENCAST_PROCESS_FOLDER, testsession_id))

        if _retry_count == 5:
            msg = "TestSession: {0} | Error while extracting face scan images. Function Retry Limit Reached".format(
                testsession_id
            )
            logger.error(msg)
            break

    # delete existing snapshots created for this video
    if not is_redis_implementation:
        for image in CSnapshotImageByVideoId.objects.filter(video_id=video_id):
            image.delete()

    snapshot_objects = []
    # add extracted images as SnapshotImage objects
    for image_file in image_files:
        # split only if image name is in the format: /tmp/2c690ac2-d5a6-40dc-a58d-6f193afdcbe5_1.jpg
        timeindex = image_file.split(".")[0].split("_")[1]
        timeindex = int(timeindex) - 1  # subtract by 1 for zero indexing
        face_features = ""
        if save_features and os.path.exists(image_file):
            face_features = extract_face_features(image_file)
            if not face_features:
                face_features = np.array([-1])
            else:
                face_features = face_features[-1]  # It has list of features
            if not is_redis_implementation:
                csnapshot = CSnapshotImage(
                    testsession_id=testsession_id, video_id=video_id, timeindex=timeindex, face_features=face_features
                )
        elif not is_redis_implementation:
            csnapshot = CSnapshotImage(testsession_id=testsession_id, video_id=video_id, timeindex=timeindex)
        # TODO: Bulk create Cassandra models
        if is_redis_implementation:
            object_name = upload_processing_image(image_file, test_session)
            if object_name:
                snapshot_dict = {
                    "timeindex": timeindex,
                    "is_visible": False,
                    "image_name": object_name,
                    "face_features": face_features.tolist(),
                }
                redis_snapshots_list.append(snapshot_dict)
        else:
            file_uploaded_size = csnapshot.save_image(image_file, testsession_id=testsession_id)
            # logger.info ("TestSession: {0} | CImage uploaded with size {1}".format(testsession_id, file_uploaded_size))
            try:
                csnapshot.save()
            except Exception as e:
                logger.exception("TestSession: {0} | Exception in csnapshot.save(): {1}".format(testsession_id, e))

    # trash the tmp video
    try:
        if os.path.exists(video_filename):
            os.remove(video_filename)
    except Exception as e:
        logger.exception(
            "TestSession: {0} | Exception while deleting fscan video from tmp: {1}".format(testsession_id, e)
        )

    if is_redis_implementation:
        if not redis_snapshots_list:
            logger.error(
                "TestSession: {0} | No face scan images even after executing the task!".format(testsession_id)
            )
        else:
            redis_video_obj.store_video_snapshots("fscan", redis_snapshots_list)
    else:
        for video in cvideo_fscan_list:
            if CSnapshotImageByVideoId.objects.filter(video_id=video.id).count() <= 0:
                logger.error(
                    "TestSession: {0} | No face scan images even after executing the task!".format(testsession_id)
                )
                # test_session.mark_as_needs_attention()

    for image_file in image_files:
        try:
            if os.path.exists(image_file):
                os.remove(image_file)
        except Exception as e:
            logger.exception(
                "TestSession: {0} | Exception while deleting fscan images from tmp: {1}".format(testsession_id, e)
            )


def _extract_images(video_filename, resolution, snapshot_filename, testsession_id):
    """Method to extract images from video using FFMPEG"""
    try:
        image_extract_cmd = "ffmpeg -loglevel panic -i {0} -r 1 -vf {1} {2}".format(
            video_filename, resolution, snapshot_filename
        )
        resp = call(shlex.split(image_extract_cmd), shell=False)

    except Exception as e:
        logger.error("TestSession: {0} | Image extraction failed! {1}".format(testsession_id, e))
        try:
            command = ["ffmpeg", "-i", video_filename, "-r", "1", "-vf", resolution, snapshot_filename]
            call(command)
        except Exception as e:
            logger.error("TestSession: {0} | Image extraction re-try failed! {1}".format(testsession_id, e))
            