import sys, os, datetime
#sys.path.append("/home/sheetal/Downloads/Benchmarking_Essentials_copy")
from cv_utils import cv
import glob, cv2, json
import numpy as np
from utils import read_json
from ultralytics import YOLO

from ultralytics import __version__

print("Ultralytics version:", __version__)
# Models = read_json(r"E:/Scripts/Benchmarking_Essentials_copy/models.json")
Models = read_json(r"/home/ajeet/Downloads/Benchmarking_essentials_updated/Benchmarking_Essentials_copy/models.json")
js_coords_dir = r'E:\Output'
cp_coords_dir = r'E:\Output\yolov8_cpp_outputs'
cv = cv()


# src_list=[
#     "/media/shabuddin/Elements1/datasets_benchmarks/m14/test/mobile_output/*.txt"
#      ]


# if false positives iou is greater than NMS then only the are false positive
# if objects detected but have low confidances than they are missed not false positives
# thres value lower than iou value is not false positive






# src_folder = r"E:/Data/client_data/dummy_fti/"
# src_folder = r'E:/Data/client_data/2021/test/test'
# src_folder = r'E:/Data/client_data/2023_test_data/Labelled_RD_New_1/'
src_folder = r"/home/ajeet/codework/datasets/script_test/"
src_list = cv.skip_hidden_folders(src_folder, extension="*.txt")
print(os.path.exists(src_folder), len(src_list))
# Collect all the test data folders to be benchmarked default=.txt, set .xml for XML
# output_path_list = r"E:\Output\yolov8_benchmark\nano_26_dec"
output_path_list = r"/home/ajeet/benchmark_script"
if not os.path.exists(output_path_list):
    os.mkdir(output_path_list)
VERBOSE = 1
# 0.4 iou nms 0.5 mobile 0.3 conf bnms - skip


def calculate_percentiles(confidences, data):
    # for calculating distribution of confidence levels in the predicted data
    try:
        data += "\n"
        confidences = sorted(confidences)
        length_ = len(confidences)
        if length_ >= 1:
            for i in range(10, 100, 10):
                idx = int(i * length_ / 100)
                data += (
                    "at percentile: " + str(i) + " -> " + str(confidences[idx]) + " \n"
                )
            data += (
                "at percentile: "
                + str(100)
                + " -> "
                + str(confidences[length_ - 1])
                + " \n"
            )
            data += "\n"
    except:
        print(i, length_, confidences)


for FCOUNT, src in enumerate(src_list):
    print(src)
    current_dir = src
    print(src)
    get_month = src.split("/")[-2]
    output = os.path.join(
        output_path_list, str(datetime.date.today()) + "_" + src_folder.split("/")[-1]
    )
    if not os.path.isdir(output):
        os.mkdir(output)
    if not os.path.isdir(os.path.join(output, str(get_month))):
        os.mkdir(os.path.join(output, str(get_month)))
    output_path = os.path.join(output, str(get_month))

    ITERATE_MODELS = ["9"]  # 0 ->m11, 1 ->m12
    # ITERATE_CONFIDANCES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    ITERATE_CONFIDANCES = [0.5]  #
    ITERATE_NMS = [1]  # NMS values should go here if multiple should be seperated by comma's
    ITERATE_IOU = [0.4]  #
    ACTIVATE_LIMIT = False
    IF_LIMIT_THEN_COUNT = 10000
    VISUALIZE = True
    info = {}
    info["XML_SOURCE"] = glob.glob(src)
    info["HOW"] = "CLASSWISE"
    validation_users = list(Models["validation_users"].values())
    print("Data read:{}".format(src))

    for MCOUNT, i in enumerate(ITERATE_MODELS):

        # --------------------Folder Naming Section Start--------------------
        Models["SELECTED"] = i
        Predicted, Groundtruths = [], []
        info["MODEL"] = Models[Models["SELECTED"]]["TYPE"]
        info["MODEL_ID"] = Models[Models["SELECTED"]]["ID"]

        model_name = info["MODEL"] + "_" + info["MODEL_ID"] + "_1"
        if not os.path.isdir(os.path.join(output_path, str(model_name))):
            os.mkdir(os.path.join(output_path, str(model_name)))
        else:
            counter = 2
            while True:
                model_name = info["MODEL"] + "_" + info["MODEL_ID"] + "_" + str(counter)
                if not os.path.isdir(os.path.join(output_path, str(model_name))):
                    os.mkdir(os.path.join(output_path, str(model_name)))
                    break
                counter += 1

        # --------------------Folder Naming Section Ends----------------------

        Config = {
            "CLASSES": Models[Models["SELECTED"]]["CLASSES"],
            "NMS": ITERATE_NMS,
            "CONF": ITERATE_CONFIDANCES,
            "IOU": ITERATE_IOU,
            "LIMIT": ACTIVATE_LIMIT,
            "SAMPLE_LIMIT": IF_LIMIT_THEN_COUNT,
            "TOTAL": len(info["XML_SOURCE"]),
            "RESULTS": os.path.join(output_path, str(model_name)),
            "RESULTS_TEXT": str(datetime.date.today())
            + "_"
            + info["MODEL"]
            + "_"
            + info["HOW"]
            + "_"
            + info["MODEL_ID"]
            + ".txt",
            "FALSE_POSITIVES_FOLDER": "False_Positives",
            "GROUND_TRUTHS_FOLDER": "Ground_Truths",
            "TRUE_POSITIVES_FOLDER": "True_Positives",
            "JOIN_FP_GT_FOLDER": "GT_FP",
            "JOIN_TP_GT_FOLDER": "GT_TP",
            "DATA": "",
            "Visualize": VISUALIZE,
            "NO_CLASSES_DETECTED": "Missed",
            "EMPTY_DELETE": True,
        }

        # ---------------------This set the limit on number of samples we want to benchmark-----------------
        if Config["LIMIT"]:
            limit = Config["SAMPLE_LIMIT"]
        else:
            limit = Config["TOTAL"]
        # ---------------------End of the logic------------------------------------------------------------

        # ---------------------This code section creates the necessary urls for data storage---------------
        Save_folders = {
            "SAVE_RESULTS_TXT_FILE": os.path.join(
                Config["RESULTS"], Config["RESULTS_TEXT"]
            ),
            "FALSE_POSITIVES_FOLDER": os.path.join(
                Config["RESULTS"], Config["FALSE_POSITIVES_FOLDER"]
            ),
            "MISSED_OBJECTS": os.path.join(
                Config["RESULTS"], Config["NO_CLASSES_DETECTED"]
            ),
            "GROUND_TRUTH_FOLDER": os.path.join(
                Config["RESULTS"], Config["GROUND_TRUTHS_FOLDER"]
            ),
            "TRUE_POSITIVES_FOLDER": os.path.join(
                Config["RESULTS"], Config["TRUE_POSITIVES_FOLDER"]
            ),
            "JOIN_FP_GT_FOLDER": os.path.join(
                Config["RESULTS"], Config["JOIN_FP_GT_FOLDER"]
            ),
            "JOIN_TP_GT_FOLDER": os.path.join(
                Config["RESULTS"], Config["JOIN_TP_GT_FOLDER"]
            ),
            "GROUND_TRUTH_FOLDER_DICT": {},
            "FALSE_POSITIVES_FOLDER_DICT": {},
            "TRUE_POSITIVES_FOLDER_DICT": {},
            "JOIN_FP_GT_FOLDER_DICT": {},
            "JOIN_TP_GT_FOLDER_DICT": {},
            "MISSED_OBJECTS_DICT": {},
        }
        # -----------------------End of the logic------------------------------------------------------------

        # ----------------------- This code selects the model paths and id, type, etc variables from models.json
        Selected_model = {
            0: Models[Models["SELECTED"]]["0"],
            # 1: Models[Models["SELECTED"]]["1"],
            "ID": Models[Models["SELECTED"]]["ID"],
            "TYPE": Models[Models["SELECTED"]]["TYPE"],
            "NAME": Models[Models["SELECTED"]]["NAME"],
            "CLASSES": Models[Models["SELECTED"]]["CLASSES"],
        }
        # ----------------------End of the logic--------------------------------------------------------------

        # -----------------------Based on the Selected_model["TYPE"] variable value model will be selected-----
        print(Selected_model["TYPE"], Models["SELECTED"])
        if Selected_model["TYPE"] == "YOLO":
            net = cv2.dnn.readNet(Selected_model[0], Selected_model[1])
            layer_names = net.getLayerNames()
            output_layers = [
                layer_names[i - 1] for i in net.getUnconnectedOutLayers()
            ]
        elif Selected_model["TYPE"] == "SSD":
            net = cv2.dnn.readNetFromTensorflow(Selected_model[0], Selected_model[1])
        elif Selected_model["TYPE"] == "VGG":
            net = cv2.dnn.readNetFromCaffe(Selected_model[0], Selected_model[1])
        elif Selected_model["TYPE"] == 'YOLOV8':
            net = YOLO(Selected_model[0])

        # -----------------------End of the logic-------------------------------------------------------------




        # This code collects all files stored in info["XML_SOURCE"] and process them and store them in the list
        # called Predicted model processor are changed according to Selected_model["TYPE"] variable

        # -------------------------------- Model Process Logic started ----------------------------------
        print(len(info['XML_SOURCE']))
        for index, src_files in enumerate(info["XML_SOURCE"]):
            #print(src_files)
            #print(index)
            if 0 <= index < limit:
                if ".xml" in src_files:
                    gt = cv._xml_data(src_files)
                elif ".txt" in src_files:
                    gt = cv._from_txt(
                        src_files, ["person", "face", "cellphone", "hand"]
                    )
                #print('done')
                src = gt["src"]
                Groundtruths.append(gt)
                if Selected_model["TYPE"] == "YOLO":
                    Predicted.append(
                        cv.models.yolo_prediction(
                            net, src, output_layers, Selected_model["CLASSES"]
                        )
                    )
                elif Selected_model["TYPE"] == "SSD":
                    Predicted.append(
                        cv.models.ssd_detector(net, src, Selected_model["CLASSES"])
                    )
                elif Selected_model["TYPE"] == "VGG":
                    Predicted.append(
                        cv.models.vgg_detector(net, src, Selected_model["CLASSES"])
                    )
                elif Selected_model["TYPE"] == 'YOLOV8':
                    Predicted.append(
                        cv.models.yolov8_prediction(net, src, Selected_model["CLASSES"])
                    )
                elif Selected_model["TYPE"] == "YOLO_JS":
                    current_fname, _ = os.path.splitext(os.path.basename(src_files))
                    json_name = current_fname+'.json'
                    #print(current_dir.split('\\*'), )
                    json_root_dir = os.path.join(js_coords_dir, os.path.basename(current_dir.split('\\*')[0]))
                    json_path = os.path.join(json_root_dir, json_name)
                    h, w, _ = cv2.imread(src).shape
                    #print(os.path.exists(src), os.path.exists(json_path), src_files, json_name, src, json_path)
                    jsoutput = {}
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                        for dp in json_data:
                            cord = [float(dp['score']), max(0, dp['x1'])*w, max(0, dp['y1'])*h, max(0, dp['x2'])*w, max(0, dp['y2'])*h]
                            if dp['class'] not in jsoutput.keys():
                                jsoutput[dp['class']] = []
                            jsoutput[dp['class']].append(cord)
                    
                    #print(os.path.exists(src), os.path.exists(json_path))
                    Predicted.append(
                        {'bbox': jsoutput}
                    )
                elif Selected_model["TYPE"] == "YOLOV8_CPP":
                    current_fname, _ = os.path.splitext(os.path.basename(src_files))
                    json_name = current_fname + '.txt'
                    # print(current_dir.split('\\*'), )
                    json_root_dir = os.path.join(cp_coords_dir, os.path.basename(current_dir.split('\\*')[0]))
                    json_path = os.path.join(json_root_dir, json_name)
                    h, w, _ = cv2.imread(src).shape
                    # print(os.path.exists(src), os.path.exists(json_path), src_files, json_name, src, json_path)
                    jsoutput = {}
                    labels = []
                    with open(json_path, 'r') as f:
                        for line in f.readlines():
                            line = line.replace('\n', '')
                            line = [float(value) for value in line.split()]
                            labels.append(line)
                        for label in labels:
                            if len(label) == 0:
                                continue
                            cord = [float(label[5]), max(0, label[1]), max(0, label[2]),
                                    max(0, label[3]), max(0, label[4])]
                            current_cls = Selected_model["CLASSES"][int(label[0])]
                            if current_cls not in jsoutput.keys():
                                jsoutput[current_cls] = []
                            jsoutput[current_cls].append(cord)
                    # print(os.path.exists(src), os.path.exists(json_path))
                    Predicted.append(
                        {'bbox': jsoutput}
                    )
                    #print({'bbox': jsoutput}, end='\n\n')
                    #print('\n\ngt\n\n')
                    #print(gt, end='\n\n')
                    #print('-'*20)

        # ----------------- End of the logic -----------------------------------------------------------



                if index % 200 == 0:
                    print(index, limit)
                elif (index + 1) % Config["TOTAL"] == 0:
                    print(index, limit)
        print("MAX Read:{}, Total Processed:{}".format(limit, len(Predicted)))



        # ---------------- all important folders will be created here ----------------------------------
        if Config["Visualize"]:
            if not os.path.isdir(Save_folders["GROUND_TRUTH_FOLDER"]):
                os.mkdir(Save_folders["GROUND_TRUTH_FOLDER"])

            if not os.path.isdir(Save_folders["FALSE_POSITIVES_FOLDER"]):
                os.mkdir(Save_folders["FALSE_POSITIVES_FOLDER"])

            if not os.path.isdir(Save_folders["MISSED_OBJECTS"]):
                os.mkdir(Save_folders["MISSED_OBJECTS"])

            if not os.path.isdir(Save_folders["TRUE_POSITIVES_FOLDER"]):
                os.mkdir(Save_folders["TRUE_POSITIVES_FOLDER"])

            if not os.path.isdir(Save_folders["JOIN_FP_GT_FOLDER"]):
                os.mkdir(Save_folders["JOIN_FP_GT_FOLDER"])

            if not os.path.isdir(Save_folders["JOIN_TP_GT_FOLDER"]):
                os.mkdir(Save_folders["JOIN_TP_GT_FOLDER"])

            for clas in Selected_model["CLASSES"]:
                Save_folders["GROUND_TRUTH_FOLDER_DICT"].update(
                    {clas: os.path.join(Save_folders["GROUND_TRUTH_FOLDER"], clas)}
                )
                Save_folders["MISSED_OBJECTS_DICT"].update(
                    {clas: os.path.join(Save_folders["MISSED_OBJECTS"], clas)}
                )
                Save_folders["FALSE_POSITIVES_FOLDER_DICT"].update(
                    {clas: os.path.join(Save_folders["FALSE_POSITIVES_FOLDER"], clas)}
                )
                Save_folders["TRUE_POSITIVES_FOLDER_DICT"].update(
                    {clas: os.path.join(Save_folders["TRUE_POSITIVES_FOLDER"], clas)}
                )
                Save_folders["JOIN_FP_GT_FOLDER_DICT"].update(
                    {clas: os.path.join(Save_folders["JOIN_FP_GT_FOLDER"], clas)}
                )
                Save_folders["JOIN_TP_GT_FOLDER_DICT"].update(
                    {clas: os.path.join(Save_folders["JOIN_TP_GT_FOLDER"], clas)}
                )
        # ----------------- End of the logic -----------------------------------------------------------



        for iou_thres in Config["IOU"]:
            for nms in Config["NMS"]:
                low_confidance_alert = 0
                for conf in Config["CONF"]:
                    tmap = []

                    for count, cls in enumerate(Config["CLASSES"]):
                        Calc = {
                            "GTruth": 0,
                            "table": [],
                            "tp": 0,
                            "fp": 0,
                            "tp_conf": [],
                        }
                        # Config["DATA"] += "Confidance-" + str(conf) + ", IOU-" + str(
                        #     iou_thres) + ", Samples used - " + str(limit) + "\n"
                        if Config["Visualize"]:
                            if not os.path.isdir(
                                Save_folders["GROUND_TRUTH_FOLDER_DICT"][cls]
                            ):
                                os.mkdir(Save_folders["GROUND_TRUTH_FOLDER_DICT"][cls])

                            if not os.path.isdir(
                                Save_folders["MISSED_OBJECTS_DICT"][cls]
                            ):
                                os.mkdir(Save_folders["MISSED_OBJECTS_DICT"][cls])

                            if not os.path.isdir(
                                Save_folders["FALSE_POSITIVES_FOLDER_DICT"][cls]
                            ):
                                os.mkdir(
                                    Save_folders["FALSE_POSITIVES_FOLDER_DICT"][cls]
                                )

                            if not os.path.isdir(
                                Save_folders["TRUE_POSITIVES_FOLDER_DICT"][cls]
                            ):
                                os.mkdir(
                                    Save_folders["TRUE_POSITIVES_FOLDER_DICT"][cls]
                                )

                        # single image iteration till N images and processing of TP and FP values
                        for index, row in enumerate(range(len(Predicted))):
                            Bboxes = {"bbox": [], "bbox_copy": [], "Confidances": []}
                            image_to_visual = cv2.imread(Groundtruths[row]["src"])
                            if len(Predicted[row]) > 0:
                                img = cv2.imread(Groundtruths[row]["src"])
                                fp_detected = False
                                if cls in Groundtruths[row]["bbox"].keys():
                                    # img = cv2.imread(Groundtruths[row]["src"])
                                    print("-*40")
                                    for BOX_INDEX, box1 in enumerate(Groundtruths[row]["bbox"][cls]):
                                        if Config["Visualize"]:
                                            
                                            fp = cv2.imread(Groundtruths[row]["src"])
                                            #print(type(fp))
                                            cv.visualize.zero_iou_middle(
                                                fp,
                                                Groundtruths[row]["bbox"][cls],
                                                cls,
                                                1,
                                                1,
                                                color=(102, 255, 51),
                                            )
                                            filename = (
                                                Save_folders[
                                                    "GROUND_TRUTH_FOLDER_DICT"
                                                ][cls]
                                                + "/"
                                                + Groundtruths[row]["name"]
                                            )
                                            #print('visualizing', filename)
                                            cv2.imwrite(filename, fp)
                                        Calc["GTruth"] += 1
                                    # TRUE POSITIVE AND FALSE POSITIVE CALCULATIONS
                                    if cls in Predicted[row]["bbox"].keys():
                                        for box2 in Predicted[row]["bbox"][cls]:
                                            Bboxes["Confidances"].append(
                                                float(box2[0])
                                            )
                                            box = cv.box_to_center(box2)
                                            Bboxes["bbox"].append(box)
                                            Bboxes["bbox_copy"].append(box2)

                                    else:
                                        if Config["Visualize"]:
                                            fp = cv2.imread(
                                                Groundtruths[row]["src"]
                                            )
                                            cv.visualize.zero_iou_middle(
                                                fp,
                                                Groundtruths[row]["bbox"][cls],
                                                cls,
                                                1,
                                                1,
                                                color=(255, 255, 0),
                                            )
                                            filename = (
                                                Save_folders["MISSED_OBJECTS_DICT"][
                                                    cls
                                                ]
                                                + "/"
                                                + Groundtruths[row]["name"]
                                            )
                                            cv2.imwrite(filename, fp)
                                            print("markinG as missed for image", {Groundtruths[row]["name"]})

                                if cls in Groundtruths[row]["bbox"].keys():
                                    # print(f"Number of bounding boxes before NMS: {len(Bboxes['bbox'])}")
                                    # for i, box in enumerate(Bboxes["bbox_copy"]):
                                    #     confidence = box[0].item()  # Extract confidence (convert tensor to float if necessary)
                                    #     print(f"Confidence: {confidence}")
                                        
                                    idxs = list(
                                        cv2.dnn.NMSBoxes(
                                            Bboxes["bbox"],
                                            Bboxes["Confidances"],
                                            conf,
                                            nms,
                                        )
                                    )
                                    # print(f"Number of bounding boxes after NMS: {len(idxs)}")

                                    checked = []
                                    if idxs.__len__() > 1:
                                        IMAGE_TP = cv2.imread(Groundtruths[row]["src"])
                                        IMAGE_FP = cv2.imread(Groundtruths[row]["src"])
                                        #print(idxs)
                                        temp_count = 0
                                        for c, id in enumerate(idxs):
                                            count = id
                                            pd_box = Bboxes["bbox_copy"][count]
                                            tp_flag = False
                                            for BOX_INDEX, gt_box in enumerate(
                                                Groundtruths[row]["bbox"][cls]
                                            ):
                                                iou = cv.iou(gt_box[1:], pd_box[1:])
                                                iou = round(iou, 3)
                                                if iou >= iou_thres:
                                                    tp_flag = True

                                            if tp_flag:
                                                Calc["table"].append(
                                                    [Bboxes["Confidances"][count], 1, 0]
                                                )
                                                Calc["tp"] += 1
                                                Calc["tp_conf"].append(
                                                    Bboxes["Confidances"][count]
                                                )
                                                cv.visualize.zero_iou_rectified_middle(
                                                    IMAGE_TP,
                                                    pd_box,
                                                    cls,
                                                    Bboxes["Confidances"][count],
                                                    iou,
                                                    color=(102, 0, 255),
                                                )
                                                if temp_count > 0:
                                                    print("More than one time object detected for", cls, Groundtruths[row]["src"])
                                                temp_count = temp_count + 1

                                            else:

                                                Calc["table"].append(
                                                    [Bboxes["Confidances"][count], 0, 1]
                                                )
                                                Calc["fp"] += 1
                                                if Config["Visualize"]:
                                                    cv.visualize.zero_iou_rectified_middle(
                                                        IMAGE_FP,
                                                        pd_box,
                                                        cls,
                                                        Bboxes["Confidances"][count],
                                                        iou,
                                                        color=(255, 0, 0),
                                                    )

                                                    filename_FP = (
                                                        Save_folders[
                                                            "FALSE_POSITIVES_FOLDER_DICT"
                                                        ][cls]
                                                        + "/"
                                                        + Groundtruths[row]["name"]
                                                    )
                                                    cv2.imwrite(filename_FP, IMAGE_FP)
                                                    fp_detected = True
                                                    print("markinG as fp for image", {Groundtruths[row]["name"]})
                                        if tp_flag:
                                            filename_TP = (
                                                Save_folders[
                                                    "TRUE_POSITIVES_FOLDER_DICT"
                                                ][cls]
                                                + "/"
                                                + Groundtruths[row]["name"]
                                            )
                                            cv2.imwrite(filename_TP, IMAGE_TP)
                                            print("markinG as tp for image", {Groundtruths[row]["name"]})

                                    else:
                                        for BOX_INDEX, box1 in enumerate(
                                            Groundtruths[row]["bbox"][cls]
                                        ):
                                            tp_found = False
                                            #print(idxs)
                                            for id in idxs:
                                                count = id
                                                box2 = Bboxes["bbox_copy"][count]
                                                iou = cv.iou(box1[1:], box2[1:])
                                                iou = round(iou, 3)
                                                if iou >= iou_thres:
                                                    tp_found = True

                                            if not tp_found:
                                                fp = cv2.imread(
                                                    Groundtruths[row]["src"]
                                                )
                                                cv.visualize.zero_iou_rectified_middle(
                                                    fp,
                                                    box1,
                                                    cls,
                                                    1,
                                                    1,
                                                    color=(255, 255, 0),
                                                )
                                                filename = (
                                                    Save_folders["MISSED_OBJECTS_DICT"][
                                                        cls
                                                    ]
                                                    + "/"
                                                    + str(BOX_INDEX)
                                                    + "_"
                                                    + Groundtruths[row]["name"]
                                                )
                                                cv2.imwrite(filename, fp)
                                                print("markinG as missed for image", {Groundtruths[row]["name"]})
                                            else:
                                                break

                                        if tp_found:
                                            for index, id in enumerate(idxs):
                                                count = id
                                                Calc["table"].append(
                                                    [Bboxes["Confidances"][count], 1, 0]
                                                )
                                                Calc["tp"] += 1
                                                Calc["tp_conf"].append(
                                                    Bboxes["Confidances"][count]
                                                )

                                                IMAGE_TP = cv2.imread(
                                                    Groundtruths[row]["src"]
                                                )
                                                cv.visualize.zero_iou_rectified_middle(
                                                    IMAGE_TP,
                                                    box2,
                                                    cls,
                                                    Bboxes["Confidances"][count],
                                                    iou,
                                                    color=(102, 0, 255),
                                                )
                                                filename_TP = (
                                                    Save_folders[
                                                        "TRUE_POSITIVES_FOLDER_DICT"
                                                    ][cls]
                                                    + "/"
                                                    + str(index)
                                                    + "_"
                                                    + Groundtruths[row]["name"]
                                                )
                                                cv2.imwrite(filename_TP, IMAGE_TP)
                                                print("markinG as tp for image", {Groundtruths[row]["name"]})
                                        else:
                                            for index, id in enumerate(idxs):
                                                count = id
                                                Calc["table"].append(
                                                    [Bboxes["Confidances"][count], 0, 1]
                                                )
                                                Calc["fp"] += 1
                                                IMAGE_FP = cv2.imread(
                                                    Groundtruths[row]["src"]
                                                )
                                                cv.visualize.zero_iou_rectified_middle(
                                                    IMAGE_FP,
                                                    box2,
                                                    cls,
                                                    Bboxes["Confidances"][count],
                                                    iou,
                                                    color=(255, 0, 0),
                                                )

                                                filename_FP = (
                                                    Save_folders[
                                                        "FALSE_POSITIVES_FOLDER_DICT"
                                                    ][cls]
                                                    + "/"
                                                    + str(index)
                                                    + "_"
                                                    + Groundtruths[row]["name"]
                                                )
                                                cv2.imwrite(filename_FP, IMAGE_FP)
                                                fp_detected = True
                                                print("markinG as fp for image", {Groundtruths[row]["name"]})

                                if cls in Predicted[row]["bbox"].keys():
                                    unmatched_predictions = []
                                    for box2 in Predicted[row]["bbox"][cls]:
                                        match_found = False
                                        if cls in Groundtruths[row]["bbox"].keys():
                                            for box1 in Groundtruths[row]["bbox"][cls]:
                                                iou = cv.iou(box1[1:], box2[1:])
                                                if iou >= iou_thres:
                                                    match_found = True
                                                    break

                                        if not match_found:
                                            unmatched_predictions.append(box2)

                                    # Process False Positives
                                    for box2 in unmatched_predictions:
                                        if not fp_detected:
                                            Calc["table"].append([box2[0].item(), 0, 1])
                                            Calc["fp"] += 1
                                            if Config["Visualize"]:
                                                IMAGE_FP = cv2.imread(Groundtruths[row]["src"])
                                                cv.visualize.zero_iou_rectified_middle(
                                                    IMAGE_FP,
                                                    box2,
                                                    cls,
                                                    box2[0].item(),
                                                    iou, 
                                                    color=(255, 0, 0),
                                                )
                                                filename_FP = (
                                                    Save_folders["FALSE_POSITIVES_FOLDER_DICT"][cls]
                                                    + "/"
                                                    + Groundtruths[row]["name"]
                                                )
                                                cv2.imwrite(filename_FP, IMAGE_FP)
                                                print("Marking as FP for image:", Groundtruths[row]["name"])

                                        fp_detected = False

                            else:
                                if cls not in Predicted[row]["bbox"].keys() and cls in Groundtruths[row]["bbox"].keys():
                                    for box1 in Groundtruths[row]["bbox"][cls]:
                                        if Config["Visualize"]:
                                            img = cv2.imread(Groundtruths[row]["src"])
                                            cv.visualize.zero_iou_rectified_middle(
                                                img, box1, cls, 1, 1, color=(255, 255, 0)
                                            )
                                            filename = (
                                                        Save_folders["MISSED_OBJECTS_DICT"][
                                                            cls
                                                        ]
                                                        + "/"
                                                        + str(BOX_INDEX)
                                                        + "_"
                                                        + Groundtruths[row]["name"]
                                                    )
                                            
                                            cv2.imwrite(filename, img)
                                            print(f"Marking as missed (FN) for image {Groundtruths[row]['name']}")




                        # Collective processing of all dat
                        table = np.array(Calc["table"])
                        try:
                            Config["DATA"] += "\n"
                            confidences = sorted(Calc["tp_conf"])
                            length_ = len(confidences)
                            if length_ >= 1:
                                for i in range(10, 100, 10):
                                    idx = int(i * length_ / 100)
                                    Config["DATA"] += (
                                        "at percentile: "
                                        + str(i)
                                        + " -> "
                                        + str(confidences[idx])
                                        + " \n"
                                    )
                                Config["DATA"] += (
                                    "at percentile: "
                                    + str(100)
                                    + " -> "
                                    + str(confidences[length_ - 1])
                                    + " \n"
                                )
                                Config["DATA"] += "\n"
                        except:
                            print(i, length_, confidences)

                        try:
                            sorted_table = table[table[:, 0].argsort()[::-1]]
                            ap, prec, rec,actual_prec, actual_recall  = cv.map(sorted_table, Calc["GTruth"])
                            Config["DATA"] += (
                                "class - "
                                + str(cls)
                                + " , AP - "
                                + str(int(ap * 100))
                                + "% , TP-"
                                + str(Calc["tp"])
                                + " , FP-"
                                + str(Calc["fp"])
                                + " , GT-"
                                + str(Calc["GTruth"])
                                + " , actual_prec - "
                                + str(int(actual_prec * 100))
                                + "% , actual_recall - "
                                + str(int(actual_recall * 100))
                                + "% , Confidance-"
                                + str(conf)
                                + ", IOU-"
                                + str(iou_thres)
                                + ", Samples used - "
                                + str(limit)
                                + "\n"
                            )
                            tmap.append(ap)

                            print("{}:{}% Avg Prec".format(cls, round(ap * 100, 3)))
                        except:
                            tmap.append(0)

                total_map = 0
                print(tmap)
                for ap in tmap:
                    total_map += round(ap, 2)
                    print("-----------",total_map, "----------------")
                map = round((total_map / len(tmap)) * 100, 0)
                print("mAP:{}%".format(map))
                Config["DATA"] += "mAP - " + str(map) + "%"
                Config["DATA"] += "\n"
                Config[
                    "DATA"
                ] += "------------------------------------------------------"
                Config["DATA"] += "\n"

        Config["DATA"] += "\n\n\n"
        Config["DATA"] += "General Information:"
        Config["DATA"] += "\n"
        Config["DATA"] += "------------------------------------------------------"
        Config["DATA"] += "\n"
        Config["DATA"] += "0: " + Selected_model[0] + "\n"
        # Config["DATA"] += "1: " + Selected_model[1] + "\n"
        Config["DATA"] += "ID: " + Selected_model["ID"] + "\n"
        Config["DATA"] += "TYPE: " + Selected_model["TYPE"] + "\n"
        Config["DATA"] += "NAME: " + Selected_model["NAME"] + "\n"
        Config["DATA"] += "SRC Input path: " + src + "\n"
        Config["DATA"] += "Dest Output path: " + output_path + "\n"
        Config["DATA"] += (
            "NMS:"
            + str(Config["NMS"])
            + ", IOU:"
            + str(Config["IOU"])
            + ", CONF"
            + str(Config["CONF"])
            + " \n"
        )
        Config["DATA"] += (
            "CLASSES INDEX ORDER: " + str(Selected_model["CLASSES"]) + "\n"
        )

        # Config["DATA"] += "Average Processing Time: " + str(int(val)) + " Sec\n"
        # Config["DATA"] += "Max Processing Time: " + str(int(max(Config["PTime"]))) + " Sec\n"
        # Config["DATA"] += "Min Processing Time: " + str(int(min(Config["PTime"]))) + " Sec\n"
        # Config["DATA"] += "Overall Time: " + str(int(add)) + " Sec\n"
        Config["DATA"] += "Sample Count: " + str(limit) + "\n"
        Config["DATA"] += "Date: " + str(datetime.date.today()) + " (Y-M-D)\n"

        file1 = open(Save_folders["SAVE_RESULTS_TXT_FILE"], "w")
        file1.writelines(Config["DATA"])
        file1.close()

for count, cls in enumerate(Config["CLASSES"]):
    if Config["EMPTY_DELETE"]:
        if len(os.listdir(Save_folders["GROUND_TRUTH_FOLDER_DICT"][cls])) == 0:
            os.rmdir(Save_folders["GROUND_TRUTH_FOLDER_DICT"][cls])

        if len(os.listdir(Save_folders["MISSED_OBJECTS_DICT"][cls])) == 0:
            os.rmdir(Save_folders["MISSED_OBJECTS_DICT"][cls])

        if len(os.listdir(Save_folders["FALSE_POSITIVES_FOLDER_DICT"][cls])) == 0:
            os.rmdir(Save_folders["FALSE_POSITIVES_FOLDER_DICT"][cls])

        if len(os.listdir(Save_folders["TRUE_POSITIVES_FOLDER_DICT"][cls])) == 0:
            os.rmdir(Save_folders["TRUE_POSITIVES_FOLDER_DICT"][cls])

    if not os.path.isdir(Save_folders["JOIN_FP_GT_FOLDER_DICT"][cls]):
        os.mkdir(Save_folders["JOIN_FP_GT_FOLDER_DICT"][cls])

    if not os.path.isdir(Save_folders["JOIN_TP_GT_FOLDER_DICT"][cls]):
        os.mkdir(Save_folders["JOIN_TP_GT_FOLDER_DICT"][cls])

        for count, cls in enumerate(Config["CLASSES"]):
            if os.path.isdir(Save_folders["FALSE_POSITIVES_FOLDER_DICT"][cls]):
                fp_folder = os.listdir(Save_folders["FALSE_POSITIVES_FOLDER_DICT"][cls])
                gt_folder = os.listdir(Save_folders["GROUND_TRUTH_FOLDER_DICT"][cls])
                if len(fp_folder) >= 1:
                    for file in fp_folder:
                        try:
                            file_in_fp_folder = cv2.imread(
                                os.path.join(
                                    Save_folders["FALSE_POSITIVES_FOLDER_DICT"][cls], file
                                )
                            )
                            file_in_gt_folder = cv2.imread(
                                os.path.join(
                                    Save_folders["GROUND_TRUTH_FOLDER_DICT"][cls], file
                                )
                            )
                            joined = np.concatenate(
                                (file_in_gt_folder, file_in_fp_folder), axis=1
                            )
                            save_path = os.path.join(
                                Save_folders["JOIN_FP_GT_FOLDER_DICT"][cls], file
                            )

                            cv2.imwrite(save_path, joined)
                        except:
                            pass

            if os.path.isdir(Save_folders["TRUE_POSITIVES_FOLDER_DICT"][cls]):
                fp_folder = os.listdir(Save_folders["TRUE_POSITIVES_FOLDER_DICT"][cls])
                gt_folder = os.listdir(Save_folders["GROUND_TRUTH_FOLDER_DICT"][cls])
                if len(fp_folder) >= 1:
                    for file in fp_folder:
                        try:
                            file_in_fp_folder = cv2.imread(
                                os.path.join(
                                    Save_folders["TRUE_POSITIVES_FOLDER_DICT"][cls], file
                                )
                            )
                            file_in_gt_folder = cv2.imread(
                                os.path.join(
                                    Save_folders["GROUND_TRUTH_FOLDER_DICT"][cls], file
                                )
                            )
                            joined = np.concatenate(
                                (file_in_gt_folder, file_in_fp_folder), axis=1
                            )
                            save_path = os.path.join(
                                Save_folders["JOIN_TP_GT_FOLDER_DICT"][cls], file
                            )
                            cv2.imwrite(save_path, joined)
                        except:
                            pass

print("Process is finished!")