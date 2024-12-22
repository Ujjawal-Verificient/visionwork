//import { fork } from 'child_process';
//import { ipcMain,BrowserWindow } from 'electron';
const {fork} = require('child_process');
const {ipcMain,BrowserWindow, dialog } =require('electron');
const fs = require('fs');
let mainWindow;
const addonProcess = fork('addon_process_child.js');



// ipcMain.on('decrypt', (event, {n, a, b,c,d}) => {
//     console.log('ipcMain get msg from ipcRender process  ',n,a,b,c,d);
//     addonProcess.send({ n,a, b,c,d });
// });

// ipcMain.on('init', (event, {n, a, b }) => {
//     console.log('ipcMain get msg from ipcRender process  ',n,a,b);
//     addonProcess.send({ n,a, b });
// });
// ipcMain.on('process_voice_sample', (event, {n, a }) => {
//     console.log('ipcMain get msg from ipcRender process  ',n,a);
//     addonProcess.send({ n,a});
// });
// ipcMain.on('detect_object', (event, {n, a }) => {
//     console.log('ipcMain get msg from ipcRender process  ',n,a);
//     addonProcess.send({ n,a});
// });
// ipcMain.on('calculate', (event, { a, b }) => {
//     console.log('ipcMain get msg from ipcRender process  ',a,b);
//     addonProcess.send({ a, b });
// });


addonProcess.on('message', (result) => {
    //console.log('addonProcess get result as message:->',result);
    let windowId=1;
    mainWindow = BrowserWindow.fromId(windowId);
    //console.log('addonProcess message',result,mainWindow);
    if (mainWindow) {
        mainWindow.webContents.send('analysis-result', result);
    }
});

// ipcMain.on('setMainWindow', (event, windowId) => {
//     console.log('msg from main process',windowId);
//     mainWindow = BrowserWindow.fromId(windowId);
// });


ipcMain.on('analyze-frame', (event, frameBase64) => {
    console.log('msg in main using electronAPI using analyze-frame');
        const n='od';
       addonProcess.send({n,frameBase64});
    // Call the native addon with the base64 string
    //const result = nativeAddon.analyze(frameBase64);
    //console.log('Analysis result:', result);
});

ipcMain.on('decrypt',(event)=>{
    console.log('msg as decrypt');
    let n='decrypt';
    let seven_z_dll_path=".\\_7z.dll";
    let encypted_password="fc93b9c2-31cd-45a2-9f81-f9967ad24653";
    let encrypted_model_path=".\\models.7z";
    let decrypted_model_path=".\\decrypted_model\\";
    addonProcess.send({n,seven_z_dll_path,encypted_password,encrypted_model_path,decrypted_model_path});

});

ipcMain.on('init',(event)=>{
    console.log('msg as init');
    let n='init';
    let Configuration={
        "is_onboarding": false,
        "uuid": 'd2e64738d26341888bac1c13ed0792f6',
        "cv_configurations": {
          "is_face_verification_activated": true,
          "is_lscr_detection_enabled":false,
          "is_camera_block_detection_enabled": false,
          "is_face_detection_check_for_verification_enabled": false,
          "is_face_pose_required": true,
          "is_background_motion_violation_blur_enabled": false,
          "is_multiple_people_violation_blur_enabled": false,
          "is_blur_all_enabled": false,
          "is_imposter_blur_req":false,
          "check_front_face_for_imposter": true,
          "is_imposter_blur_enabled": false,
          "impostor_detection_frame_infer_time": 5,
          "background_motion_confirmation_count": 2,
          "camera_block_confirmation_count": 5,
          "face_not_detected_confirmation_count": 5,
          "pixel_count_camera_block_detection": 150,
          "imposter_face_box_max_area": 50000,
          "imposter_face_box_min_area": 2500,
          "imposter_hand_face_overlap_area": 500,
          "fsla_wait_period_in_sec": 3,
          "face_verification_threshold": '0.50',
          "class_score_threshold": '0.50',
          "nms_threshold": '0.40',
          "face_detection_threshold": '0.60',
          "real_time_imposter_detection_threshold": '0.50',
          "left_fsla_detection_threshold": '0.75',
          "right_fsla_detection_threshold": '0.68',
          "frontal_fsla_detection_threshold": '0.80',
          "baseline_increasing_min_threshold": '0.40',
          "baseline_increasing_max_threshold": '0.40',
          "imposter_front_pose_threshold": '0.40',
          "is_mobile_camera_voliation_enabled":true,
          "hand_detection_threshold":'0.3',
          "confirmation_count_for_suspicious_mobile_activity":1,
          "face_pose_confidence": '0.85',
          "frame_infer_time": 2,
          "frame_infer_time_slow_system": 4,
          "fsla_confirmation_count": 4,
          "fv_euclidean_dist_threshold": '1.1',
          "fv_faceDetectionThreshold":'0.5',
          "general_classifier_threshold": '0.4',
          "hand_detection_threshold": '0.4',
          "hand_mobile_overlap_area": 100,
          "imposter_detection_confirmation_count": 3,
          "imposter_detection_frame_infer_time": 7,
          "incident_duration": 3,
          "is_fsla_detection_enabled": true,
          "is_imposter_detection_enabled": true,
          "is_left_screen_detection_enabled": true,
          "is_ls_detection_enabled": true,
          "is_mobile_detection_enabled": false,
          "is_mp_detection_enabled": true,
          "is_real_time_incident": true,
          "is_suspicious_background_detection_enabled": false,
          "is_suspicious_mobile_detection_enabled": true,
          "left_screen_confirmation_count": 7,
          "left_session_confirmation_count": 7,
          "long_imposter_detection_confirmation_count": 10,
          "long_left_screen_confirmation_count": 15,
          "long_left_session_confirmation_count": 15,
          "long_multiple_people_confirmation_count": 15,
          "mobile_detection_confirmation_count": 1,
          "mobile_detection_threshold": '0.48',
          "multiple_people_confirmation_count": 5,
          "person_detection_threshold": '0.65',
          "suspicious_background_confirmation_count": 5,
          "suspicious_mobile_detection_threshold": '0.48',
          "video_height_for_cv": 480,
          "video_width_for_cv": 640,
          "nose_deviation_threshold":'30.0'
        },
        "is_face_scan_required": true,
        "is_cv_required": true,
        "video_frame_width": 320,
        "video_frame_height": 240,
        
      };
      let path_config={
       //yolo_v8_nano_model_path:'.\\models\\od_v8_nano_feb24_2.onnx',
        yolo_v8_nano_model_path:'.\\models\\best_1.onnx',
        pose_network_path:".\\models\\hp_v8_nano_feb_2024.pb",
        log_dir_path:'.\\Log',
        retina_para:'.\\models\\retina\\mnet.25-opt.param',
        retina_mode:'.\\models\\retina\\mnet.25-opt.bin',
        arc_para:'.\\models\\mobilefacenet\\mobilefacenet.param',
        arc_mode:'.\\models\\mobilefacenet\\mobilefacenet.bin',
        
      };
      addonProcess.send({n,Configuration,path_config});
});

ipcMain.on('open-folder-dialog', async (event) => {
    const result = await dialog.showOpenDialog({
        properties: ['openDirectory']
    });

    if (!result.canceled) {
        //event.sender.send('selected-folder', result.filePaths[0]);
        let n='setBaselineImg';
        let dirPath=result.filePaths[0]+'\\';   //+'\\';
        console.log(dirPath);
        addonProcess.send({n,dirPath});
    } else {
       // event.sender.send('selected-folder', null);
       let n='setBaselineImg';
       let dirPath='';
       addonProcess.send({n,dirPath});
    }
});

ipcMain.handle('select-folder', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
    });

    return result.canceled ? null : result.filePaths[0];
});
ipcMain.handle('save-page-as-pdf', async (_, filePath) => {
    try {
        const pdfData = await mainWindow.webContents.printToPDF({});
        fs.writeFileSync(filePath, pdfData);
        return { success: true };
    } catch (error) {
        console.error('Failed to save PDF:', error);
        return { success: false, error };
    }
});