
//const { ipcRenderer } = require('electron');
// const form = document.getElementById("form");
// form.addEventListener("submit", img_Submit);
// Video and analysis column elements
//const fs = require('fs');
//const path = require('path');
var output_file_name;
// const video = document.getElementById('video');
const analysisColumn = document.getElementById('analysisColumn');
// const videoInput = document.getElementById('videoInput');


document.getElementById("decryptButton").addEventListener("click", () => {
  console.log('decrypt button clicked!...')
  window.electronAPI.decryptModel();
  document.getElementById("decryptButton").disabled=true;
});

document.getElementById("initButton").addEventListener("click", () => {
  console.log('initButtonclicked!...')
  window.electronAPI.initAddon();
  document.getElementById("initButton").disabled=true;
});

document.getElementById('selectFolder').addEventListener('click', () => {
  console.log('selectFolder button clicked!...')
  window.electronAPI.selectFolder();
  //ipcRenderer.send('open-folder-dialog');
  document.getElementById('selectFolder').disabled=true;
});
document.getElementById('selectImgFolder').addEventListener('click',async () => {
  console.log('selectImgFolder button clicked!...')
  const folder_path=await window.electronAPI.selectImgFolder();
  console.log(`folder_path: ${folder_path}`);
  output_file_name=folder_path;
  await processImages(folder_path);
  //ipcRenderer.send('open-folder-dialog');
  document.getElementById('selectImgFolder').disabled=true;
  alert('Session Completed successfully!');
});


// Function to convert an image to Base64
function imageToBase64(filePath) {
  const imageBuffer = fs.readFileSync(filePath);
  return `data:image/${path.extname(filePath).slice(1)};base64,${imageBuffer.toString('base64')}`;
}

// Function to process images in a selected folder
async function processImages(folderPath) {
  const files = window.electronAPI.readDirectory(folderPath);
  // const folderPath = window.electronAPI.selectImgFolder();
  // if (!folderPath) {
  //     console.log('No folder selected.');
  //     return;
  // }

  // fs.readdir(folderPath, (err, files) => {
  //     if (err) {
  //         console.error('Error reading folder:', err);
  //         return;
  //     }

      files.forEach((file) => {
        console.log(`File: ${file}`);
        const base64Data = window.electronAPI.fileToBase64(folderPath+'\\'+file);
        if (base64Data) {
         // console.log('Base64 Data:', base64Data);
          window.electronAPI.analyzeFrame(base64Data); 
          // Optionally display the Base64 data in an image tag
         // const imgElement = document.getElementById('base64-image');
          //imgElement.src = `data:image/jpeg;base64,${base64Data}`;
      } else {
          console.error('Failed to convert file to Base64.');
      }
          //const filePath = path.join(folderPath, file);

          // // Filter out non-image files
          // if (/\.(jpg|jpeg|png|gif)$/i.test(file)) {
          //     const base64Image = imageToBase64(filePath);
          //     const base64Data = base64Image.split(',')[1]; // remove the data URL prefix
          //     window.electronAPI.analyzeFrame(base64Data); // Send the Base64 image to the main process
          // }
      });
  
}
// document.getElementById("startImgAnalysis").addEventListener("click", () => {
//   console.log('startImgAnalysisButtonclicked!...')
//   const folderPath = "C:\\Users\\ujjawalkumar_verific\\Pictures\\Sessions\\02dba7a3-926b-4135-bbe8-47d6a767b02b_merged";
//   processImages(folderPath);
//   // window.electronAPI.startImgAnalysis();
//   document.getElementById("startImgAnalysis").disabled=true;
// });
// ipcRenderer.on('selected-folder', (event, folderPath) => {
//   if (folderPath) {
//       document.getElementById('folderPath').textContent = `Selected folder: ${folderPath}`;
//   } else {
//       document.getElementById('folderPath').textContent = 'No folder selected';
//   }
// });

// Handle video file selection
// videoInput.addEventListener('change', () => {
//   const file = videoInput.files[0];
//   if (file) {
//       const fileURL = URL.createObjectURL(file);
//       video.src = fileURL; // Load the selected video file into the video element
//       video.play();
//   }
//   document.getElementById("startCVD").disabled=true;
// });

// function captureFrame() {
//   console.log('captureFrame called ...');
//   //const canvas = document.getElementById('canvas');
//   const canvas = document.createElement('canvas');
//   const ctx = canvas.getContext('2d');
//   canvas.width = video.videoWidth;
//   canvas.height = video.videoHeight;
//   ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
//   // Convert to base64
//   const frameBase64 = canvas.toDataURL('image/jpeg', 0.8); // quality 80%
//   const base64Data = frameBase64.split(',')[1]; // remove the data URL prefix
  
//   // Send frame to native addon for analysis
//   window.electronAPI.analyzeFrame(base64Data);  // Assuming you have a preload script exposing API
// }
// document.getElementById("startCVD").addEventListener("click", () => {
//   console.log('startCVDclicked!...')  
//   // Access the camera feed
//     navigator.mediaDevices.getUserMedia({ video: true })
//     .then(stream => {
//         video.srcObject = stream;
//     })
//     .catch(err => console.error('Error accessing camera:', err));   
//   //window.electronAPI.startCVDAddon();
//   //document.getElementById("startCVD").disabled=true;
//    // Capture frames every 100ms (or adjust interval as needed)
//    setInterval(captureFrame, 2000);
//    document.getElementById("startCVD").disabled=true;
//    videoInput.disabled=true;
// });

// // Capture frames at intervals if video is playing
// video.addEventListener('play', () => {
//   const captureInterval = setInterval(() => {
//       if (video.paused || video.ended) {
//           clearInterval(captureInterval);
//           return;
//       }
//       captureFrame();
//   }, 2000); // Adjust interval as needed
// });

// function decrypt(){
//   console.log('decrypt button clicked!...')
//   const decryptButton = document.getElementById('decryptButton');
//   let seven_z_dll_path=".\\_7z.dll";
//   let encypted_password="fc93b9c2-31cd-45a2-9f81-f9967ad24653";
//   let encrypted_model_path=".\\models.7z";
//   let decrypted_model_path=".\\decrypted_model\\";
//   ipcRenderer.send('decrypt', { n: 'decrypt', a: seven_z_dll_path, b: encypted_password,c:encrypted_model_path,d:decrypted_model_path});
//   // Disable the button
//   decryptButton.disabled = true;
// }
// function init() {
//     //console.log('calling calculate....')
//     let Configuration={
//       "id": 633899,
//       "app_states": [
//         {
//           "state": 1,
//           "renderer": "instructions"
//         },
//         {
//           "state": 2,
//           "renderer": "agreement-policy"
//         },
//         {
//           "state": 3,
//           "renderer": "system-check"
//         },
//         {
//           "state": 4,
//           "renderer": "face-scan"
//         },
//         {
//           "state": 5,
//           "renderer": "id-scan"
//         },
//         {
//           "state": 6,
//           "renderer": "start-proctoring"
//         }
//       ],
//       "is_onboarding": false,
//       "uuid": 'd2e64738d26341888bac1c13ed0792f6',
//       "cv_configurations": {
//         "is_face_verification_activated": true,
//         "is_lscr_detection_enabled":false,
//         "is_camera_block_detection_enabled": false,
//         "is_face_detection_check_for_verification_enabled": false,
//         "is_face_pose_required": true,
//         "is_background_motion_violation_blur_enabled": false,
//         "is_multiple_people_violation_blur_enabled": false,
//         "is_blur_all_enabled": false,
//         "is_imposter_blur_req":false,
//         "check_front_face_for_imposter": true,
//         "is_imposter_blur_enabled": false,
//         "impostor_detection_frame_infer_time": 5,
//         "background_motion_confirmation_count": 2,
//         "camera_block_confirmation_count": 1,
//         "face_not_detected_confirmation_count": 5,
//         "pixel_count_camera_block_detection": 150,
//         "imposter_face_box_max_area": 20000,
//         "imposter_face_box_min_area": 2500,
//         "imposter_hand_face_overlap_area": 500,
//         "fsla_wait_period_in_sec": 3,
//         "face_verification_threshold": '0.43',
//         "class_score_threshold": '0.50',
//         "nms_threshold": '0.40',
//         "face_detection_threshold": '0.65',
//         "real_time_imposter_detection_threshold": '0.46',
//         "left_fsla_detection_threshold": '0.75',
//         "right_fsla_detection_threshold": '1.00',
//         "frontal_fsla_detection_threshold": '0.80',
//         "baseline_increasing_min_threshold": '0.40',
//         "baseline_increasing_max_threshold": '0.40',
//         "imposter_front_pose_threshold": '0.40',
//         "is_mobile_camera_voliation_enabled":true,
//         "hand_detection_threshold":'0.3',
//         "confirmation_count_for_suspicious_mobile_activity":1,
//         "face_pose_confidence": '0.85',
//         "frame_infer_time": '2',
//         "frame_infer_time_slow_system": 4,
//         "fsla_confirmation_count": 3,
//         "fsla_warning_msg": "You appear to be looking off camera for unspecified reasons. This is a violation of proctoring rules. Please pay attention only to your exam. Repeated instances may result in a failed test.",
//         "fsla_warning_title": "Warning!",
//         "fv_euclidean_dist_threshold": 1.1,
//         "fv_faceDetectionThreshold":'0.5',
//         "general_classifier_threshold": '0.4',
//         "hand_detection_threshold": '0.4',
//         "hand_mobile_overlap_area": 100,
//         "imposter_detection_confirmation_count": 3,
//         "imposter_detection_frame_infer_time": 3,
//         "imposter_warning_msg": "Our proctoring system has detected an unauthorized individual on camera. This is a violation of proctoring rules. Repeated instances may result in a failed test.",
//         "imposter_warning_title": "Warning!",
//         "incident_duration": 3,
//         "is_disqualification_enabled": false,
//         "is_fsla_detection_enabled": true,
//         "is_fsla_warning_enabled": true,
//         "is_imposter_detection_enabled": true,
//         "is_imposter_disqualification_enabled": false,
//         "is_imposter_warning_enabled": true,
//         "is_left_screen_detection_enabled": true,
//         "is_left_screen_disqualification_enabled": false,
//         "is_left_screen_warning_enabled": true,
//         "is_ls_detection_enabled": true,
//         "is_ls_disqualification_enabled": false,
//         "is_ls_warning_enabled": false,
//         "is_md_disqualification_enabled": false,
//         "is_md_warning_enabled": true,
//         "is_mobile_detection_enabled": false,
//         "is_mp_detection_enabled": true,
//         "is_mp_disqualification_enabled": false,
//         "is_mp_warning_enabled": true,
//         "is_plugin_cv_violation_on": true,
//         "is_real_time_incident": true,
//         "is_suspicious_background_detection_enabled": false,
//         "is_suspicious_background_warning_enabled": false,
//         "is_suspicious_mobile_detection_enabled": true,
//         "is_warning_and_long_violation_enabled": true,
//         "left_screen_confirmation_count": 7,
//         "left_screen_warning_msg": "Our proctoring system is unable to fully monitor your face. Please remain in full view of the camera for your entire test. Repeated instances may result in a failed test.",
//         "left_screen_warning_title": "Warning!",
//         "left_session_confirmation_count": 7,
//         "long_imposter_detection_confirmation_count": 10,
//         "long_left_screen_confirmation_count": 15,
//         "long_left_session_confirmation_count": 15,
//         "long_multiple_people_confirmation_count": 15,
//         "ls_warning_msg": "Our proctoring system has detected that you may have left the session. Please remain on camera for the entire exam. Repeated instances may result in a failed test.",
//         "ls_warning_title": "Warning!",
//         "max_normalized_face_area_for_pose_classifier": 0.28,
//         "md_count_for_disqualification": 4,
//         "md_warning_msg": "Our proctoring system has detected a cell phone present. Please do not attempt to access any devices during your exam. Repeated instances may result in a failed test.",
//         "md_warning_title": "Warning!",
//         "min_normalized_face_area_for_imposter_detection": 0.04,
//         "min_normalized_face_area_for_pose_classifier": 0.06,
//         "mobile_detection_confirmation_count": 1,
//         "mobile_detection_threshold": '0.8',
//         "mp_duration_for_disqaulification": 10,
//         "mp_warning_msg": "Our proctoring system has detected more than one person present. Please ensure you are alone in front of your computer. Repeated instances may result in a failed test.",
//         "mp_warning_title": "Warning!",
//         "multiple_people_confirmation_count": 5,
//         "person_detection_threshold": '0.5',
//         "sb_warning_msg": "Our proctoring system has detected suspicious background motion. Please ensure that you are in a private area before proceeding.",
//         "sb_warning_title": "Warning!",
//         "suspicious_background_confirmation_count": 2,
//         "suspicious_mobile_detection_threshold": 0.8,
//         "cipher_entry": "hIWYz+YKOVG0TKK/osEiwLee9woVjS4MvgTSkxEwoegQepRN7V3IVA5kOEJbaxc6lCbt7cJgP8X2YRNPFnmh9H5HyOOLoXHyc2v2my8oE2Q=",
//         "video_height_for_cv": 480,
//         "video_width_for_cv": 640
//       },
//       "violation_types": {
//         "FSLA": "Facial Suspicion-looking away",
//         "IMPOSTER": "Imposter",
//         "LEFT_SCREEN": "Left Screen",
//         "LONG_IMPOSTER": "Long-Imposter",
//         "LONG_LEFT_SCREEN": "Long-Left Screen",
//         "LONG_LS": "Long-Left Session",
//         "LONG_MP": "Long-Multiple people",
//         "LS": "Left Session",
//         "MD": "Mobile Detected",
//         "MP": "Multiple people",
//         "SB": "Suspicious Background",
//         "WARNING_FSLA": "Warning-FSLA",
//         "WARNING_IMPOSTER": "Warning-Imposter",
//         "WARNING_LEFT_SCREEN": "Warning-Left Screen",
//         "WARNING_LS": "Warning-Left Session",
//         "WARNING_MD": "Warning-Mobile Detected",
//         "WARNING_MP": "Warning-Multiple people"
//       },
//       "session_details": {
//         "user_id": 474972,
//         "lms_type": "HOSTED",
//         "session_id": "3141493",
//         "session_meta_id": 3297437,
//         "session_uuid": "e4212f5ffd8c4e208d1a3dd030e94506",
//         "test_max_duration": 180,
//         "test_platform_domain": ":///*"
//       },
//       "firebase": {},
//       "detection_violation_after_test_start_callback": true,
//       "is_acceptance_screen_required": true,
//       "blacklist_windows_apps": [
//         "CROSSLOOPCONNECT.EXE",
//         "Dropbox.exe",
//         "atmgr.exe",
//         "eclipse.exe"
//       ],
//       "blacklist_mac_apps": ["LINE"],
//       "whitelist_windows_apps": [
//         {
//           "id": 1,
//           "name": "ACTIVEPRESENTER.EXE",
//           "verbose_name": "ACTIVEPRESENTER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 2,
//           "name": "ADOBECAPTIVATE.EXE",
//           "verbose_name": "ADOBECAPTIVATE",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 3,
//           "name": "BDCAM.EXE",
//           "verbose_name": "BDCAM",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 4,
//           "name": "FLASHBACK RECORDER.EXE",
//           "verbose_name": "FLASHBACK RECORDER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 5,
//           "name": "RECORDER.EXE",
//           "verbose_name": "RECORDER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 6,
//           "name": "CAMTASIASTUDIO.EXE",
//           "verbose_name": "CAMTASIASTUDIO",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 7,
//           "name": "EPICREWIND.EXE",
//           "verbose_name": "EPICREWIND",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 8,
//           "name": "FRAPS.EXE",
//           "verbose_name": "FRAPS",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 9,
//           "name": "GRABILLA.EXE",
//           "verbose_name": "GRABILLA",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 10,
//           "name": "GRABILLATRAY.EXE",
//           "verbose_name": "GRABILLATRAY",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 11,
//           "name": "HYCAM2.EXE",
//           "verbose_name": "HYCAM2",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 12,
//           "name": "JING.EXE",
//           "verbose_name": "JING",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 13,
//           "name": "NEROVISION.EXE",
//           "verbose_name": "NEROVISION",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 14,
//           "name": "SCREENCAM.EXE",
//           "verbose_name": "SCREENCAM",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 15,
//           "name": "SCREENPRESSO.EXE",
//           "verbose_name": "SCREENPRESSO",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 16,
//           "name": "SMARTPIXEL.EXE",
//           "verbose_name": "SMARTPIXEL",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 17,
//           "name": "SNAGIT32.EXE",
//           "verbose_name": "SNAGIT32",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 18,
//           "name": "SNAGITEDITOR.EXE",
//           "verbose_name": "SNAGITEDITOR",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 19,
//           "name": "SNAGPRIV.EXE",
//           "verbose_name": "SNAGPRIV",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 20,
//           "name": "SNIPPINGTOOL.EXE",
//           "verbose_name": "SNIPPINGTOOL",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 21,
//           "name": "VLC.EXE",
//           "verbose_name": "VLC",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 22,
//           "name": "WMENC.EXE",
//           "verbose_name": "WMENC",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 23,
//           "name": "WMENCAGT.EXE",
//           "verbose_name": "WMENCAGT",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 24,
//           "name": "WINK.EXE",
//           "verbose_name": "WINK",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 25,
//           "name": "APPSERVERCLIENT.EXE",
//           "verbose_name": "APPSERVERCLIENT",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 26,
//           "name": "APC_HOST.EXE",
//           "verbose_name": "APC_HOST",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 27,
//           "name": "BACONSOLEAPP.EXE",
//           "verbose_name": "BACONSOLEAPP",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 28,
//           "name": "TLCLIENT.EXE",
//           "verbose_name": "TLCLIENT",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 30,
//           "name": "VNCVIEWER.EXE",
//           "verbose_name": "VNCVIEWER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 31,
//           "name": "BLAZE.EXE",
//           "verbose_name": "BLAZE",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 32,
//           "name": "VDISYSTRAYICON.EXE",
//           "verbose_name": "VDISYSTRAYICON",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 33,
//           "name": "G2TRAY.EXE",
//           "verbose_name": "G2TRAY",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 34,
//           "name": "GOVERRMC.EXE",
//           "verbose_name": "GOVERRMC",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 35,
//           "name": "HAMACHI-2-UI.EXE",
//           "verbose_name": "HAMACHI-2-UI",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 36,
//           "name": "MIKOGO-HOST.EXE",
//           "verbose_name": "MIKOGO-HOST",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 37,
//           "name": "PCIDEPLY.EXE",
//           "verbose_name": "PCIDEPLY",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 38,
//           "name": "PCICTLUI.EXE",
//           "verbose_name": "PCICTLUI",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 39,
//           "name": "CITRIX ONLINE LAUNCHER.EXE",
//           "verbose_name": "CITRIX ONLINE LAUNCHER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 40,
//           "name": "G2MUI.EXE",
//           "verbose_name": "G2MUI",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 41,
//           "name": "G2MSTART.EXE",
//           "verbose_name": "G2MSTART",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 42,
//           "name": "G2MLAUNCHER.EXE",
//           "verbose_name": "G2MLAUNCHER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 43,
//           "name": "G2MCOMM.EXE",
//           "verbose_name": "G2MCOMM",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 44,
//           "name": "G2MCHAT.EXE",
//           "verbose_name": "G2MCHAT",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 45,
//           "name": "TEAMVIEWER.EXE",
//           "verbose_name": "TEAMVIEWER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 46,
//           "name": "I_VIEW32.EXE",
//           "verbose_name": "I_VIEW32",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 81,
//           "name": "SKYPE.EXE",
//           "verbose_name": "SKYPE",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 82,
//           "name": "WINWORD.EXE",
//           "verbose_name": "Word",
//           "os_type": "Windows",
//           "application_file_name": "Word.lnk"
//         },
//         {
//           "id": 83,
//           "name": "manycam-to-be-removed.exe",
//           "verbose_name": "manycam-to-be-removed",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 89,
//           "name": "QuickTimePlayer.exe",
//           "verbose_name": "QuickTimePlayer",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 92,
//           "name": "FLASHBACKRECORDER.EXE",
//           "verbose_name": "FLASHBACKRECORDER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 95,
//           "name": "CITRIXONLINELAUNCHER.EXE",
//           "verbose_name": "CITRIXONLINELAUNCHER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 128,
//           "name": "GoogleHangout",
//           "verbose_name": "GoogleHangout",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 134,
//           "name": "manycam-to-be-removed2.exe",
//           "verbose_name": "manycam-to-be-removed2",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 140,
//           "name": "AntiLogger.exe",
//           "verbose_name": "AntiLogger",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 143,
//           "name": "SpyShelter.exe",
//           "verbose_name": "SpyShelter",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 145,
//           "name": "apowersoft online launcher.exe",
//           "verbose_name": "apowersoft online launcher",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 148,
//           "name": "online screen recorder.exe",
//           "verbose_name": "online screen recorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 151,
//           "name": "NSR.exe",
//           "verbose_name": "NSR",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 154,
//           "name": "Screencast-O-Matic.exe",
//           "verbose_name": "Screencast-O-Matic",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 157,
//           "name": "eLectaRecorder.exe",
//           "verbose_name": "eLectaRecorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 160,
//           "name": "bdcam64.exe",
//           "verbose_name": "bdcam64",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 163,
//           "name": "SMM_HyperCam.exe",
//           "verbose_name": "SMM_HyperCam",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 166,
//           "name": "ShowMore.exe",
//           "verbose_name": "ShowMore",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 169,
//           "name": "Screenmailer.exe",
//           "verbose_name": "Screenmailer",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 172,
//           "name": "scrrec.exe",
//           "verbose_name": "scrrec",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 175,
//           "name": "screen2avi.exe",
//           "verbose_name": "screen2avi",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 178,
//           "name": "Basic Screen Recorder.exe",
//           "verbose_name": "Basic Screen Recorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 181,
//           "name": "screenrecord.exe",
//           "verbose_name": "screenrecord",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 184,
//           "name": "FreeScreenVideoRecorder.exe",
//           "verbose_name": "FreeScreenVideoRecorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 187,
//           "name": "ScreenRecorder.exe",
//           "verbose_name": "ScreenRecorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 190,
//           "name": "ScreenCapturing.exe",
//           "verbose_name": "ScreenCapturing",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 193,
//           "name": "NemCap.exe",
//           "verbose_name": "NemCap",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 196,
//           "name": "takescreen-lite.exe",
//           "verbose_name": "takescreen-lite",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 199,
//           "name": "videocapture.exe",
//           "verbose_name": "videocapture",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 202,
//           "name": "ScreenTwin.exe",
//           "verbose_name": "ScreenTwin",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 205,
//           "name": "SRecorder.exe",
//           "verbose_name": "SRecorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 208,
//           "name": "amcap.exe",
//           "verbose_name": "amcap",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 211,
//           "name": "Action.exe",
//           "verbose_name": "Action",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 214,
//           "name": "Action_logon.exe",
//           "verbose_name": "Action_logon",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 217,
//           "name": "rlhtmlrenderer.exe",
//           "verbose_name": "rlhtmlrenderer",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 220,
//           "name": "Carrecorder.exe",
//           "verbose_name": "Carrecorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 223,
//           "name": "Screen Recorder 6.exe",
//           "verbose_name": "Screen Recorder 6",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 226,
//           "name": "Tiny Take By MangoApps.exe",
//           "verbose_name": "Tiny Take By MangoApps",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 258,
//           "name": "TinyTake By MangoApps.exe",
//           "verbose_name": "TinyTake By MangoApps",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 262,
//           "name": "BitTorrent.exe",
//           "verbose_name": "BitTorrent",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 264,
//           "name": "remote_assistance_host.exe",
//           "verbose_name": "remote_assistance_host",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 266,
//           "name": "remoting_native_messaging_host.exe",
//           "verbose_name": "remoting_native_messaging_host",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 272,
//           "name": "w1r2s3a4.e5x6e7",
//           "verbose_name": "w1r2s3a4.e5x6e7",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 276,
//           "name": "CAMRECORDER.EXE",
//           "verbose_name": "CAMRECORDER",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 278,
//           "name": "SkypeApp.exe",
//           "verbose_name": "SkypeApp",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 280,
//           "name": "GamePanel.exe",
//           "verbose_name": "GamePanel",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 282,
//           "name": "easycapture.exe",
//           "verbose_name": "easycapture",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 284,
//           "name": "Evernote.exe",
//           "verbose_name": "Evernote",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 288,
//           "name": "rserver3.exe",
//           "verbose_name": "rserver3",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 290,
//           "name": "dexpot.exe",
//           "verbose_name": "dexpot",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 292,
//           "name": "rutserv.exe",
//           "verbose_name": "rutserv",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 294,
//           "name": "winvnc.exe",
//           "verbose_name": "winvnc",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 296,
//           "name": "DeskRollUA.exe",
//           "verbose_name": "DeskRollUA",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 298,
//           "name": "Screenleap.exe",
//           "verbose_name": "Screenleap",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 300,
//           "name": "screenleap ",
//           "verbose_name": "screenleap ",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 302,
//           "name": "notepad.exe",
//           "verbose_name": "notepad",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 304,
//           "name": "rocket.chat",
//           "verbose_name": "rocket.chat",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 306,
//           "name": "jitsi meetings",
//           "verbose_name": "jitsi meetings",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 308,
//           "name": "AeroAdmin.exe",
//           "verbose_name": "AeroAdmin",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 310,
//           "name": "AA_v3.exe",
//           "verbose_name": "AA_v3",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 312,
//           "name": "Screen Recorder Launcher.exe",
//           "verbose_name": "Screen Recorder Launcher",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 316,
//           "name": "mingleview.exe",
//           "verbose_name": "mingleview",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 318,
//           "name": "AnyDesk.exe",
//           "verbose_name": "AnyDesk",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 324,
//           "name": "ezvid.exe",
//           "verbose_name": "ezvid",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 326,
//           "name": "freecam.exe",
//           "verbose_name": "freecam",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 328,
//           "name": "ApowerRec.exe",
//           "verbose_name": "ApowerRec",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 330,
//           "name": "ShareX.exe",
//           "verbose_name": "ShareX",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 332,
//           "name": "XSplit.Core.exe",
//           "verbose_name": "XSplit.Core",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 334,
//           "name": "UsabilityStudio.exe",
//           "verbose_name": "UsabilityStudio",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 336,
//           "name": "obs64.exe",
//           "verbose_name": "obs64",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 338,
//           "name": "Filmora.exe",
//           "verbose_name": "Filmora",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 340,
//           "name": "ashsnap.exe",
//           "verbose_name": "ashsnap",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 342,
//           "name": "Reflector3.exe",
//           "verbose_name": "Reflector3",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 344,
//           "name": "x-mirage.exe",
//           "verbose_name": "x-mirage",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 346,
//           "name": "FSRecorder.exe",
//           "verbose_name": "FSRecorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 348,
//           "name": "VideoEditor.exe",
//           "verbose_name": "VideoEditor",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 350,
//           "name": " WSHelper.exe",
//           "verbose_name": " WSHelper",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 352,
//           "name": "obs32.exe",
//           "verbose_name": "obs32",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 354,
//           "name": "Capture.exe",
//           "verbose_name": "Capture",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 356,
//           "name": "Mimic.exe",
//           "verbose_name": "Mimic",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 358,
//           "name": "Recorder.exe",
//           "verbose_name": "Recorder",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 364,
//           "name": "AirServer.exe",
//           "verbose_name": "AirServer",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 368,
//           "name": "RemotePC.exe",
//           "verbose_name": "RemotePC",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 370,
//           "name": "RPCPrintServer.exe",
//           "verbose_name": "RPCPrintServer",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 372,
//           "name": "RPCSuite.exe",
//           "verbose_name": "RPCSuite",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 374,
//           "name": "RemotePC1.exe",
//           "verbose_name": "RemotePC1",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 376,
//           "name": "GoPlay Editor",
//           "verbose_name": "GoPlay Editor",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 378,
//           "name": "Capture2Text.exe",
//           "verbose_name": "Capture2Text",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 380,
//           "name": "picpick.exe",
//           "verbose_name": "picpick",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 382,
//           "name": "Greenshot.exe",
//           "verbose_name": "Greenshot",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 384,
//           "name": "EvernoteClipper.exe",
//           "verbose_name": "EvernoteClipper",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 386,
//           "name": "SnapCrab.exe",
//           "verbose_name": "SnapCrab",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 390,
//           "name": "MWSnap.exe",
//           "verbose_name": "MWSnap",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 392,
//           "name": "onedrive.exe",
//           "verbose_name": "onedrive",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 394,
//           "name": "Windows Screen Capture Tool.exe",
//           "verbose_name": "Windows Screen Capture Tool",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 396,
//           "name": "SLACK.EXE",
//           "verbose_name": "SLACK",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 400,
//           "name": "DbxSvc.exe",
//           "verbose_name": "DbxSvc",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 402,
//           "name": "Monosnap.exe",
//           "verbose_name": "Monosnap",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 404,
//           "name": "FSCapture.exe",
//           "verbose_name": "FSCapture",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 406,
//           "name": "HipChat.exe",
//           "verbose_name": "HipChat",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 408,
//           "name": "ChrisPCScreenRec.exe",
//           "verbose_name": "ChrisPCScreenRec",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 410,
//           "name": "googledrivesync.exe",
//           "verbose_name": "googledrivesync",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 418,
//           "name": "Camtasia.exe",
//           "verbose_name": "Camtasia",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 426,
//           "name": "VirtuaWin.exe",
//           "verbose_name": "VirtuaWin",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 428,
//           "name": "Desktops.exe",
//           "verbose_name": "Desktops",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 430,
//           "name": "agent.exe",
//           "verbose_name": "agent",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 434,
//           "name": "ReadyTalk Desktop.exe",
//           "verbose_name": "ReadyTalk Desktop",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 436,
//           "name": "join.me.exe",
//           "verbose_name": "join.me",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 438,
//           "name": "GLOBAL~1.EXE",
//           "verbose_name": "GLOBAL~1",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 440,
//           "name": "BlueJeans.exe",
//           "verbose_name": "BlueJeans",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 442,
//           "name": "Mikogo-Screen-Service.exe",
//           "verbose_name": "Mikogo-Screen-Service",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 444,
//           "name": "Mikogo-Service.exe",
//           "verbose_name": "Mikogo-Service",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 446,
//           "name": "Mikogo-video.exe",
//           "verbose_name": "Mikogo-video",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 448,
//           "name": "ScreenShare.exe",
//           "verbose_name": "ScreenShare",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 450,
//           "name": "EZTalks.exe",
//           "verbose_name": "EZTalks",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 452,
//           "name": "iMeet.exe",
//           "verbose_name": "iMeet",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 454,
//           "name": "VidyoDesktop.exe",
//           "verbose_name": "VidyoDesktop",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 456,
//           "name": "g2mvideoconference.exe",
//           "verbose_name": "g2mvideoconference",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 458,
//           "name": "FCC EN.exe",
//           "verbose_name": "FCC EN",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 460,
//           "name": "ClickMeeting.exe",
//           "verbose_name": "ClickMeeting",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 462,
//           "name": "Via3.exe",
//           "verbose_name": "Via3",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 464,
//           "name": "Zoom.exe",
//           "verbose_name": "Zoom",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 466,
//           "name": "ooVoo.exe",
//           "verbose_name": "ooVoo",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 468,
//           "name": "Viber.exe",
//           "verbose_name": "Viber",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 476,
//           "name": "ScnRec.exe",
//           "verbose_name": "ScnRec",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 478,
//           "name": "SRSetup.exe",
//           "verbose_name": "SRSetup",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 484,
//           "name": "connect.exe",
//           "verbose_name": "connect",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 486,
//           "name": "HprSnap8.exe",
//           "verbose_name": "HprSnap8",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 498,
//           "name": "WhatsApp.exe",
//           "verbose_name": "WhatsApp",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 502,
//           "name": "ScnRec.exe",
//           "verbose_name": "ScnRec",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 503,
//           "name": "LightShot.exe",
//           "verbose_name": "LightShot",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 505,
//           "name": "notepad++.exe",
//           "verbose_name": "notepad++",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 506,
//           "name": "onenoteim.exe",
//           "verbose_name": "onenoteim",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 507,
//           "name": "onenotem.exe",
//           "verbose_name": "onenotem",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 508,
//           "name": "lync.exe",
//           "verbose_name": "lync",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 509,
//           "name": "mspaint.exe",
//           "verbose_name": "mspaint",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 510,
//           "name": "AtAuthor.exe",
//           "verbose_name": "AtAuthor",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 511,
//           "name": "Teams.exe",
//           "verbose_name": "Teams",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 514,
//           "name": "atmgr.exe",
//           "verbose_name": "atmgr",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 515,
//           "name": "screencapture.exe",
//           "verbose_name": "screencapture",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 516,
//           "name": "sgtool.exe",
//           "verbose_name": "sgtool",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 517,
//           "name": "WeChat.exe",
//           "verbose_name": "WeChat",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 518,
//           "name": "WeChatWeb.exe",
//           "verbose_name": "WeChatWeb",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 520,
//           "name": "icq.exe",
//           "verbose_name": "icq",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 522,
//           "name": "uTox_win64.exe",
//           "verbose_name": "uTox_win64",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 524,
//           "name": "LINE.exe",
//           "verbose_name": "LINE",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 527,
//           "name": "vsee.exe",
//           "verbose_name": "vsee",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 528,
//           "name": "DroidCamApp.exe",
//           "verbose_name": "DroidCamApp",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 529,
//           "name": "ekiga.exe",
//           "verbose_name": "ekiga",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 531,
//           "name": " AtAuthor.exe",
//           "verbose_name": " AtAuthor",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 532,
//           "name": "SRServer.exe",
//           "verbose_name": "SRServer",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 533,
//           "name": "strwinclt.exe",
//           "verbose_name": "strwinclt",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 534,
//           "name": "Mirroring360.exe",
//           "verbose_name": "Mirroring360",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 535,
//           "name": "QQ.exe",
//           "verbose_name": "QQ",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 538,
//           "name": "ApowerREC.exe",
//           "verbose_name": "ApowerREC",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 540,
//           "name": "skypebackgroundhost.exe",
//           "verbose_name": "skypebackgroundhost",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 541,
//           "name": "SkypeBridge.exe",
//           "verbose_name": "SkypeBridge",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 542,
//           "name": "EXCEL.exe",
//           "verbose_name": "Excel",
//           "os_type": "Windows",
//           "application_file_name": "Excel.lnk"
//         },
//         {
//           "id": 546,
//           "name": "mstsc.exe",
//           "verbose_name": "mstsc",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 548,
//           "name": "Spotify.exe",
//           "verbose_name": "Spotify",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 550,
//           "name": "screensketch.exe",
//           "verbose_name": "screensketch",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 551,
//           "name": "soffice.exe",
//           "verbose_name": "soffice",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 552,
//           "name": "soffice.bin",
//           "verbose_name": "soffice.bin",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 556,
//           "name": "VirtualBox.exe",
//           "verbose_name": "VirtualBox",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 558,
//           "name": "youdaodict.exe",
//           "verbose_name": "youdaodict",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 561,
//           "name": "forticlient.exe",
//           "verbose_name": "forticlient",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 562,
//           "name": "kakaotalk.exe.",
//           "verbose_name": "kakaotalk",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 570,
//           "name": "libreoffice.exe",
//           "verbose_name": "libreoffice",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 571,
//           "name": "Microsoft.Notes.exe",
//           "verbose_name": "Microsoft.Notes",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 576,
//           "name": "AcroRd32.exe",
//           "verbose_name": "AcroRd32",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 578,
//           "name": "RemotePCDesktop.exe",
//           "verbose_name": "RemotePCDesktop",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 580,
//           "name": "Mtb.exe",
//           "verbose_name": "Mtb",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 584,
//           "name": "Zt.exe",
//           "verbose_name": "Zt",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 586,
//           "name": "Natspeak.exe",
//           "verbose_name": "Natspeak",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 587,
//           "name": "Kindle.exe",
//           "verbose_name": "Kindle",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 589,
//           "name": "OUTLOOK.EXE",
//           "verbose_name": "OUTLOOK",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 591,
//           "name": "dgnsvc.exe",
//           "verbose_name": "dgnsvc",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 592,
//           "name": "DWRCS.exe",
//           "verbose_name": "DWRCS",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 593,
//           "name": "firefox.exe",
//           "verbose_name": "firefox",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 595,
//           "name": "ReadAndWrite.exe",
//           "verbose_name": "ReadAndWrite",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 596,
//           "name": "Texthelp Service Bridge.exe",
//           "verbose_name": "Texthelp Service Bridge",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 599,
//           "name": "Kurzweil 3000.exe",
//           "verbose_name": "Kurzweil 3000",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 600,
//           "name": "idea64.exe",
//           "verbose_name": "idea64",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 601,
//           "name": "pycharm64.exe",
//           "verbose_name": "pycharm64",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 603,
//           "name": "itunes.exe",
//           "verbose_name": "itunes",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 605,
//           "name": "Code.exe",
//           "verbose_name": "Code",
//           "os_type": "Windows",
//           "application_file_name": "Visual Studio Code.lnk"
//         },
//         {
//           "id": 606,
//           "name": "atom.exe",
//           "verbose_name": "atom",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 607,
//           "name": "pythonw.exe",
//           "verbose_name": "pythonw",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 608,
//           "name": "netbeans64.exe",
//           "verbose_name": "netbeans64",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 609,
//           "name": "geany.exe",
//           "verbose_name": "geany",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 610,
//           "name": "komodo.exe",
//           "verbose_name": "komodo",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 611,
//           "name": "studio64.exe",
//           "verbose_name": "studio64",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 612,
//           "name": "jdeveloper.exe",
//           "verbose_name": "jdeveloper",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 613,
//           "name": "codeblocks.exe",
//           "verbose_name": "codeblocks",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 614,
//           "name": "clion64.exe",
//           "verbose_name": "clion64",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 615,
//           "name": "Codelite.exe",
//           "verbose_name": "Codelite",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 616,
//           "name": "Turbo C++.exe",
//           "verbose_name": "Turbo C++",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 617,
//           "name": "Messenger.exe",
//           "verbose_name": "Messenger",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 618,
//           "name": "thunderbird.exe",
//           "verbose_name": "thunderbird",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 619,
//           "name": "filezilla.exe",
//           "verbose_name": "filezilla",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 620,
//           "name": "JioMeet.exe",
//           "verbose_name": "JioMeet",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 621,
//           "name": "Flock.exe",
//           "verbose_name": "Flock",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 622,
//           "name": "steam.exe",
//           "verbose_name": "steam",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 623,
//           "name": "Telegram.exe",
//           "verbose_name": "Telegram",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 624,
//           "name": "GrammarlyForWindows.exe",
//           "verbose_name": "GrammarlyForWindows",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 625,
//           "name": "WWAHost.exe",
//           "verbose_name": "WWAHost",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 626,
//           "name": "WinStore.App.exe",
//           "verbose_name": "WinStore.App",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 627,
//           "name": "HxOutlook.exe",
//           "verbose_name": "HxOutlook",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 628,
//           "name": "cmd.exe",
//           "verbose_name": "cmd",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 629,
//           "name": "powershell.exe",
//           "verbose_name": "powershell",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 630,
//           "name": "UCBrowser.exe",
//           "verbose_name": "UCBrowser",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 631,
//           "name": "iexplore.exe",
//           "verbose_name": "iexplore",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 632,
//           "name": "opera.exe",
//           "verbose_name": "opera",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 633,
//           "name": "discord.exe",
//           "verbose_name": "discord",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 676,
//           "name": "MicrosoftEdge.exe",
//           "verbose_name": "MicrosoftEdge",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 677,
//           "name": "calculator.exe",
//           "verbose_name": "Calculator",
//           "os_type": "Windows",
//           "application_file_name": "calculator.lnk"
//         },
//         {
//           "id": 678,
//           "name": "chrome.exe",
//           "verbose_name": "Chrome",
//           "os_type": "Windows",
//           "application_file_name": "Google Chrome.lnk"
//         },
//         {
//           "id": 679,
//           "name": "ONLINENT.EXE",
//           "verbose_name": "",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 680,
//           "name": "ProctortrackExamBrowser.exe",
//           "verbose_name": "",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 681,
//           "name": "ProctorTA.exe",
//           "verbose_name": "",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 682,
//           "name": "WatchDogProcess.exe",
//           "verbose_name": "",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 683,
//           "name": "node.exe",
//           "verbose_name": "",
//           "os_type": "Windows",
//           "application_file_name": null
//         },
//         {
//           "id": 684,
//           "name": "JupyterLab.exe",
//           "verbose_name": "JupyterLab",
//           "os_type": "Windows",
//           "application_file_name": "JupyterLab.lnk"
//         },
//         {
//           "id": 695,
//           "name": "Acrobat.exe",
//           "verbose_name": "Acrobat",
//           "os_type": "Windows",
//           "application_file_name": "Adobe Acrobat.lnk"
//         },
//         {
//           "id": 696,
//           "name": "BDuserhost.exe ",
//           "verbose_name": "Bitdefender",
//           "os_type": "Windows",
//           "application_file_name": ""
//         },
//         {
//           "id": 697,
//           "name": "Teamviewerqs.exe",
//           "verbose_name": "PTSupport TeamViewer",
//           "os_type": "Windows",
//           "application_file_name": ""
//         }
//       ],
//       "whitelist_mac_apps": [
//         {
//           "id": 47,
//           "name": "Adobe Captivate",
//           "verbose_name": "Adobe Captivate",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 48,
//           "name": "Camtasia 2",
//           "verbose_name": "Camtasia 2",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 49,
//           "name": "Grabilla",
//           "verbose_name": "Grabilla",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 50,
//           "name": "Jing",
//           "verbose_name": "Jing",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 51,
//           "name": "QuickTime Player",
//           "verbose_name": "QuickTime Player",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 52,
//           "name": "ScreenFlow",
//           "verbose_name": "ScreenFlow",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 53,
//           "name": "Snapz Pro X",
//           "verbose_name": "Snapz Pro X",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 54,
//           "name": "VLC",
//           "verbose_name": "VLC",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 55,
//           "name": "2X Client RDP",
//           "verbose_name": "2X Client RDP",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 56,
//           "name": "iChat",
//           "verbose_name": "iChat",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 57,
//           "name": "tlclient",
//           "verbose_name": "tlclient",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 58,
//           "name": "Chicken of the VNC",
//           "verbose_name": "Chicken of the VNC",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 59,
//           "name": "CrossLoop",
//           "verbose_name": "CrossLoop",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 60,
//           "name": "vncviewer",
//           "verbose_name": "vncviewer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 61,
//           "name": "blaze",
//           "verbose_name": "blaze",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 62,
//           "name": "GoToMyPC",
//           "verbose_name": "GoToMyPC",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 63,
//           "name": "Toolkit",
//           "verbose_name": "Toolkit",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 64,
//           "name": "Mac HelpMate",
//           "verbose_name": "Mac HelpMate",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 65,
//           "name": "Mikogo",
//           "verbose_name": "Mikogo",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 66,
//           "name": "nxdock",
//           "verbose_name": "nxdock",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 67,
//           "name": "nxplayer",
//           "verbose_name": "nxplayer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 68,
//           "name": "vncserver_servic",
//           "verbose_name": "vncserver_servic",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 69,
//           "name": "vnc_viewer",
//           "verbose_name": "vnc_viewer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 70,
//           "name": "Splashtop Stream",
//           "verbose_name": "Splashtop Stream",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 71,
//           "name": "Splashtop Person",
//           "verbose_name": "Splashtop Person",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 72,
//           "name": "TEAMVIEWER",
//           "verbose_name": "TEAMVIEWER",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 73,
//           "name": "TEAMVIEWER_SERVI",
//           "verbose_name": "TEAMVIEWER_SERVI",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 74,
//           "name": "TEAMVIEWER_DESKT",
//           "verbose_name": "TEAMVIEWER_DESKT",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 75,
//           "name": "TightVNC Viewer",
//           "verbose_name": "TightVNC Viewer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 76,
//           "name": "JavaApplicationS",
//           "verbose_name": "JavaApplicationS",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 77,
//           "name": "GST_Launcher",
//           "verbose_name": "GST_Launcher",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 78,
//           "name": "Vash_Wrapper",
//           "verbose_name": "Vash_Wrapper",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 79,
//           "name": "Window-Switch",
//           "verbose_name": "Window-Switch",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 80,
//           "name": "Xpra_Launcher",
//           "verbose_name": "Xpra_Launcher",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 86,
//           "name": "GoToMeeting",
//           "verbose_name": "GoToMeeting",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 87,
//           "name": "Skype",
//           "verbose_name": "Skype",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 98,
//           "name": "AdobeCaptivate",
//           "verbose_name": "AdobeCaptivate",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 101,
//           "name": "Camtasia2",
//           "verbose_name": "Camtasia2",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 104,
//           "name": "QuickTimePlayer",
//           "verbose_name": "QuickTimePlayer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 107,
//           "name": "SnapzProX",
//           "verbose_name": "SnapzProX",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 110,
//           "name": "2XClientRDP",
//           "verbose_name": "2XClientRDP",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 113,
//           "name": "ChickenoftheVNC",
//           "verbose_name": "ChickenoftheVNC",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 116,
//           "name": "MacHelpMate",
//           "verbose_name": "MacHelpMate",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 119,
//           "name": "SplashtopStream",
//           "verbose_name": "SplashtopStream",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 122,
//           "name": "SplashtopPerson",
//           "verbose_name": "SplashtopPerson",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 125,
//           "name": "TightVNCViewer",
//           "verbose_name": "TightVNCViewer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 131,
//           "name": "GoogleHangout",
//           "verbose_name": "GoogleHangout",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 229,
//           "name": "Snagit",
//           "verbose_name": "Snagit",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 232,
//           "name": "SnagitAppleScriptExecutor",
//           "verbose_name": "SnagitAppleScriptExecutor",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 235,
//           "name": "SnagitHelper",
//           "verbose_name": "SnagitHelper",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 238,
//           "name": "SnagitRecorder",
//           "verbose_name": "SnagitRecorder",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 241,
//           "name": "ScreenFlowHelper",
//           "verbose_name": "ScreenFlowHelper",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 244,
//           "name": "ScreenFlowRecorder",
//           "verbose_name": "ScreenFlowRecorder",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 247,
//           "name": "ActivePresenter",
//           "verbose_name": "ActivePresenter",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 250,
//           "name": "Screenmailer",
//           "verbose_name": "Screenmailer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 253,
//           "name": "TinyTake",
//           "verbose_name": "TinyTake",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 256,
//           "name": "Screencast-O-Matic",
//           "verbose_name": "Screencast-O-Matic",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 260,
//           "name": "BitTorrent",
//           "verbose_name": "BitTorrent",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 268,
//           "name": "remote_assistance_host",
//           "verbose_name": "remote_assistance_host",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 270,
//           "name": "remote_assistanc",
//           "verbose_name": "remote_assistanc",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 286,
//           "name": "Evernote",
//           "verbose_name": "Evernote",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 314,
//           "name": "Screenflick",
//           "verbose_name": "Screenflick",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 320,
//           "name": "LogMein Client",
//           "verbose_name": "LogMein Client",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 322,
//           "name": "Slack",
//           "verbose_name": "Slack",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 360,
//           "name": "Capto",
//           "verbose_name": "Capto",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 362,
//           "name": "AirServer",
//           "verbose_name": "AirServer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 366,
//           "name": "x-mirage",
//           "verbose_name": "x-mirage",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 388,
//           "name": "SnapNDrag",
//           "verbose_name": "SnapNDrag",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 412,
//           "name": "Backup and sync from Google",
//           "verbose_name": "Backup and sync from Google",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 414,
//           "name": "Dropbox.exe",
//           "verbose_name": "Dropbox",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 416,
//           "name": "HipChat",
//           "verbose_name": "HipChat",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 420,
//           "name": "Camtasia 3",
//           "verbose_name": "Camtasia 3",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 422,
//           "name": "Monosnap",
//           "verbose_name": "Monosnap",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 424,
//           "name": "Parallels Desktop",
//           "verbose_name": "Parallels Desktop",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 474,
//           "name": "ScnRec",
//           "verbose_name": "ScnRec",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 480,
//           "name": "Messages",
//           "verbose_name": "Messages",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 482,
//           "name": "FaceTime",
//           "verbose_name": "FaceTime",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 488,
//           "name": "Xpra",
//           "verbose_name": "Xpra",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 490,
//           "name": "Snagit 2018",
//           "verbose_name": "Snagit 2018",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 492,
//           "name": "VNC Server",
//           "verbose_name": "VNC Server",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 494,
//           "name": "Spashtop Streamer",
//           "verbose_name": "Spashtop Streamer",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 496,
//           "name": "Spashtop personal",
//           "verbose_name": "Spashtop personal",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 500,
//           "name": "WhatsApp",
//           "verbose_name": "WhatsApp",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 504,
//           "name": "Screenshot",
//           "verbose_name": "Screenshot",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 512,
//           "name": "teams",
//           "verbose_name": "teams",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 513,
//           "name": "meeting center",
//           "verbose_name": "meeting center",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 519,
//           "name": "WeChat",
//           "verbose_name": "WeChat",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 521,
//           "name": "ICQ",
//           "verbose_name": "ICQ",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 523,
//           "name": "utox",
//           "verbose_name": "utox",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 525,
//           "name": "Kakaotalk",
//           "verbose_name": "Kakaotalk",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 526,
//           "name": "FaceTime",
//           "verbose_name": "FaceTime",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 530,
//           "name": "VSee",
//           "verbose_name": "VSee",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 536,
//           "name": "QQ",
//           "verbose_name": "QQ",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 537,
//           "name": "zoom.us",
//           "verbose_name": "zoom.us",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 539,
//           "name": "ApowerREC",
//           "verbose_name": "ApowerREC",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 543,
//           "name": "Microsoft Excel",
//           "verbose_name": "Microsoft Excel",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 547,
//           "name": "Skype for Business",
//           "verbose_name": "Skype for Business",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 549,
//           "name": "Photo Booth",
//           "verbose_name": "Photo Booth",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 553,
//           "name": "LibreOffice",
//           "verbose_name": "LibreOffice",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 554,
//           "name": "soffice",
//           "verbose_name": "soffice",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 555,
//           "name": "VirtualBox",
//           "verbose_name": "VirtualBox",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 557,
//           "name": "Microsoft Remote",
//           "verbose_name": "Microsoft Remote",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 559,
//           "name": "YoudaoDict",
//           "verbose_name": "YoudaoDict",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 569,
//           "name": "Pages",
//           "verbose_name": "Pages",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 575,
//           "name": "Acrobat Reader",
//           "verbose_name": "Acrobat Reader",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 577,
//           "name": "Acrobat Reader",
//           "verbose_name": "Acrobat Reader",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 579,
//           "name": "remotepcdesktoph",
//           "verbose_name": "remotepcdesktoph",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 581,
//           "name": "Minitab",
//           "verbose_name": "Minitab",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 582,
//           "name": "Notes",
//           "verbose_name": "Notes",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 583,
//           "name": "TextEdit",
//           "verbose_name": "TextEdit",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 585,
//           "name": "Microsoft Word",
//           "verbose_name": "Microsoft Word",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 588,
//           "name": "Kindle",
//           "verbose_name": "Kindle",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 590,
//           "name": "Microsoft Outlook",
//           "verbose_name": "Microsoft Outlook",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 594,
//           "name": "firefox",
//           "verbose_name": "firefox",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 597,
//           "name": "Read&Write",
//           "verbose_name": "Read&Write",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 598,
//           "name": "Kurzweil 3000",
//           "verbose_name": "Kurzweil 3000",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 634,
//           "name": "Intelij IDEA CE",
//           "verbose_name": "Intelij IDEA CE",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 635,
//           "name": "PyCharm",
//           "verbose_name": "PyCharm",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 636,
//           "name": "iTunes",
//           "verbose_name": "iTunes",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 637,
//           "name": "Eclipse",
//           "verbose_name": "Eclipse",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 638,
//           "name": "Visual Studio",
//           "verbose_name": "Visual Studio",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 639,
//           "name": "Atom",
//           "verbose_name": "Atom",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 640,
//           "name": "IDLE",
//           "verbose_name": "IDLE",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 641,
//           "name": "NetBeans 8.2",
//           "verbose_name": "NetBeans 8.2",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 642,
//           "name": "Geany",
//           "verbose_name": "Geany",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 643,
//           "name": "Komodo IDE 12",
//           "verbose_name": "Komodo IDE 12",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 644,
//           "name": "BlueJ",
//           "verbose_name": "BlueJ",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 645,
//           "name": "Android Studio",
//           "verbose_name": "Android Studio",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 646,
//           "name": "JDeveloper",
//           "verbose_name": "JDeveloper",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 647,
//           "name": "CodeBlocks",
//           "verbose_name": "CodeBlocks",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 648,
//           "name": "CLion",
//           "verbose_name": "CLion",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 649,
//           "name": "CodeLite",
//           "verbose_name": "CodeLite",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 650,
//           "name": "Xcode",
//           "verbose_name": "Xcode",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 651,
//           "name": "AnyDesk",
//           "verbose_name": "AnyDesk",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 652,
//           "name": "Messenger",
//           "verbose_name": "Messenger",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 653,
//           "name": "ThunderBirdVPN",
//           "verbose_name": "ThunderBirdVPN",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 654,
//           "name": "Microsoft Office",
//           "verbose_name": "Microsoft Office",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 655,
//           "name": "FileZilla Pro",
//           "verbose_name": "FileZilla Pro",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 656,
//           "name": "JioMeet",
//           "verbose_name": "JioMeet",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 657,
//           "name": "Flock",
//           "verbose_name": "Flock",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 658,
//           "name": "Microsoft Teams",
//           "verbose_name": "Microsoft Teams",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 659,
//           "name": "Telegram",
//           "verbose_name": "Telegram",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 660,
//           "name": "Grammarly for Safari",
//           "verbose_name": "Grammarly for Safari",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 661,
//           "name": "Twitter",
//           "verbose_name": "Twitter",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 662,
//           "name": "Dropbox",
//           "verbose_name": "Dropbox",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 663,
//           "name": "Google Drive",
//           "verbose_name": "Google Drive",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 664,
//           "name": "onedrive",
//           "verbose_name": "onedrive",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 665,
//           "name": "iCloud Drive",
//           "verbose_name": "iCloud Drive",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 666,
//           "name": "App Store",
//           "verbose_name": "App Store",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 667,
//           "name": "Mail",
//           "verbose_name": "Mail",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 668,
//           "name": "iMail",
//           "verbose_name": "iMail",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 669,
//           "name": "Terminal",
//           "verbose_name": "Terminal",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 670,
//           "name": "UC Browser",
//           "verbose_name": "UC Browser",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 671,
//           "name": "Opera",
//           "verbose_name": "Opera",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 672,
//           "name": "Discord",
//           "verbose_name": "Discord",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 673,
//           "name": "Viber",
//           "verbose_name": "Viber",
//           "os_type": "Mac",
//           "application_file_name": null
//         },
//         {
//           "id": 675,
//           "name": "Safari",
//           "verbose_name": "Safari",
//           "os_type": "Mac",
//           "application_file_name": null
//         }
//       ],
//       "is_webcam_not_required": false,
//       "virtual_machine_check_required": true,
//       "vm_guard_string": "/QEMU|Linux KVM|Linux lguest|OpenVZ|Microsoft Virtual PC|VMWare|linux-vserver|Xen|FreeBSD Jail|OpenVZ Host|VirtualBox|Parallels|Linux Containers|LXC|innotek GmbH|red hat|Microsoft corporation|HVM domU|Bochs|kvm/gi",
//       "is_keyboard_check_required": true,
//       "is_mouse_check_required": true,
//       "is_harddisk_check_required": true,
//       "is_monitor_check_required": true,
//       "is_blapps_check_required": true,
//       "is_cast_device_check_required": true,
//       "is_wireless_monitor_allowed": false,
//       "is_ram_check_required": false,
//       "minimum_ram_required": 4,
//       "is_photo_id_required": true,
//       "is_real_time_idv_required": false,
//       "idv_max_attempt_count": 3,
//       "is_face_scan_required": true,
//       "is_screen_sharing_detection_enabled": false,
//       "is_external_monitor_disabled": false,
//       "is_offline_chunck_supported": true,
//       "is_cv_required": true,
//       "is_copy_paste_disabled": true,
//       "is_ds_monitoring_required": true,
//       "is_keyboard_shortcut_disabled": true,
//       "is_multiple_monitor_allowed": true,
//       "is_new_tabs_windows_restricted": false,
//       "is_online_aid_allowed": false,
//       "is_print_disabled": true,
//       "is_print_screen_disabled": true,
//       "is_right_click_disabled": true,
//       "is_video_monitoring_required": true,
//       "multiple_monitor_alert_interval": 10000,
//       "whitelisted_urls": ["https://dev.verificient.com"],
//       "video_frame_width": 320,
//       "video_frame_height": 240,
//       "screenshot_width": 1024,
//       "screenshot_height": 576,
//       "test_proctoring_level": "2",
//       "show_custom_message_on_incident": true,
//       "survey_url": "https://www.surveymonkey.com/r/NN2Q2N2",
//       "proxy_config": {
//         "mode": "fixed_servers",
//         "rules": {
//           "singleProxy": {
//             "scheme": "http",
//             "host": "34.75.88.182",
//             "port": 8888
//           }
//         }
//       },
//       "proxy_password": "fcutd9RzXpsnC47j",
//       "proxy_username": "dmvuser",
//       "restrict_user_in_vm": false,
//       "sentry_dns_url": "https://6e81cb04bc6c4adeaaeca0d984bf2674@o489068.ingest.sentry.io/5551042",
//       "allow_short_breaks": false,
//       "violation_rule_book": {
//         "background_audio": {
//           "disqualify_after_attempts": "NA",
//           "max_infraction_time_in_sec_to_disqualify": "NA",
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "NA",
//           "warning_message": "Our proctoring system has detected unauthorized sound. Please ensure you are in a quiet environment before proceeding. Repeated instances may result in a failed test."
//         },
//         "background_motion": {
//           "disqualify_after_attempts": "NA",
//           "max_infraction_time_in_sec_to_disqualify": "NA",
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected suspicious background motion. Please ensure that you are in a private area before proceeding. "
//         },
//         "camera_feed_issue": {
//           "disqualify_after_attempts": 1,
//           "max_infraction_time_in_sec_to_disqualify": 30,
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system is unable to successfully capture your video feed. Please fix within 30 seconds\"or your test will be disqualified"
//         },
//         "copy_paste": {
//           "disqualify_after_attempts": 2,
//           "max_infraction_time_in_sec_to_disqualify": "NA",
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected a possible data copy attempt. This is a violation of proctoring rules. A second instance may result in a failed test."
//         },
//         "data_upload": {
//           "disqualify_after_attempts": 3,
//           "max_infraction_time_in_sec_to_disqualify": 15,
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system is experiencing delays finalizing your session. Please do not close browser tab until upload is complete."
//         },
//         "face_out_of_frame": {
//           "disqualify_after_attempts": 3,
//           "max_infraction_time_in_sec_to_disqualify": 15,
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system is unable to fully monitor your face. Please remain in full view of the camera for your entire test. Repeated instances may result in a failed test."
//         },
//         "fsla": {
//           "disqualify_after_attempts": 3,
//           "max_infraction_time_in_sec_to_disqualify": 15,
//           "min_infraction_time_in_sec_to_warn": 5,
//           "real_time_action": "warning",
//           "warning_message": "You appear to be looking off camera for unspecified reasons. This is a violation of proctoring rules. Please pay attention only to your exam. Repeated instances may result in a failed test."
//         },
//         "full_screen_exit_attempt": {
//           "disqualify_after_attempts": 2,
//           "max_infraction_time_in_sec_to_disqualify": "NA",
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected that you may be accessing unauthorized materials. Please close all browser windows other than your exam. Repeated instances may result in a failed test."
//         },
//         "imposter": {
//           "disqualify_after_attempts": 1,
//           "max_infraction_time_in_sec_to_disqualify": 30,
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected an unauthorized individual on camera. This is a violation of proctoring rules. Repeated instances may result in a failed test."
//         },
//         "left_session": {
//           "disqualify_after_attempts": 3,
//           "max_infraction_time_in_sec_to_disqualify": 15,
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected that you may have left the session. Please remain on camera for the entire exam. Repeated instances may result in a failed test."
//         },
//         "mobile_detected": {
//           "disqualify_after_attempts": 4,
//           "max_infraction_time_in_sec_to_disqualify": 5,
//           "min_infraction_time_in_sec_to_warn": 5,
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected a cell phone present. Please do not attempt to access any devices during your exam. Repeated instances may result in a failed test."
//         },
//         "multiple_people": {
//           "disqualify_after_attempts": 1,
//           "max_infraction_time_in_sec_to_disqualify": 30,
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected more than one person present. Please ensure you are alone in front of your computer. Repeated instances may result in a failed test."
//         },
//         "online_aid": {
//           "disqualify_after_attempts": 2,
//           "max_infraction_time_in_sec_to_disqualify": "NA",
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected that you may be accessing unauthorized materials. Please close all browser windows other than your exam. Repeated instances may result in a failed test."
//         },
//         "print": {
//           "disqualify_after_attempts": 2,
//           "max_infraction_time_in_sec_to_disqualify": "NA",
//           "min_infraction_time_in_sec_to_warn": "NA",
//           "real_time_action": "warning",
//           "warning_message": "Our proctoring system has detected a possible data copy attempt. This is a violation of proctoring rules. A second instance may result in a failed test."
//         }
//       }
//     }
//     let path_config={
//       //darknet_model_configuration_path:'D:\\cvd_node_addon\\cvd_code\\models\\yolo_tiny_enet_18_febb_21\\enet-coco-train.cfg',
//       //darknet_model_weights_path:'D:\\cvd_node_addon\\cvd_code\\models\\yolo_tiny_enet_18_febb_21\\enet-coco-train_3500.weights',
//       yolo_v8_nano_model_path:'.\\models\\od_v8_nano_feb24_2.onnx',
//       dlib_shape_pred_model_path:'D:\\CVD\\models\\shape_predictor_68_face_landmarks.dat',
//     // pose_network_path:'D:\\CVD_Addon\\CVD1\\resnet_model_18_feb_21.pb',
//       pose_network_path:'.\\models\\hp_v8_nano_feb_2024.pb',
//       dlib_face_recognition_resnet_model_v1_path:'D:\\CVD_Addon\\CVD1\\dlib_face_recognition_resnet_model_v1.dat',
//       flash_detector_net_path:'D:\\CVD\\models\\flash_ported.pb',
//       log_dir_path:'.\\Log',
//       retina_para:'.\\models\\retina\\mnet.25-opt.param',
//       retina_mode:'.\\models\\retina\\mnet.25-opt.bin',
//       arc_para:'.\\models\\mobilefacenet\\mobilefacenet.param',
//       arc_mode:'.\\models\\mobilefacenet\\mobilefacenet.bin',
//       onnx_vad_path:'D:\\Demo\\CVD_VAD\\src\\silero_vad.onnx'
//     };
//     const initButton = document.getElementById('initButton');
//     ipcRenderer.send('init', { n: 'init', a: Configuration, b: path_config });
//     // Disable the button
//     initButton.disabled = true;

// }
// function img_Submit(event){
//     event.preventDefault();
//     console.log(event.target.w3review.value);
//     let base64_img=event.target.w3review.value;
//     ipcRenderer.send('detect_object',{n:'od',a:base64_img});
// }
// function detect_object(){
//     let base64_img="/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAD6ASwDASIAAhEBAxEB/8QAHQAAAAcBAQEAAAAAAAAAAAAAAQIDBAUGBwgACf/EAEcQAAEDAgUCBAMGAwQHCAMBAAECAxEABAUGEiExB0ETIlFhcYGRCBQVIzKhQrHBUmJy0RYkM0OCosI0NlOSstLh8DVj8YP/xAAbAQACAwEBAQAAAAAAAAAAAAABAgADBAUGB//EADARAAICAQQBBAEDAwMFAAAAAAABAhEDBBIhMUEFEyJRYRQygXGh8CMksTNSwdHh/9oADAMBAAIRAxEAPwCKyLj+KYnigsrh/WkgrBMEiOf51qlvYkpALhJ5rHelLQOOOkJ2S0T8JIFbbaohA3Jjmq8krmGtsVQVNlIjX+1LJw8ad3eKXSgAyTt9KXSoRsfTYighbGgsEggByTzxRzh451707CUp3BJk778UYQeDRtE7GqcPE/7WaP8AhqDALh5+dOEngCPpRxKiNx8aO3yHd4G34e2OHFUIsm5JJPtTggjvvXjxHempA3WNzYNn+JQNe/DwRBcNOQSeRQysngRTLnglvsaHDUEAazt7Cg/D0TGsn1p7IIoAByBvRXBNzGn4e2DutXoKH8Ob3853245p0VbSBQFRHpRFGv4c2nv9RRDYNKJ8/wBKdKUqd429aAkxOpI+VRRDbYxXh6CrZxQmkzh7adi6Z9KdrWSZIG9EUvUI0hJO81FDkA0XYtnZKlbDvSf3JMfqNPORQEATHJpmiDJVik7lZn2prcWqNBGoj3qTWdNM7kwg70Gvohh/U++vbHG2rdm5cQhTQPl5nUa0DJjH3rA7J15SitTKSqe5is86uyMetlbR4H/Ua0bIJ8TLtguP1NJ/qP6Va3UORmixizbiJI+VD91aBgk7UqDtvt86KdhMyKriml2LYiWGk7gn503ct2yDvvTxRhPlJpFwnTRZOyFbASdPrS7QMTG00kRqcJMmSflSzRH6SaLplbXgcJ3MggD0pZB79vfikUkERAFQOc8zW2XcNcUFhT7g0to7796lXwgcEX1Ezo3hNsrC7JaS+4k+IR/CDt9aw/MWPfgNup0KKsTu0y0n/wABBH6z7+gp9jmNizadx7EiH3Vz92ZVMuuepE/pHf32qFyblW/zdiqsZxeVpKtaisQNv4vhWmGPatwl2xz05yPc3734vikJSBrJUP0COd+9aO5j7toRb4asM26AAgaZJ9zTfEL+3ZYThtgnRatcqHLivVX9KhVl1R1F0NzvB5+NJkyOX9BtpfukqArELpSEnZoSTz+qtjt/0AAAVk3SBn/tzpJkFCdvTmtatyNAgyK5sq3to2ydpDpCI2JpYJgwaSTICRApUKUmJjc0xU3YZO59Zo8eh2oiVEzv+1GUoHaYFRdkug0juN69OngAzRAoHvQa5MAintgFAokE7CfWjAeUaqT8QHy+lH8TjbmgGkARBECjat4miKWNpIoCrzbCRTJUBioFegzIFFSoRsDQ69/0mPWiwAiRM0VRA37UGpPBP0oFQBtRRD0k896TUR2MR2oS4IKgBHEUipZKtI7enemogJiSVCJ2+FJncwqIFCsmRq5jaikgmY3qEC8iQAPfvRDMzR1K5Pek1uHgiZopgoIvghRI95po+SQdQ4pyVAzPPbamz5VpO3Y0H0FGIdXkH8btjA3ZPb+8a0Lp9tlfD9+G/wCpmqF1fGrGbNRP+6VPw1Ve+nhjLVlO8NEf8xplbjyPKXCLXtEzIr20n0oYnyjvvXlD5UE6K2EcEfCkVplPH70uoSImkVgx+1TyRMhjs6oHgHfelW0yCdKT8aScSPHVzE0uiEx6VLJQliF61h1i9ev/AOzYQpavgBJrCMzZiOMXVxjOKOKTaMkgJ/tf2UJHqY39BvW7YhZs4jYP2bkBDyCgx71k2YekH4i8ylWMFFpbpISwlHJP6lEzyatwuCdSdFcot9GaYNhGJZ+xwXNw2UW7Z0JQmQhCR2T7eprTbhy2wizRhWGgIaSAlxY21n/KrBgWVGMFs04ZZmGgIccGxI7JFV3PRThS7m90BTbSAUtjjUdvpNHJNSkoLokYtfLyVrGcUZw1oPPK1OLB8JoHdR9/aqk8i9xN1V3duK8RfZKoCR2Apy21dX7ov71zW6vf2A9KefdwOVVYkodiOT8G6dImtNjdLkSp1M9+xrU2gogCAY9BWddIrcJwd5wJjU+f5CtOt2NSQNXyiuY1yzZO0wGxqUAof0pbT3596MhpIV+kelKJQB5gncdqiEsS8wB5AigI4kx8qcFMiCKANA9v3ooYQ077cUcJ3O0Ur4W8SRtQ6ACJinFoSCQk8bGhAJ7zSugTpijeFH6QBUXIBDwpPt+9ClEK9ppYMz3oS0QNqbcyCagI32pPk+1LlpXBFBo3ExNC0yCcDsPjSbnECnKkhKZE+9QuY8w4dlyzF5frWdR0oQ2nUtZ9APWmSt0iPhWOyTuFbUmRpMgyaw7NvXjGLdZ/DkWlgwiQqU+M6NuNylM+wmonB+redsWYuLu1xOxcaaTKA4wAtR8uwAMfxCfSrXgyJWIskX0dCqVwYM/GipKSdh86wu86549gbYTcWtherKQQtC1ACSdtM87Uhhn2pLZ1amb3LpUobqLDoVsPZUUY6fJJcKwPLGPZvaoB0gT3maJVGy11hyvmXQq0uWWXHB5WH3PDdUe8SNB+GqaulpfWl/4ibdwFbRhxs7LQTwCP/oqtxkuJKh4yUugxGpMneO8RTV4DzbdqfFHlgEU2dRAOrelsJinWFs/ieHrO35bg4/vA1denEKyvYqTO7av/AFqqpdX0AX1hCRsh2J+Kat3TZJ/0XsyeAF8f41U8f2jSXCLckEDYb14+4pRKFKGw2FA4kjcj9qKX2V0IqBPtNJr22BFLK8w4pJSNtVSkiEG7P3hfxo6CTHcd96F4KFyrjY/Wipjc6aFfQehRZAEbewpotvxjuYA7U5WNgZ5oW07RE+8cUVGwbqGqbYk+GhHyjmqZ1GwVLuD4h5QVFhSt+xEGtHZYSnnn1qq53aSu1vGwJm3dH/IaCXKFurMOt2QhhLYTwPSj+HGytzTltBUymFQDQeEuf0T86ufLKVZvvShhKcuoKifO6s/QitHaRpTttNUTpekDLNuAOVKI+tX1oAgTNcu2mzo5Ew4SB2oykg77igAO9GjaZNWFACUyYANHCQOY+dCk9iaOExJmjYyCCP7IoQkEHbejaQVc0MGealgYWAQNqOADuKCY2NDoj5UwAODpHEUAKTCTxRtO00IT3ioEKdjsaIZJMAQO/eheX4QC1RpHNQ97mTD7BCru4fbRbAFSnifIAP1En2qJEdEheXTNkwt64cShtA1KUowAK5R61daFuY89ZZduwpCEhoPiFACTOknb4mPgaqXUrrHnDFnLqzxDFXlMKdcCUIVpSUyY2G0RFY47eOOOeLqVJ966ODAsfzl2ZcmRzVRJl3FL68u/GccUtSlyrUed+9WGxxTEbMpDCgGyFakpMeGVDn+tU21d1rHh6vEmCByasKL/ABJNlptVW10gp3GmFiP3q2UnIEY0S7+J3os1XaFgonw1gmY3kg/MAxVYbxAMvOaWxB3JmJE8Ug3mO6t3nSUqaMaVtqEhSfQioy6vG31qcb0oSrcpB2FSLavwGVNULLxpyzudTTxCZ2AO4FbJ0u60XuDYlbh+5uLq3KPDW04uYHbSTuAI44571gN5JX4iBP8AKneEX67RwPBzcGRJqxyU1TK9ji7ifSHAMw4VmTDmsRwq5Q8056GFA9wodj7U9uSSlVch9GOpl9g2PN271wv7lctkLREglIJB+O0fSusrW5Te2aXW1EhaErSSCNSSJB/n9K5+XHsfBrxT3qmZR1gSBc4cs/xB1J/5TVn6YLKssWqIiPEH0War3WEefDQmZKnp/wCSp7pZ/wB3GxpPldcA+BM/1pVTgWSVUXlIITHMUCkhSoP1ijNmfKQYoVJg7fKpFtcCMQcAHAikVcbjanC+IpFe3lmi1YtkHcJIulApMTQBO9LXgJfKSKIhvzAkb9qFOg2mCpGoCRJApZtpKY7zzXkI+G31pRAk7fQ0yBQohIgVXM0W4dLhKQdSSPjKTNWZA0mdjUPmFhK3E6pgxInkbj+ppW7ZGrMBt0gspPY7+9eJCSQlJO+8mlENlpJRIlBKfoaEhavMAnf1q0yt1Kjofpo0UZWw8xy2VceqlVeECAJFU/p80WssYakgA/d0mPjNXBIMSO9cujpZG7FOTAETRgk9yKJ2MH96OnfiraroRCkACQBFeoNZ4ihkjioBgpA596HcySZ3mglQ3j40J3G0/OoA9EyfShVsJmJougzuKNpUJ2p7IBMCJBmh5A9qENg715QjeKhBhit41a2rj1x5G0p8yyP0j1rkPqt1LtLrGb20s2gm2t3SW0hJDbjiTGsoO4nYkV0J1wzgjJmRrvEi4pLzq0MMpTyoqO4ntsD8prgvHcadxm9eu3iEF1WoIQIAHYRWvTQtObRXkkl8fI1xnF3795bq1Tq9TG9RIvAiA4Dv3G9A+lU9j7d6RRauuq2BPtNadq7srX4JZi7tfDK1OIJHG8KSaaP4g4Vfluc9+9ERhL6iITBqUsctPPFKlhQ9vWmuCQVBsh9dw6QQtayTT5GHulA0skqPtV3wbJKFlKlNGPSKvFhkOzW2AG4mBxVEsqj0XQ07l2Ys1lvEbryoaMR9aInB1WLoRiCVobSoa9IBP7mukWsi2tuz50gLCYiN4is+z3lxyy/NDI0q7xtVMdUnLaaJaSo2ikoxxqyubdvDbcoabXr1r3UvYdhwP867gyJfi9y/ha2lKWk4bbqKu3+zER+9cFYoliz8JwLWVavMqfoBXRPRTqjdW1pa4StWsBIbQgnZxAAAAPZQA2AEEDiauyp5YJx8GKLeOfJoHV5JWnDVjbzu8f8ABU30rg5eIBM+Ovn1gVCdTLm2v7PC7i3XqQpbvy2RzU50q1qwZaSnh9cfRNZY8RNcncS+NoBEE715QjcgH3o6GyB5UnevKBCxNMmVDdU9/lSCwI4p2tJPAO1JLQT5oqbvACBvQDdGVDftQI8u8Exx6UpiCf8AWdYOwAoqFhUJ07VE2xHwKIUSAYilkCT6n2ogBjalUz+ruOKZAsWZ1TuJpjjrQ0trmOO096ft7bnimOYlpFmhRO4SQPhqTP8AShLhcEtmBYgx4WJXrCQPy7p5H0WaQAEbGpHH/wDvDigA0pF24rnsST/WoyB6mrOSmlds6VygytrBbBuZKbVsD/yirSkGABUHlxC28NtW1K3Sy2JH+Gp5ERz+1ctSs6M+wUoMkAGjIT5oMbV4HcyO3NHT7VYithtInijaY3oBzzvRzEb9+4puhTwRI2iK9oHO/FeIEyBsO1DEUSAQCd5o0V6DNDUIeA2oFAEQRPtQzA+FEWVaZBM1AnJ32yM3hb2H5YYWpLVuFPLCdgtZlM/LSR865fsG1Xt4lP6oP9a2v7XVs/adSChc+G7aoebIMSFEkmPiTWO5eKRdiAd1Cupje3BEzNbsrJZzLzSkgrEGlrXAW2p8w9tqlXlgRuQYoiLgI3CgY71z3lkzpxxRQpbYOwgCW/33qUt7VlrSCJIqPbvCvzAAmnibgkDYz39qaFyLHFInrK8aYhJ1c7VbMDxBhULW6Bp4k1naXnD5uB7GlG8TeahKgqK0vDuRXvpmsu4mgCAsKneZnmqzmxpnEMPWHEzv61AWWMXSzpKjHvUs/cJubRSVnciufmx+2bMLT7MVxe2QcTcYGkJble/cinmXcTu8Hum3HfIZKm1KAgEcbfSk8zpTheNJuFpBTqCik7gx2qOxrFG777u40mAkEQB3NdLCrijjahKM2jom1xtOP4fb3jD5LQWWlIj9LiUAqI9jNa30qEYO9O4NwqCP8KaxvKGVr/BOnuFXOJoWi4vn13AbVyEKQkJPzCa2XpMUqwl9IMf6wY3/ALo/+KomludDxv21ZoCUbaga8pImD2pRCQoQVcHiOaBY3Ko24FVpWARUNKTG9IKQe4PHypyRPIMUmsGOBTxpAsgMTQEvggEbU3SAkahvHanuLoUXBvECmbY1CTtPYUaT6FdC6FahsRvxNLpGw9uYpuhO0RNOUDywIoU0DgVb+EUzx5CnMNWBsQlQB+YP9KfNgATG9NcYT4lmETufEj3htR/pQkrRE+TBcyaU5lxBAHK0qI+KEn+tMA2SJCR9Kks2IKc03nm3Ulpfy8MCo6U9l1Ykyh/uZ1PgrYbtm0gABKQN+eKlUJJE6qj8MQfCG5gbRUq2nyAVy8btHQn2FSjeAZPxpVKfQUISNXFKpG8bVbYjQmEmRsaMEE/WjETRgDFNyKE0xtsaEmPSj6U9+a9pSYiKNECAcwKMEiYmjkQJBFAdvjUIEWIMCiwPXgUqBq3JFAUwduPhQ5GVHJ322cvXChgGY0pT4X5tko9wowtP/VXMeWWyq9CtJISZ32ruz7U+WTj/AEnv3W0LU9hTiL5Gn0SYV/yqNcSZOtwUPXDgIUlWmD2rfjlen/oVRj/rWTzoKwAAAaNb4a64RuIP7177/h7BIddE+tO7fHMORAS8n2rntvwjqRSYsjBtwNWk04aw8MAhRk+p3pa2xW1UJU4B86LcYjahWpKxV2JsslGIsxaauDPy5qbssqPXiUkBG+59qr7GOWts4StaT3E7VKJ6gYZaN6i6AR6GtW6Xgo2x8k4MmPsp1It3Cn+2Uwk/A0heWht06fCgAb1Cu9YW2UlDCkqJ7nuKKx1Ow/EFabxpA7bGseRZJc0WwljXko3UbDFuJU+B+kelUTBnPGu0MuEqAUIH9K1/MFxh2M27n3VJ/SQeIrMMrYFd4jnG1wS02dubpDSZ7Srk+wrXo538X4MWvguJxO1c7sJt8s4IhsFKQ20hAI3SA0In5VL9Izqwy5O+1yQd/wC4mm3Uxxr/AEew5TbzbifGCdaFApJCFSAfiKd9IYOGXSYiX/8ApG9UqndC18EmaS0PJ615QJJAgjtR20QAJo6kCON6BU27Gi0K0cCfSkiFciIp4pqQdue9N1tFPA39KZJUDsg8YQC4kHfaaYIEAeX51J42gpUhQ5jeo5uYBBn5U6VIV14FGvNMU4QO/pSLciTFLolRHlG3eovwKLJG0Qab4moN2alhE7lIJ5GpCht9acpG+n07+1NMXUn8OXJ2Ckz/APfnUbtUHhcmFZ2QU5rUdyldoyf/AFA/yFRBO+4J+Iqw9RkIbzHaLQTpXZDj2cX/APFQDenTKkhUmZp30kZ5NqTpHWdgk+GklQlQqRR+kQkUwsQQ2BzBqQGw3OwrmRVI6UrvkOn1NGTuTFEERO9H4gUy/IoYCNzRgoe9Fn1ryfcxRToFMMBJKqPpJEGBQJG2x2PJo55o2KF0GInagjUrejykRJr0UQoDSBQKjb09qN3P86LwKg3RX8+4nZ4Pk3GsVvbdD7NrYvOLbX+lY0nyn2PHzr5rJDrTN2bWEt+OQkJMgDsJ+FfRLrLgN7mXplmLB8OcKLl+zUUf3tJCin5hMfOuAbC0cbwEnwvO+4pSQdiOBv8AQ1rwtRxy/qJTlNJFUvHLmJB34O9R7l5dsnUlZ+pqWxKyu0r1pQNzvq/pUTirDTZbXaLcXKfOk/wmrYJSpAkmhW3x++aUCXlfWptjMK3EDxHTNVhduj7sg6FeMD5vT2qTwLC7nFbhNsw0paifTtTZMezkfDlcnQ+vMbddT5HTtVfvcRuH1aQ8rnieatWMZZfwoaXreCQNxxVdFohp4P8Ag6wOQamOO58MfNOl8kI2rrqiAX4j1VUg28824C5Kge4M/wAqjmLNSrwOK0FvVMH0mrbY5cbxa7VctseCFEQ2jYVZkW1cmfH82SWAPPKMsrJSoQQTTa1+94Vm0X1oCh9BK2yOQYO9XbL+UU2/5qiSEid6r2emk4ZiFte26fMUlO1ZITXu0u2bJY2sdvwXvIeecXxm0vctYzePXCmrgXza3FSpJIUhQ+HmTtXQnRiF2F6mZh5M/SuY+nlsq5xC4xMJI/1dCHI/vKBH/prpvopBtcQSAJS83+6T/lQktomWSao1LREelGUmIPpSiUAiSJo2nbeq27KRuZ0wBApIpJJ9eBThXAoigJE0eRX2QOOoA8L5zUQIBgVO48keEgkGoRIGqfSrF0LJUHagGd96XQDMH5UmkUqkxvE1NoErFgflTfEmw9h7iSJlSRHvqAH7mlRI/in1obhBXZPQD+WgufTf+lDbXJDCc/qS5iOGOpTE27jR+S5/rUAUxtqj5VY+oqfDfsEqICUXF0g/EFNVw6TEqExVlquCiVXyddWg/KG8d6doUIAPNN7bZobb+lOEABXAM1y4M6UuGKojgb0cCaKlMdhRwYpxGDpPMUeJEzFFCjwRXgo8UaFD7SAd6EJGqRQhMnmI9aMBHmnioA8OKKOIPPc0aJ3k70ASJ/UN6lDUgIA4Jr2scR86HTsQT8qLp70fBKEblKXW1Nq3SoQduR3FcAZqwMYHjmIYOoaBaXr7UegDhr6BrSCnjauJvtC4avBepeMMtja7cTepMRs6JP8Azav2qyDbtFmN1JGcXOG2VwkgmSR61Xb/AAC3CjoJE+9TCblK/KpUHvSyWmikqmfjUhOUHyzbPFGa4KocEZSkiDv+9W7JeHos7tottAkmSaZos37t6W24Q3yfWrlk1FozcpcuyltIUOT2qZ87aph02nUJWL5rsmboAupAJSBuKo7mXbdJVKQfWOa2fODGFXVgLmx0eIlI7RNZxc2l4iLlTRKJg+9UYMsoOrNOfDDIraKyxgNo09q8EHf0q64C3h9u0Pyo2j0pqGbdbYXEE80ku9t7clDYkRE+9dOc/cjyc7HhUJcFoexW1t7UtMgAkdjWf5rCcYv7RtKAdJWsnvG3+VO3r9Kv0Hf0JpfBMBxHGsQ+8sKDbDAh10nYA9vjWLGtuTcacrThtLn08y/d2uTL/HnmSgP3rDCAoRKQhxX/AFD61ufREgtYkIMhbU/MK/yqGxrDFYd0rsw4wplbt008lKhBCSNKZ9PKAY96kuh7nnxVBJlRYVH/AJ6vvdbOfk4jRsKRsCO+1HiflQIA0j/KjeZI5MUhSwh8s7c0gQfhTmNjvvSKoURTKxSGx4EMJPvvUDtMb78b1YcwpItUqST+qD8IqvJEnftVoJCiAQZ/rSyCSd42pJKpMRSsAHYzQ5EQqN+aUEKtrpskplhcEesUkgDYDeaO2CfFQRsplwfPSal2NRh/U0Su3eMFX314QPcE1VfEUIhHarV1N/7Ey53Teok+gLav6xVVSoaRAnbvTJVHkpycyo7AZHlEdqdNxA2+dNrbgCY2p0O1cuNUbpOnyHBAMHvSg2T+qJ9qKBO0j6UMdppwJ2GH6djPrSg349KIkAD1oRHv8qhGxQQOZrwjuJFByQJkd6EwDzUACTtAFeG0UBMEA17VPajdEDEkiI+NeFFSSoTtXtW8EVLG6AUDFc8/a2yb95wSyzpa26Q9aL+6XKxyptW6CfgqR866GUfSqv1Ly6nNmRcawJQ89zZueF7OJGpB+op8c9skwM+dKifFOx2NPbZTivIeO1N7hhTdwppaVJWkkEH1HP70i7i1ph6D4hBUBxT5IbXwbcWS1YOL4/iGBKLdozKHBzURa5+WFLQ+VNup432qOxzM5vwpppEECJmqx93eccD0FZmSO9Pj06mrkuRcmreN/E0ZzqTiRbQi3dW57TtU1hWd8wY3cs4a5apDa4ClH+H3rJ0t3WuVILaU7wrY1a8s5oOEKS5cJ1gdxyatlpYpXViR105Sps1O5Zdt0AJiCOar9+pwpUUjcHilrbOuG4mjwyShR/tU3vXQQdBG+4NZd0rqSNVxfKY2t0rU4kE7nmut+i3RbLCMr4VmrFrZ+6vLtsXPgOPfkDc6PJG+0cmPauTMIZfvsSYt2hqW6tLaAO6iYH86+i2C4Y3hGCWOFMjSi0tm2APTQkD+lXN7YWjHmm91Iz/rS2E5UcIMBt9kkeg1R/lVc6GuA3OKI07aWCDv6rq1dZ0g5NvJJnWyZ/8A9E1T+hax+IYi3z+Q2efRZ/8AdQxtiPmBuDO/FHUCBHY0RkwBJo5ngd+KDfJUEPFJq0jcjb1rz1y0zGpUkiQlPmUR7Ab1A49nTAMuNh7GsXscObjV/rDwDnybHmP7U8Em+OxXxyOMwJP3NJG41dvgarZISTtualMmdYelGJOOvY7lbOOYrNtJHi4XYkWwPsdlH6/I0zu8RyrjLYxzJWIv3eD3DhbSLlvw7i2dH+6dHrEweDB71pyYMuJKWSLSf2iiGWGVuMZJtfTEWySR2FLiZmkEGNzz2pRLgIgcjmqhhZChPMb0qykl4AnZUj6im6TPbc9qcW5T94SkEfqFI/yPaZhvUJlw4E6QVFTd3bFW/bwwJ/l9aqKI0iSZjsavfUFhbeGYwgJJgMrAJjcOITtWepW1pSSJJHoatjuaooyJKdnZ9vwPWnSB5eKbMalRFOUNnaT3rlxRtb5FAdthvRgduaKODtQRHAFMwIPPvRkwRRANu/0oyNtiKgaFBxFAIP6qAAEzuIr3yqEoNqABAoUkHiZoikzAG1AlWnj5USWHkgkbfOikyTKvkBQFe+9e1niBUJYMTyYoqlgbwDRVrG38qxPqv9p3LGQnHMJwBhvGcTRKVw5DDShtClD9RHoKMYym6j2RtLlnM3XDLy8m9R8awpPla8c3LA//AFueYfzisIxDEHnrpxSlH9R2rROqHVHH+ouZFZhzAGQ54YZQlpsIQhAJIAHJ5O5rNLoIdulBBG5mujCLT+QHJSVJirbzGnU4oD5Udi6Y162lwQZpzaW1m5pDiAT3kVacPwLLNy3FwAhz0inVIOKEsroqrt6h8kuq5HNJ/eWA2UNKnbg1fLfLuUW3RrVKBIOozvUVi9jgjTi1WrKEJPCUimbofJp3BWyos4i60sEKIg7Ve8NxNV1hyFLJmNzVQdsGlu6k7DtUsm4TYWqWkqGoxFDJFSXBVik4OzZvs75dGbOp2FW7iSpmyX99dTHKW9/5mu9VEGRMkc7185ukGYcQytiysetLly3uVjw0rQqClMg7fT511hlXr+u7YQxj+Fh3wwAq4YUApQ9dB2n4EVmyQbVRC5XK2WXrEjXkrERtKPCV8D4qKz3ole21lit+u6eDYNsNzz+sdqR6ndbsKu1pwKysXLxF0A4i21pa1oJTpU6snypKuAOY3NU0Zwfddt7DE87WeB2S1ALtcvW/jOIBG6lOwEg9tpqzS6PPqOILv7JmzY8MPmzfcy9WssZVbBvngFLHkQolK1+4SAVd+4HHNVezzv1c6hMA9PMkYo+1rSlV2pnw2EyoAEqVsEwd/N2n2KGA9FcgMFjMlzfXGIKTcQ+LhS7h9wABQcIOlKd9h8DXYfT7ErfCLJpy1unL0IU02u3JStLKSklYGkxqiTtsNhNdTF6Zjx/9T5SXjpGDJqpS/YqX32cto6W9c8Ut8RxXH83ptGcRQ5c3VngrbaFHSAVBLhhKSAmPLJ24p7gfRrpNgeItpxInFry8tUPpu3nHMQdbWXAkoUkthCVQonUdtomumMSyy5dXeLWdxdq8O3uFqbJRIKVDUDtxtt//AGs5xvpfmG6vrbFsrNPfiNuAVsiB4rJ3KT7R6+tehxfp8Ed2GKX+fbOPlhmzP5zbGGGWdlhuFosrPC1htDilJuHEFKtOrywn9IAAjaZNZNimT7jImaEXKkOowbNClWq1qUNKL1J1tLjgTwe8A+tdD3mF4qhgs3DbQbbQChIB1xyQZ2n4VT+pWEXmbcg3eB2QbTcsgXlsSDrS+0CpMR32jf1rbrcMNXpmyvTxenzJmcWrxfaS4pGlW4UnulQMEfIg0slQEwAAeTFRWDYq3itnb4m1sm/Z8VQ7peBh1JHrq83/AB1I7DYEg+lfPpweOTi/B6KMty3fYslUnYil7ZX56B/fSP3pomJ2VED0pwwSFpJP8QqqTXQ25GT9TEqQjHWZTKUEkxB8ryv/AG1mjKwW0kkzFaf1cR92fzEkHSpNtcSiedJUf61lVmpKrZslZPlG9WYk2V5Ks7cZSQkRxTlsEdqaIuwlJ/KVHxpX783yEqjtXJU0jZQ4HpIrwkx3+FIi/bJjQrejm+Qn/cr5H0plNPglCwBmQTHxoU7+9N14oy3JLKiJoUYu0oSq2UfQ0HkSG2tjgoMhQodKhz8abKxm2EE2zh9hp/zqs5x6nYXla1UhDJfviJbZBEJ917yBHbk+3NPFqTpAaaXJZcQxGywtn7xiN4zbNcBbq9In0HqfhVSvuq2TrJfh/iLrv95tpUfUgCuaM851zRmG7evX7xwrhQT2AT6JHCR7Cs6/0sxRy1Wwu5UVNnua2R0ratszyzJPhHYVz1twtl2GMHu7lscOJcbAI+BIqVy/1Vy/mDW2lNzZOI5TdthAPuCFEGuMMGzu8ysMXzp8NWxI5T71O3mebDCbbVZ3rby/dRJmklgS4DHImaT17+0FiOHXF7lfKt6hu38LwnrhsedRP6glXb0kfWuSrjEnsRuVLdWTySVEk/U1I5vxy4xJ5d2+rzuGdh61XGHYB33NbMOOOOPAk5OQjdpS+tSFyZkzNV99wtLjVuDFTzoCHJjYpMb1A3FuXda0gShXrTt7SRVjpu6KkbEyOCK8jMF1bDSoaj3PFMLZ0Nq0k/Wgf0rMkc0u1XaLFkcSUVmm4cACWt6dMXN1eEOOKqAZQlKwoCQDUqq9DDQSEgewpttje8+mKvXam1kFZEdqdYa2vEbhJJJQkyTNQqVLunpQhRBPpxVwwW1Sw2lASJ5mo/iuQKW6VFmsXFMsKDGym06hHtV/yzjaE29vdKVKQQlceh2P86zy1d8JKyoj9J2qcy8+U4USsbKTIE+1U4vI+oW1I1PJ9ozimOY5gVzYfeXsVwtDVtpPmCmHEqKR33SBxW4dM+hGX0ZSdxLNHTxx3MDz6zbLvFSSgBOkhBUAOeSBzWBZXzKjLWbMEzOpoLTh1y088FfxNKAS4D7QZr6HYc3a4/aMYhc4gy2VhKEEhR0tr4O/CSQOJ7V6n09XhTT64OPmcZ5NslwUDLfSNly2ZTjT6kKWwh15pp0Rq/8AD1DuCTPb+dajkXGWsFwZ7ArrC04dY2alOXN0gBQKE7DcAysxtp5I7U5srCzZty+q8t39PnT4R0pVzKVAyfTjeagM4Y2wMsYvheXXUJuHGCpxOsMoWrVz4hBVISdhBBJiImtc8MMnCX8mX3mrPZ2625Myg1Y32HuvYirEFKt3WVoNutvQNYMLHBSpPtuN96s2Tc+YPmLAW83YDfpbbuV+BpJhbRjdKh7etcnZlytedYMRtcLxLEvutxh7zz6m7ceY69IUkqcPYtqgEHdUiAQK07p5b5Myrki9awh1xIsyV33jJ0vB1AiFJJjUZ29QdtorTPRYdm2Lt/8AgzR1U1JNqkzV8fPgq+/FxLzS1JSoCNyTyCeeaqd67bm7U6w0UqaPmQSNl/yNLYdit87hTNvftL8dxTboCnCpLRj9CZ7CPrQIZadaSu3SC7qHihR3Mkya1YIf6bTJOe52znp3CX8DzrjOXW2Slp90YzhqRwUKURcI3PAEq+SRUmgQonvt71YOuuHvYVYYXnrDmvz8t3aXbgEbuWi4S4kn0nSY+NVt5pFusfd1ardxKXWFhU62lCUfsY+INeL9V07xZd/hnZ0uXctrHAIkb9qVSSlQUFCDtTRJJPt2inCFAkA+tchxNilZnPVtGrFcYZISrxLa6kneQUA7Vj2HLSbNvQdoraOpbSncxuMjdLzKkCP7zSAf5/vWHYesN2jaA0DpAG5jcbH+VPHqkyrNb5O8Tgl8P0BB+fFFGCXqDHkH/FUkvHVMk+I20EJHPijf9qaOZls0qKnrlhKCd9Tg2rhSywXbOjHDOX7UR9yw5ZPJbeACiCoQZ2mm63VfqBMGm2P5ry85dIWrGLVPhoKSA4FHczNI2OJ4dewm1vmne+ytzQ/U44vmRf8AodQ47tjr+g6eeBR5uPakUvrCdIO/xo9w6mSdA+NVbMGbbDBgpttQeudoQncJ/wAX+VWqsjW0zN7P3DPqjmt3KuWHr9m9ZYeLiAAVwtaZhQQO53B/yrEk5jexBBvzcC6YWNRgytJ/r396nMy4i7j1w4rFm03QWCkBxOoJT6D0FZHmRTuS74XeHrJsLgwtpSj+Wfb2rqafDtVeTHlyWy2YzjeDOW5KHEkrBhXaseubhTGIrSgp0EkD09qd3+KpXcG4Qr8t0zHpUHirukouARCSDMVshBx7KJOxe5dKXDqI39KZouFFfm4J232pS4Ul9vxAZ2BEUghKRB1gHmmpeSJsb4p+Y2Sf2qHaIAICt6lMUeQ20oEg1AWy9StSVcmKij9BbFn1EKB1kgbUwWNDigQrSsb0+fSY27bmmdwnxW9QJBAj41HAeMvJG3VqpJKkTBpsVOQAqaetPlxJQs7jak3EFOx3FUuTi6NKgpK0INuL7TTlhC7khDh8tFCSn+GKXttSVCZmmU2K8ZN2Fu00kBAA7EntU5aKA2SqY2qvW9wQkCKlLe7DQBIg/GqZNtl+KKRYWgotKbTBUsaQKs1o193smmu+kJj0NVTK4/Eb9brhhq3AV8VdqsD1+F3BCTKG9hPr2q3FF+SrVTUqSLQ0sXCEoc3SpPhqkTIO3+Vdm/ZuxJrNWRcMDzrwu7RSsKvXUL1KJQoFBIUqANJSe8xFcOW91uEahJ5roX7K+OhzNd/lhGLLsXMXtk31u8kSA+0rQ4iP7yVAz/drv+lyqTgcfVuqkjs3L2F4QiwL9vdh0qa1oW4gA61KAkgq0yO4gckzxVYzAwi5cvmmoSy48rwllIMiT5Se49/fanrT1jgdzbYZYtXV4tCS2suuBpJEkkmJneNttoouK3l7iGFG1ZYRbNoc1BKZ2AAHJ447+ld+UKW++DDb3OJyf1sxnN/TzF7bFrBa7dbjZYL7IlDqkpgLSSI1FKQCnkKTPBovQHPmP5/zU5hWJ6V2l94d3iLulKFKSwoLlRAhRVASJ3Oo+laP1I6jdIsLtbvA874rYYiD5jYtRcKKgOBonSed5EGud8B654B0yxTG09NctKubLE3Qtl3FXCXGUjhACdymSTBM+pNdLRx9zC3GHy+zieo+5HJBb0o3z9/4zvYqNzd60OLBMBJI3SP8tqi8e6gZDyRbHEcwZnsbW6KSCx4oK1b/ANgSTFcF5v8AtI9Vs4Mhu4zK5ZWyhp+7WMMIj08vmPzNUBvELi5ui/cXC3HFnzqUokn61Z+hlVt0IvVIY40o7v7f/TuDMP2jMi485c4E/YXeJYXiDarZ1/w0tJShW2oJVJMc9qz7D8zJwBlnAMZuEAYNcOYb4hO5aJC2FfApnf3rGbHEnl2TTKyooQdSU9p7/XarDmq+uLvCcKzOy4QtxJwq+UQD+Yj/AGCzPfQY/wCGvNesaP3sT9vtHU9I9T9zPtzUl+DoLDLjLt7YFxrFE+Of0gnb5+9E1JOySD7jvXJ1jnjFbN1Vg7cK8ZpYQtIV3Hf4GukMj3r1/lPDb94fmvs6lfUj+leIeLJib3s93qI4PbjLCNOo1spGYLN8CCq2t3T8C2Af3SKpTmV8CfcU6q0SNRJ2JH9asGc7jNOI4oHWrdhy3t0hluEwvQOAfXk1WV4ljVuotKwJ9RHJTx/KhFSowSUX2XfGLvqUbJzEsTwy6YYTuZKRpH+EGR9Kb5cyfnXORDyS5bWp38Z8lKT/AIRyr/7vW42LTGL2q3ry3Q6wTCUqEpVB/endze2tkkIbSltMQEgcCvIbIvln0VesezH2seJKX/H8GdWXQzCGm9WLY3ePuq5DcITPsDJ+tMMc6dXOANG8yvi9wt1o6iy6AVKA7Aj+UVfMSvX7ltSbUyuNoquZ5y9mvDskKzbc4th+H2ijpaS8+fHuVH+FsJSfQmSRt33FNiw/qMixwRRP1TVYIe9mnx9fZQ8V6lYixhAtlJ0XX6XFmAoD4VmCs9tG7Uw6vSrUeTzVxwrIGe834XfZgwTLGI4nh9isIffYYUtKFcx3JMbmJjvWQZ+wVdjcC8ZZU0pPlcSSRpPuDxXr9HoVghsfg8L6hrVq8zyJKKfgvS8bt3myvWCAO9Zn1AxNvELdy2KkqBEADtTO2zG8bXwSo6gNJk8VUbrEXX7txt1Uzx9a1Y8bUrMU5DCwuHFNrsnlg6ZgnmlPE8W2Wwv9QlNN1Nht/wAVJjSYNJqd8LEEgHZxO1aJNiWOrRwqsgCZUJBpndvLQPIeRJpSxWoB1E8LMfOkL5Kikr0x2pFG3yGUq6IO/ulvEJCzv2obVJSkJ7DcUk43qcj3p2ygIASBJiKsXAeWgXDtuYpvCUKgnYml3CBzz6UhpCjClccUKsaLrgjr2zcZeLrUlCjJorZU4Ig1IFakK0LAUn0PeitsW6jqbVpUDumqJK+0XRlQzWytEKCSfSjNqOoSkipNLCVAgQfaaJ9zVOzcme1BRovTTEwsgT2ilmg88QgAn4Us1aSQXEpSPjUnavWdulRaSFrHc0qhzZJTUVwSOFOO2DCWGwSpXEHck1Opt1M24W65Ok6iPVf/AMUwy1al66Fy9EnZI9Pepp51p+58Bkw0ydu8mr49mObbDYdKyVr2STyaveQM3uZLzFhOaWYAwq/bce0/xW6/I4OeNKjVJALcICwRyd4p2wpvV4ThhLwKFfMc1s0+T25qRkzx3RcTpLqF9o/NrWZcSwnJ9hY4db4a+1brxHEX/KC6CEOaB2PIICuB61l2bupGYc1As5s6sYxjD7yFqFhg9upLEjcJ1HTtyTCTt3mq5fPYba2+CZ4zXhIxlp+2cwp+3S+popu7chKVLUNyC2UH33qPV1PxVp8/6IYPh2XkKWFtCxZl5J0xs6uV7gmflXu9H7axRpWec1mTM8jqVJfmv+B8vJOOZgwy3Vb5StcCt2lHxcSxC7LSnSP7RdUJgEbJR796Tcyz07ypdtpzDm1ePkoDhZwNI0BW/lU85A9OEmoa+wvO2YEXuKYmbpS7RPi3Bvn/AA3IjkJcIJ2HYelVZIWtQVIJPG9dOM74XBxsmOL+Td/2X+fyWbNWMZYxPwGcs5b/AAli3Ct13Sn3nZjdxR27bQBzUPbOgKCthU/gnTDO+O234gzgptrMK3ubxxNu0BEyVLI296rhZdt31MqI1oUUmNxI/nxTT5W0o/JoGDLQu3aClSkgE6eY7xNXDALW3xq3xHKC3leHijBValQ3F0152T8SQU7f2qjchdPc65pt2vwTL106CBDjg8Jqf8aoH860rLvQHNNljlq7mPELeyW2fGZbtnNZccQQrRq8oTIB3E1wM62KTl0bNLGWTLH2uWYfnvB7Rm7wXMFi2pH4jbBu6QlOyHkHSo+gOwkV05lyy/DsDw/DOFW1u22fiBv+81mmacCZXiOZMpqs1NAn8awxC0EAOpP5qUg7nhY99q1CzdKm23NgVJBMn1FeB9Q4nt/k+i4JSeNRfFD1DLRnxEJVvRVYbYKUSbYH5UZKoPlJ3MzSulZ9D8a5zLUq7Ll97YwjCW2GVpbaYQEkk/p//tVN7Md1jd+1hmFNpdU6sp8datLaIBJJUfQCq91Zvbc9P8YuUPqTNvKdPrIisX6MZ/ew3G05eDbzrl/ctobWFk6CTH6fjXl1pJZIboPhdnrtNrcWNynONy8HWF7k3P2XMTwfB8VRhVkjFlkpu3b5vyoSJVCJBnt8YFUzqA5jfWXqphnTXJqjcWGHK+6tlsShCh/tnlHiAABPsPWrvk3JWIfaJ6rXF3i1y+nKOXNFs+8gk+IB/ukkd1qnccJE8xXTmKOdFujWHYjn21wrD8GUzaIt3nLRgIcdQCAhsJ2lRJHofXiu7pcWLSNbV8q/uea1+qy613N8X/YsPTrKeE9Oso4dlPCrZDNraNAE6fM6v+JavdRkyfb0rL/tIfZYyB1qy3iF9ZWrGD5nSyV2t8wgJDqwJSh5I/UCdp5E81JYrjWbepmO5SubXDMRwLJ6bdrGr27dAbXdOz+VbbGQBGoyNxFK9ZOo1p04yZiebb+4R4WHsKdQJ/2jp2Qgeskj5TV+CWbHkjKL5ZgyxhNNSXB8XsfZusvY3eYTeiHrR9xh2P7aFFJ/cGq/iCvBuGrhKpS5tNWLOF87jmL3+Lvx413cOXC4/tLUVGPmTVVVNyw5arUQtIlJ967eRKM3tMuJ3BbiQS0HCYMhQmfeo2+U2m9twOyjNN7HE3UEsOKhYMQeRSNy6ty9Z1b9ztzStMtJDD3Epu3QD6EClL1JWkkzIplYOD7+4kj+Ed6k3kpUCPWo19hqyvONEOEE/tSiE6e/FLXTZS7+iKTUSBsRTxpisTfICoI+dNidJCpPwpw64YKSRPwpBUlJ3ig0RBikOJ1A/Tmm67cmYJ1dqBtwNkySDxvS5cbVBVv8KVfTHXPQ3DdwgxJ3pVDl2DAUTNOEFsGUuKHtzS6XEAbrWPhtUpBtiCGLxxUkkD3qWw+wRIWUlZTvtwDSTbluIBSVT6+tODiLiUw0NMDgCpRLZLJv02Q0JUC6sQI7VNYU0lmyDq4U4rzH41T7PVc3IUoEkfSrGq/W20EKITG1CmlwLbRLMJLjupfAouJOKQgFtwpIggimlm84d9VLYh+a15RuRUSYrqycwu5GN5WxvArpQW6ptGK2mqSfFZ8rw5/ibJJ9kGrTh1tm3DsPTjSDl/KWF3xRdMuBTSndCwkKLQhTp9YERuKzvK2LJwPGLLELhBcatHtbqYnUyoFDiY90qNdKdO+iXSO8btsWubp7EUuNvuNsXV0kaG0iQAgQTE9yTXsPStdjeJQyM4eu9PyZZPJiVmRpayK7fqViF9mDOGK+KRoYSWmVgL2IWqXTIngCJqew64zFhuO2uA4HknAMmN4kQym8xJjWtsH+NTrkqT230it5wHBDhuKLcwLDLLCcOwsNut27bSW/vik6UlsGAVklKwQATIT/AGq91dyrl3NWXhi7uGKw69tlK8MJa0lK1beZQHEhJAUO9dN6+MI7kuP8+jmx9OvP7Mnz9/n+SkZL6CWefrReL5xz9iOMPJt1Fhu3XobUUmAkKVqMbgjYbT6VO/Zt6f5ION3TeIYHaOXuHrcZLj5CllROx8x32ETG29UvJWec0dPsLS1b4O9e2zryXkJbQpS0OHycJ3AVP1PetNyjjWX765/GbDBbjCca8dC7lt9S0eadQIHcEj071mxeozyTf/a/pG3Wejwjgaj+9fbNuxvBrTAGBc4U14TOv8watkg7Tv2mqD1CcdtcGN2dTjocbU2lJMr3Bge/7Vozz9nnDCnMP0gpfSttaOdDgUQUkesgbbVj6rR9FylCXmxZWjqvDYJjRKiTH19ar1eojhglPyZ/RtLkzVJ8bSj5/VjV/cZfzI1hVxaDCrjU65cNBJDbp8yVFKjqBI9t6s9nhmH4nhjBwnMAs3UpCEpfaDoMcCZEenerJm3AP9Och45g+BXbCsQftQ5aoU6EypCgrc9hKYntNci2Wd8x5ext3BsRU9a3Nq4W3WXvKttST3+leN9awTk4zx8Kj3fpKwZlJZOZWbdeZlv8sY0MLzEhpTbkFFwynyqT6ge3cVbm7hLiA4hYKVAKSRwQRzXPmeepLuOJtC40Qu2VrLhX7cCr5076h4Xc5Xtxij/hvMqU0BqG6Rwf3riQg2vkX6nDHHL49Fk6lYba5gSjKrWY8PwuwxMhSrl92Esb7pUBJgnisixbL+V+lGbGbfLmakZgxktJi8tQBbtKcGyUckqI31HieJmidQVKXqUtRUQ4ACTJj0qi5EQh3P2EtOoStCrxEpUJB39KX0vTJYFud2rDr87jlcYcc0fR/wCz9n/ArDpLhWGYE0mzWoKNx50+I++VQpZAJO52AO8Csm6l5ye6vdWrPJlleF3L2VXhe4u4FflOv/wt+hg+X6+lQt/bW1rmXDF21u0yoJUQW0BJEIWRx6EVn/2eHHFnMry3FKcWpoqWTJUYUZJ77k1v0+L2pSl2/wD3wcnPLc1HxZ2Zin2m8L6eZNvMUzCpFxb2zPhsWbij+asiENjkCTtMbCfSuFuuH2ic6dYMGtsHxV63tbK2fXceDbApC1KmAdzISDAp19qh10WGCIDqwk3T5I1GDCER/M/U1hCFqNtuo/p9fahDDHAlkj2De8qcWQV6tSVFPPcmoRbyUvlSCAQeKmrgkhUmdqruI7P7bb1sjJzdsSkNsRAYum7hJgOGVfGjIUlbwcVOySfrQ4t/2IH0iKaWRJJk/wANM+exq8DuxeIxRSRq/T2qemR3J96rWGf/AJZX+E1ZU7mqW+aC1SI+8QSsKgnftTJxvSDuZqUuNwuexFR/enTEGhJnaKKsgmPaTQ/7xVe9auXRBosFZmaOgHad96FQEExvPNEk6hvVe6xouhwltW/m54pVCSBvv7etJo/QmlU/w/GhSGsXbJIMAdqUT5iDsI/eiQIB9hSqOSPaiSx5ZuaFEoHA4p0FrcIC5M0wY2Soj1p9ak6k7mh2xWTViFggKVJNSKktrRKiNttjTCz/ANofhUioALAAEVJqgRV9jB1jwzqgkDY+wNaH0/zViTWDt2zFm6Llh9Nsm8VBbQhBkgDusjw/kT61RlbodB32q69PwDlDGZH6cRtSPYllckfQfQVdgdGrTz9vJtXk6a6b59wLNeDpubp95FzYv6WHnUagVJPJUBsofLYjmntn1AuL7Gb3L+Y8FNwziTziLdjSCQyORIESoaiJIO0iuRul2JYjZP44qyv7i3IvWiC06pEHxFjsfQCt8ydd3acVeuU3Toe+6uOeIFnVq0rGqeZ966vuvFJR+zPkwR1G6T4aZZE5PwfJeaMQurO2xfFLZ62BtHrdYKrVaSCQ80SFbECFwI5q24lmHDsRxuzxjw7C3aw+xce8cNea4cUiCNXz2G9Y5g+PY5cYAlm4xm+dbU8+lSV3C1AjUdiCeKlbny4chSdim1cgjtsiupgybY7Iqkzha7Tbpb5ytm34a01gtzimMu3QYs28ZUtpKl6fFW62hJH/AJlyJrHOqb+YMs4urMVnej8Nu7nXqAnwFLMmdogkmDxUn9oO5uW/9E2W7hxLa7vWpKVkBSgBBI7n3pnm9xx7pDm3xXFL02KtOozG3asOqnvXtteToaLB7UY6hP8Adw0DlHqZbs5Scy/Y5ZurPGvvin04stzUhJUFEhtIjUlX6VDVEkVzZ9ojH8Bzh1CYxvJoSxcPWrbd8Wh+W+8kD8wDsoglJHqk1RrjGcYOHNsHFbzw9UaPHVpjbaJqJw8lOI2ygSD44MjmdVLh/wB5ik8vhF+thH09qWDht2L2+LXqb1WH35cQts6XErBBSQe9adlPCsYxfCvHwgKUw26WpE/qABP8xUH1Es7NfVBSV2rKg6UlwFAIWdKdz61v+QGGbLLyGbNlDDfiE6GkhImBvArzUoRhbSOm8spxTZ//2Q==";
//     ipcRenderer.send('detect_object',{n:'od',a:base64_img});
// }

// async function vad() {
//     const durationInSeconds = 15; // Duration of the recording in seconds
//     await recordAndSaveAudio(durationInSeconds);
//     let wav_file_path = "D:/Demo/CVD_VAD/addon/recorded_audio.wav";
//     ipcRenderer.send('process_voice_sample', { n: 'VAD', a: wav_file_path });

// }


// ipcRenderer.on('result', (event, result) => {
//     console.log('html', event, result);
//     document.getElementById('result').innerText = `Result: ${result}`;
// });
document.getElementById('save-pdf').addEventListener('click', async () => {
  const filePath = output_file_name+'.pdf'; // Replace with desired path
  const result = await window.electronAPI.saveAsPDF(filePath);

  if (result.success) {
      alert('PDF saved successfully!');
  } else {
      console.error('Failed to save PDF:', result.error);
      alert('Failed to save PDF. Check the console for details.');
  }
  document.getElementById('save-pdf').disabled=true;
});
const CV_RESULT = {
  'Mobile Camera':0,
   'Suspicious Mobile Activity':0,
   'Multiple people':0,
   'Suspicious Background Activity':0,
   'Left Session':0,
   'FSLA':0,
   'Left Screen':0,
  'Long Multiple people':0,
   'Long Left Session':0,
   'Long Left Screen':0,
   'Camera Block':0,
   'Imposter':0
  //'1000': 'Nothing',
  //'-1000': 'Exception',
  //'8': 'Normal Session After checking Imposter and FSLA',
};
window.electronAPI.onAnalysisResult((event, res) => {
 //console.log('Analysis result:', res); // Process the result as needed
  
  if(res.n==='init'){
    let result=res.result;
    document.getElementById('result').innerText = `Result: ${result}`;
  }
  if(res.n==='decrypt'){
    let result=res.result;
    document.getElementById('result').innerText = `Result: ${result}`;
  }
  if(res.n==='setBaselineImg'){
    let result=res.result;
    document.getElementById('folderPath').innerText = `folderPath: ${result}`;
  }
  if(res.n==='od'){
    //let result=res.result;
    

    let violation=res.violation;
    let frame_count=res.frame_count;
    let frameBase64=res.frameBase64;
    //let imp_result=res.imp_result;
   // console.log(imp_result);
    if(violation){
      if (violation in CV_RESULT){
        CV_RESULT[violation]=CV_RESULT[violation]+1;
      }
      // Create a wrapper div for the image and the frame count label
      const resultWrapper = document.createElement('div');
      resultWrapper.className = 'result-wrapper';
      // Create the frame count label
      const frameLabel = document.createElement('p');
      frameLabel.textContent = `Frame: ${frame_count}`;
      frameLabel.className = 'frame-label';

      const resultLabel = document.createElement('p');
      resultLabel.textContent = `Violaton: ${violation}`;
      resultLabel.className = 'result-label';


      const img = document.createElement('img');
      img.src = `data:image/jpeg;base64,${frameBase64}`;
      img.className = 'analysis-image';
      
      // Append the label and image to the wrapper, and add to the analysis column
      resultWrapper.appendChild(frameLabel);
      resultWrapper.appendChild(resultLabel);
    //   if(frame_count % 7 === 0){
    //     if(imp_result === 5){
    //       const imp_resultLabel = document.createElement('p');
    //       imp_resultLabel.textContent = `Imp_Result: ${imp_result}`;
    //       imp_resultLabel.className = 'imp-result-label';
    //       resultWrapper.appendChild(imp_resultLabel);
    //     }
        
    //     }
     resultWrapper.appendChild(img);
     analysisColumn.appendChild(resultWrapper);
     document.getElementById('result').innerText = `Result: ${JSON.stringify(CV_RESULT)}`;
    //   document.getElementById('imp-result').innerText = `Imp_Result: ${imp_result}`;
     }
    
    //analysisColumn.appendChild(img);

    //document.getElementById('result').innerText = `Result: ${result}`;
  } 
  
 
});


