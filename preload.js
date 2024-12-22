// // preload.js
// const { MicVAD,utils } = require('@ricky0123/vad-web');
// //const { interpolateInferno}=require('d3-scale-chromatic');
// //import { MicVAD,utils } from '@ricky0123/vad-web';
// //import interpolateInferno from'd3-scale-chromatic';
// window.MicVAD = MicVAD;
// window.utils=utils;
// //window.interpolateInferno=interpolateInferno;
const { ipcRenderer, contextBridge } = require('electron');
const fs = require('fs');
const path = require('path');
// Expose API to send frame for analysis
contextBridge.exposeInMainWorld('electronAPI', {
    analyzeFrame: (frameBase64) => ipcRenderer.send('analyze-frame', frameBase64),
    decryptModel: () => ipcRenderer.send('decrypt'),
    initAddon: () => ipcRenderer.send('init'),
    startImgAnalysis: () => ipcRenderer.send('startImgAnalysis'),
    selectFolder:() => ipcRenderer.send('open-folder-dialog'),
    selectImgFolder: () => ipcRenderer.invoke('select-folder'),
    //startCVDAddon: () => ipcRenderer.send('od'),
    onAnalysisResult: (callback) => ipcRenderer.on('analysis-result', callback),
    readDirectory: (folderPath) => fs.readdirSync(folderPath),
    readFile: (filePath) => fs.readFileSync(filePath),
    fileToBase64: (filePath) => {
        try {
            const fileBuffer = fs.readFileSync(filePath); // Read file as a Buffer
            return fileBuffer.toString('base64'); // Convert Buffer to Base64
        } catch (error) {
            console.error('Error reading file:', error);
            return null;
        }
    },
    saveAsPDF: (filePath) => ipcRenderer.invoke('save-page-as-pdf', filePath)
});
