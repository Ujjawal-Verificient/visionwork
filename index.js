//import { app, BrowserWindow, ipcMain } from 'electron';
const { app, BrowserWindow, ipcMain } = require('electron');

//import {path} from 'path';
const path =require('path');
let mainWindow;
// app.whenReady().then(() => {
//     mainWindow = new BrowserWindow({
//         width: 800,
//         height: 600,
//         webPreferences: {
//            // preload: path.join(__dirname, 'preload.js'),
//             contextIsolation: true,
//         },
//     });

//     mainWindow.loadFile('index.html');

//     mainWindow.on('closed', () => {
//         // Dereference the window object
//         mainWindow = null;
//     });
// });
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
            enableRemoteModule: false,
            sandbox: false, 
           // webSecurity: false
        }
    });

    mainWindow.loadFile('index.html');
    
    mainWindow.on('closed', function() {
        mainWindow = null;
    });
}
app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow(); // Recreate the window on macOS when the dock icon is clicked
    }
});
app.on('ready', () => {
    console.log('ready....');
    createWindow();
    //console.log('ready addon_process....', mainWindow.id);
    require('./addon_process');
    //console.log('ready addon_process....', mainWindow.id);
    //mainWindow.webContents.send('setMainWindow', mainWindow.id);
    // setTimeout(() => {
    //     console.log('ready addon_process....', mainWindow.id);
    //     mainWindow.webContents.send('setMainWindow', mainWindow.id);
    // }, 5000); // Delay sending message by 1 second
    
});

app.on('window-all-closed', function() {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', function() {
    if (mainWindow === null) {
        createWindow();
    }
});

// Camera block= -1
        //Mobile Camera = 0
        //Suspicious Mobile Activity =1
        //Multiple people = 2
        //Long Multiple people = 22
        //Suspicious background activity = 3
        //Left Session = 4
        // Long Left Session=44
        //Imposter = 5
        //FSLA=6
        //Face Not Detected=7
        //Noramal Session After checking Imposter and FSLA=8
        //Left Screen =9
        //Long Left Screen=99
        // camera Flash=10
// ipcMain.on('calculate', (event, { a, b }) => {
//     console.log('ipcMain calculate with event',event,a,b);
//     mainWindow.webContents.send('calculate', { a, b });
// });

// ipcMain.on('result', (event, result) => {
//     console.log('result------------->',result);
//     mainWindow.webContents.executeJavaScript(`document.getElementById('result').innerText = ${result}`);
// });
// Handle folder selection request from renderer
