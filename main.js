const { app, BrowserWindow, Menu, ipcMain } = require('electron');
const moment = require("moment")

const findermenu = [
    {
        label: "操作",
        submenu: [
            {
                label: "请勿尝试切换窗口",
                accelerator: 'Cmd+Tab'
            }
        ]
    }
];
const menu = Menu.buildFromTemplate(findermenu);
Menu.setApplicationMenu(menu);
//通过将数组菜单设置为空数组，让顶部的finder菜单消失

function createWindow() {
    const win = new BrowserWindow({
        height: 600,
        width: 800,
        fullscreen: true,
        frame: true,
        //通过全屏选项和框架的消失，让用户无法关闭这个窗口
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
    });

    win.setTitle('阅读器');
    win.loadFile('./src/first.html');
    win.setFullScreen(true);
    //设置全屏
}

setInterval(() => {

    var hour = moment(new Date()).format("HH");
    var mins = moment(new Date()).format("mm");
    if (hour == 19 && mins == 20) {
        app.whenReady().then(createWindow);
    }

}, 60000)

// app.whenReady().then(createWindow);
//启动窗口

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
