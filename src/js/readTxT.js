const fs = require("fs");
function readtxt() {
    txtValue = fs.readFileSync(__dirname + "/pythonServer/reading.txt").toString();
    if (txtValue == "reading") {
        return true
    } else {
        return false
    }
}

