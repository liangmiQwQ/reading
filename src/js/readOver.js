const fs = require("fs");
const child = require("child_process");
function finishiRead() {
    var bookList = fs.readFileSync(__dirname + "/json/bookList.json").toString();
    bookList = JSON.parse(bookList);
    var i = 0;
    console.log(bookList)
    while (bookList[i].finishi) {
        // 如果是true就一直执行
        i++;
    }
    bookList[i].finishi = true;
    fs.writeFileSync(__dirname + "/json/bookList.json", JSON.stringify(bookList));
    console.log(__dirname + "/json/bookList.json");
    console.log(JSON.stringify(bookList))
    child.exec("killall Python", (e) => {
        console.log(e);
    });
}



