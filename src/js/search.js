const fs = require("fs");
function search() {
    bookList = fs.readFileSync(__dirname + "/json/bookList.json");
    // fs获取json
    bookList = JSON.parse(bookList.toString());
    //写一个查询算法
    var book = bookList[0];
    console.log(book)

    var i = 0;
    while (/*要true里面才能执行*/book.finishi) {
        book = bookList[i];
        i++
    }
    return book;
}