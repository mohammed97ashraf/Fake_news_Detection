function myFunction() {
    var para = document.createElement("P");
    var t = document.createTextNode("This is a paragraph.");
    para.appendChild(t);
    document.getElementById("myDIV").appendChild(para);
  }