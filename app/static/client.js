var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function removal() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select an image to style!");

  el("removal-button").innerHTML = "processing..";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/removal`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.responseType = "blob";

  xhr.onload = function(e) {
    if (this.readyState === 4) {
      const blobUrl = URL.createObjectURL(e.target.response);
      el("image-picked").src = blobUrl;
    }
    el("removal-button").innerHTML = "remove glasses";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}