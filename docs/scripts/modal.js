/**
 * Script for loading/initializing modal
 */

// Populate modal information
var _modal = new bootstrap.Modal(document.getElementById("modal"));
var _modal_title = document.getElementsByClassName("modal-title");
var _modal_image = document.getElementsByClassName("modal-image");
var _images = document.querySelectorAll("figure img");
for (var i = 0; i < _images.length; i++) {

    (function(index) {
        var title = _images[index].parentNode.children[1].innerHTML;
        var image = _images[index].src;
        _images[index].addEventListener("click", function() {
            for (var j = 0; j < _modal_title.length; j++) {
                _modal_title[j].innerHTML = title;
            }

            _modal_image[0].src = image;
            _modal.show();
        })
    })(i)

}