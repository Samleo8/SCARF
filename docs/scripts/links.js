/**
 * Script that takes care of figures, links and galleries
 */

// Populate figure IDs
var _figures = document.getElementsByTagName("figure");

var id = 1;
for (var i = 0; i < _figures.length; i++) {
    var fig = _figures[i];

    // Ignore figures that are part of a gallery
    if (!fig.parentNode.className.startsWith("gallery-item")) {
        fig.id = "fig-" + id;
        id++;
    }
}

// Populate gallery ids
var _galleries = document.getElementsByClassName("gallery");
for (var i = 0; i < _galleries.length; i++) {
    var gal = _galleries[i];
    var id = gal.getAttribute("fignum");
    gal.id = "fig-" + id;
}

// Populate references to images
var _figrefs = document.getElementsByClassName("figref")
for (var i = 0; i < _figrefs.length; i++) {
    var _figref = _figrefs[i];

    var figId = _figref.innerHTML.replace(/[A-Za-z]+/g, '');

    _figref.href = "#fig-" + figId;
}
