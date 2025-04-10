$('.arrow').on('click', function(e) {
    e.preventDefault;
    $(this).toggleClass('arrow_active');
});
// Sidebar 
let counter = 0;
$('.navBarActivateBtn').on('click', function(e) {
    e.preventDefault;
    $(this).toggleClass('navBarActivateBtn_active');
    $('#pageWrapper').toggleClass('content_activeSlider');
    counter += 1;
    if (counter % 2 != 0) {
        $('#pageWrapper').animate({ scrollTop: 2000 }, 0);
    }}
);
// Dropzone
const { Dropzone } = require("dropzone");
Dropzone.autoDiscover = false;
let myDropzone = new Dropzone("#my-dropzone", {
    url: "upload/",
    maxFiles: 3,
    maxFilesize: 2,
    acceptedFiles: '.jpg',
});
myDropzone.on("addedfile", file => {
    console.log(`File added: ${file.name}`);
});