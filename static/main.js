$('#result').hide();
$('form#detect-form').on("submit", function (ev) {
    ev.preventDefault();
    var vcfData = new FormData($('form#detect-form')[0]);
    $.ajax({
        url: '/api/detect',
        type: 'post',
        dataType: 'json',
        processData: false,
        contentType: false,
        cache : false,
        data : vcfData,
        success: function(data) {
          $('#result').show();
          $('#line-img').attr("src", data.lines_image);
          $('#mask-img').attr("src", data.mask_image);
        }
    });
});
