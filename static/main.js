$(document).on({
    ajaxStart: function(){
        $("body").addClass("loading"); 
    },
    ajaxStop: function(){ 
        $("body").removeClass("loading"); 
    }    
});

function readURL(input) {
    if (input.files && input.files[0]) {
  
        var reader = new FileReader();
    
        reader.onload = function(e) {
            $('.image-upload-wrap').hide();
    
            $('.file-upload-image').attr('src', e.target.result);
            $('.file-upload-content').show();
    
            // $('.image-title').html(input.files[0].name);
        };
    
        reader.readAsDataURL(input.files[0]);
  
    } else {
        removeUpload();
    }
}
  
function removeUpload() {
    $('.file-upload-input').replaceWith($('.file-upload-input').clone());
    $('.file-upload-content').hide();
    $('.image-upload-wrap').show();
}

$('.image-upload-wrap').bind('dragover', function () {
    $('.image-upload-wrap').addClass('image-dropping');
});

$('.image-upload-wrap').bind('dragleave', function () {
    $('.image-upload-wrap').removeClass('image-dropping');
});

// $("#form").submit(function(){
//     var reader = new FileReader();
//     var url;
//     reader.onload = function(event){
//         url = event.target.result;
//         $('#destination').html("<img src='"+url+"' />");

//         //save
//         $.ajax('/solve', {
//             cache: false,
//             method: 'POST',
//             data: {url}
//         });
//     }

//     //when the file is read it triggers the onload event above.
//     reader.readAsDataURL(e.target.files[0]);
// });

function upload_files() {
    var files = $('#file')[0].files;
    imagebox = $('#imagebox');
    
    let formData = new FormData();

    for(let i=0; i<files.length; i++) {
        formData.append("file", files[i]);
    }

    $.ajax({
        url: 'solve',
        type: 'POST',
        data: formData,
        cache: false,
        processData: false,
        contentType: false,
        error: function(data){
            console.log("upload error" , data);
            console.log(data.getAllResponseHeaders());
        },
        success: function(data){

            $('#solve_section').attr("style", "display:block")
            
            bytestring = data['solved']
            image = bytestring.split('\'')[1]
            imagebox.attr('src' , 'data:image/jpeg;base64,'+image)

            bytestring = data['processed']
            image = bytestring.split('\'')[1]
            $('#processed_image').attr('src' , 'data:image/jpeg;base64,'+image)
        }
    });

    // xhr.onreadystatechange = function() {
    //     if(xhr.readyState === XMLHttpRequest.DONE) {
    //         alert(xhr.responseText);
    //     }

    //     console.log(xhr.response);
    // }

    // console.log("Let's upload files: ", formData);
    // xhr.open('POST', 'upload_handler.py', true); // async = true
    // xhr.send(formData); 
}