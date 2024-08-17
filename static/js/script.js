document.addEventListener('DOMContentLoaded', (event) => {
    const fileInput = document.getElementById('file');
    const fileLabel = document.querySelector('.file-upload label');

    fileInput.addEventListener('change', function(e) {
        let fileName = e.target.value.split('\\').pop();
        if (fileName)
            fileLabel.innerHTML = fileName;
        else
            fileLabel.innerHTML = '이미지 업로드';
    });
});