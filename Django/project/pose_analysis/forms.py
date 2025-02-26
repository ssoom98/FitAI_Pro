from django import forms
from django.core.validators import FileExtensionValidator

class VideoUploadForm(forms.Form):
    video = forms.FileField(
        label="동영상 파일 선택",
        help_text="MP4 파일만 업로드 해주세요.",
        validators=[FileExtensionValidator(allowed_extensions=['mp4'])]
    )