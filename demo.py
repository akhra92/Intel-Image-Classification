import torch
import os
from dataloader import get_transforms
from PIL import Image
import timm
import streamlit as st


DEFAULT_CHECKPOINT = 'saved_model/best_model_final.pth'
CLS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
DEFAULT_IMAGE = 'sample_images/475.jpg'
SAMPLE_PATH = 'assets/samples.png'


def run():
    tfs = get_transforms(train=False)
    model = load_model(len(CLS_NAMES), DEFAULT_CHECKPOINT)
    st.title('Intel Image Classification')
    file = st.file_uploader(f'Please upload your image from this category: {CLS_NAMES}')
    st.write('Sample Images: ')
    st.image(SAMPLE_PATH)

    if file:
        im, out = predict(m=model, path=file, tfs=tfs, cls_names=CLS_NAMES)
        st.write('Input Image: ')
        st.image(im)
        st.write(f'Predicted as {out}')
    elif os.path.exists(DEFAULT_IMAGE):
        im, out = predict(m=model, path=DEFAULT_IMAGE, tfs=tfs, cls_names=CLS_NAMES)
        st.write('Input Image: ')
        st.image(im)
        st.write(f'Predicted as {out}')
    else:
        st.error(f'Default image not found at "{DEFAULT_IMAGE}". Please upload an image to continue.')


@st.cache_resource
def load_model(num_classes, checkpoint_path):
    m = timm.create_model(model_name='resnext101_32x8d', pretrained=False, num_classes=num_classes)
    m.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    return m.eval()


@torch.no_grad()
def predict(m, path, tfs, cls_names):
    im = Image.open(path)

    return im, cls_names[int(torch.max(m(tfs(im).unsqueeze(0)).data, 1)[1])]


run()