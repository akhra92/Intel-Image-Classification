import torch
import argparse
from dataloader import get_transforms
from PIL import Image
import timm
import streamlit as st
import config as cfg


def run(args):

    cls_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    num_classes = len(cls_names)
    tfs = get_transforms(train=False)
    default_path = './sample_images/image_0.jpg'

    model = load_model(num_classes, args.checkpoint_path)
    st.title('Intel Image Classification')
    file = st.file_uploader(f'Please upload your image from this category: {cls_names}')

    im, out = predict(m=model, path=file, tfs=tfs, cls_names=cls_names) if file else predict(m=model, path=default_path, tfs=tfs, cls_names=cls_names)
    st.write('Input Image: ')
    st.image(im)
    st.write(f'Predicted as {out}')


def load_model(num_classes, checkpoint_path):
    m = timm.create_model(model_name='resnext101_32x8d', pretrained=True, num_classes=num_classes)
    m.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    return m.eval()


def predict(m, path, tfs, cls_names):
    im = Image.open(path)
    im.save(path)

    return im, cls_names[int(torch.max(m(tfs(im).unsqueeze(0)).data, 1)[1])]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Intel Image Classification Demo')
    parser.add_argument('-cp', '--checkpoint_path', type=str, default='./saved_models/best_model.pth', help='Path to the checkpoint')

    args = parser.parse_args()

    run(args)