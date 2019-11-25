import os, sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from IPython.display import display, clear_output

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)

def load_img(path, img_max_dim=512):
    # load image from two paths
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    max_dim = max(shape)
    scale = img_max_dim / max_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def load_img2(img):
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    img = img[tf.newaxis, :]
    return img

class StyleTransfer:
    def __init__(self,
            content_img_path, style_img_path,
            content_layers=["block5_conv2"],
            style_layers=["block1_conv1",
                          "block2_conv1",
                          "block3_conv1",
                          "block4_conv1",
                          "block5_conv1"],
            video=False):
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        if not video:
            self.content_img = load_img(content_img_path)
        else:
            self.content_img = load_img2(content_img_path)
        self.style_img = load_img(style_img_path)
        self.img = tf.Variable(self.content_img)
        self.load_layers()

    def load_layers(self):
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in (self.style_layers + self.content_layers)]
        self.vgg = tf.keras.Model([vgg.input], outputs)
        self.vgg.trainable = False
        content_target = self.content_img * 255.0
        style_target = self.style_img * 255.0
        content_target_prep = tf.keras.applications.vgg19.preprocess_input(content_target)
        style_target_prep = tf.keras.applications.vgg19.preprocess_input(style_target)
        content_target = self.vgg(content_target_prep)[self.num_style_layers:]
        self.content_target = {content_name:value for content_name, value in zip(self.content_layers, content_target)}
        style_target = self.vgg(style_target_prep)[:self.num_style_layers]
        style_target = [self.gram_matrix(output) for output in style_target]
        self.style_target = {style_name:value for style_name, value in zip(self.style_layers, style_target)}

    def display_tensor(self, tensor):
        tensor = tensor * 255.0
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def get_frame(self, tensor):
        tensor = tensor * 255.0
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return tensor

    def gram_matrix(self, tensor):
        result = tf.linalg.einsum("bijc,bijd->bcd", tensor, tensor)
        shape = tf.shape(tensor)
        num = tf.cast(shape[1]*shape[2], tf.float32)
        return (result / num)

    def calc_loss(self, style_weight=1e-2, content_weight=1e4):
        img = self.img * 255.0
        img_prep = tf.keras.applications.vgg19.preprocess_input(img)
        outputs = self.vgg(img_prep)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [self.gram_matrix(val) for val in style_outputs]
        style_outputs = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        content_outputs = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_target[name])**2) for name in style_outputs.keys()])
        style_loss *= style_weight / self.num_style_layers
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_target[name])**2) for name in content_outputs.keys()])
        content_loss *= content_weight / self.num_content_layers
        return (style_loss + content_loss)

    def train(self, opt, style_weight, content_weight, denoise, denoise_weight):
        with tf.GradientTape() as tape:
            loss = self.calc_loss(style_weight, content_weight)
            if denoise:
                loss += denoise_weight * tf.image.total_variation(self.img)
        grad = tape.gradient(loss, self.img)
        opt.apply_gradients([(grad, self.img)])
        self.img.assign(tf.clip_by_value(self.img, clip_value_min=0.0, clip_value_max=1.0))

    def transfer(self, opt, epochs=10, step_per_epoch=100, style_weight=1e-2, content_weight=1e4, denoise=True, denoise_weight=30):
        start = time.time()
        step = 0
        for n in range(epochs):
            for m in range(step_per_epoch):
                step += 1
                self.train(opt, style_weight, content_weight, denoise, denoise_weight)
                print(".", end="")
            clear_output(wait=True)
            display(self.display_tensor(self.img))
            print("Train Step: {}".format(step))
        print("Total Time: {:.1f}".format(time.time() - start))

    def savefig(self, path):
        self.display_tensor(self.img).save(path)