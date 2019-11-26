from StyleTransfer import *
import pickle

content_image_path = "stretching-white-cat-979247.jpg"
style_image_path = "blue-and-red-abstract-painting-1799912.jpg"

content_weights = [2*(10**x) for x in range(-2, 2)]
style_weights = [2*(10**x) for x in range(-2, 2)]

images = []

for i in range(len(content_weights)):
    for j in range(len(style_weights)):
        tf.keras.backend.clear_session()
        model = StyleTransfer(content_image_path, style_image_path)
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        model.transfer(opt, style_weight=style_weights[j], content_weight=content_weights[i])
        images.append(model.display_tensor(model.img))

with open("images.pickle", "wb") as outFile:
    pickle.dump(images, outFile)
