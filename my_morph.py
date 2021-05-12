import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import os
import shutil
import argparse


# Parameters choice
binary_images = True
image_size = (768, 1024)
map_size = (768, 1024)
integer_to_float_scaling = 1
integer_to_float_bias = 0
binary_threshold = 0.95  #in scale [0,1]
value_if_greater_than_threshold = 1
video_fps = 48
video_format = 'mp4v'
images_to_generate = 100
TRAIN_EPOCHS = 100
warp_scale = 0.05
mult_scale = 0.0
add_scale = 0.0
add_first = True
targets_loss_weight = 1
origins_loss_weight = 2
maps_loss_weight = 1
scale_images_for_video = True
video_scale_ratio = 255

height = image_size[0]
width = image_size[1]
map_height = map_size[0]
map_width = map_size[1]

@tf.function
def warp(origins, targets, preds_org, preds_trg):
    if add_first:
        if binary_images:
            res_targets = tfa.image.dense_image_warp(
                (origins + preds_org[:, :, :, 1:2] * 2 * add_scale) * tf.maximum(0.1,
                                                                                 1 + preds_org[
                                                                                     :,
                                                                                     :,
                                                                                     :,
                                                                                     0:1] * mult_scale),
                preds_org[:, :, :, 2:4] * height * warp_scale)
            res_origins = tfa.image.dense_image_warp(
                (targets + preds_trg[:, :, :, 1:2] * 2 * add_scale) * tf.maximum(0.1,
                                                                                 1 + preds_trg[
                                                                                     :,
                                                                                     :,
                                                                                     :,
                                                                                     0:1] * mult_scale),
                preds_trg[:, :, :, 2:4] * height * warp_scale)
        else:
            res_targets = tfa.image.dense_image_warp((origins + preds_org[:, :, :, 3:6] * 2 * add_scale) * tf.maximum(0.1,
                                                                                                                      1 + preds_org[
                                                                                                                          :,
                                                                                                                          :,
                                                                                                                          :,
                                                                                                                          0:3] * mult_scale),
                                                     preds_org[:, :, :, 6:8] * height * warp_scale)
            res_origins = tfa.image.dense_image_warp((targets + preds_trg[:, :, :, 3:6] * 2 * add_scale) * tf.maximum(0.1,
                                                                                                                      1 + preds_trg[
                                                                                                                          :,
                                                                                                                          :,
                                                                                                                          :,
                                                                                                                          0:3] * mult_scale),
                                                     preds_trg[:, :, :, 6:8] * height * warp_scale)
    else:
        if binary_images:
            res_targets = tfa.image.dense_image_warp(
                origins * tf.maximum(0.1, 1 + preds_org[:, :, :, 0:1] * mult_scale) + preds_org[:, :, :,
                                                                                      1:2] * 2 * add_scale,
                preds_org[:, :, :, 2:4] * height * warp_scale)
            res_origins = tfa.image.dense_image_warp(
                targets * tf.maximum(0.1, 1 + preds_trg[:, :, :, 0:1] * mult_scale) + preds_trg[:, :, :,
                                                                                      1:2] * 2 * add_scale,
                preds_trg[:, :, :, 2:4] * height * warp_scale)
        else:
            res_targets = tfa.image.dense_image_warp(
                origins * tf.maximum(0.1, 1 + preds_org[:, :, :, 0:3] * mult_scale) + preds_org[:, :, :,
                                                                                      3:6] * 2 * add_scale,
                preds_org[:, :, :, 6:8] * height * warp_scale)
            res_origins = tfa.image.dense_image_warp(
                targets * tf.maximum(0.1, 1 + preds_trg[:, :, :, 0:3] * mult_scale) + preds_trg[:, :, :,
                                                                                      3:6] * 2 * add_scale,
                preds_trg[:, :, :, 6:8] * height * warp_scale)

    return res_targets, res_origins


def create_grid(height, width):
    # Create a grid of the dimensions of the images
    grid = np.mgrid[0:height, 0:width]
    # Scale the grid in height
    grid = grid / (height - 1) * 2 - 1

    # Obtain a grid of shape (height, width, 2)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 0, 1)

    # Expand of one dimension in position 0, obtaining a shape (1,height, width, 2)
    grid = np.expand_dims(grid, axis=0)
    return grid


def produce_warp_maps(origins, targets):
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(64, (5, 5))
            self.act1 = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.conv2 = tf.keras.layers.Conv2D(64, (5, 5))
            self.act2 = tf.keras.layers.LeakyReLU(alpha=0.2)
            if binary_images:
                self.convo = tf.keras.layers.Conv2D((1 + 1 + 2) * 2, (5, 5))
            else:
                self.convo = tf.keras.layers.Conv2D((3 + 3 + 2) * 2, (5, 5))

        def call(self, maps):
            x = tf.image.resize(maps, [map_height, map_width])
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.convo(x)
            return x

    model = MyModel()

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(maps, origins, targets):
        with tf.GradientTape() as tape:
            preds = model(maps)
            preds = tf.image.resize(preds, [height, width])

            # a = tf.random.uniform([maps.shape[0]])
            # res_targets, res_origins = warp(origins, targets, preds[..., :8] * a, preds[...,8:] * (1 - a))
            if binary_images:
                res_targets_, res_origins_ = warp(origins, targets, preds[:, :, :, 0:4], preds[:, :, :, 4:8])

                res_map = tfa.image.dense_image_warp(maps, preds[:, :, :,
                                                           2:4] * height * warp_scale)  # warp maps consistency checker
                res_map = tfa.image.dense_image_warp(res_map, preds[:, :, :, 6:8] * height * warp_scale)
            else:
                res_targets_, res_origins_ = warp(origins, targets, preds[:, :, :, 0:8], preds[:, :, :, 8:16])

                res_map = tfa.image.dense_image_warp(maps, preds[:, :, :,
                                                           6:8] * height * warp_scale)  # warp maps consistency checker
                res_map = tfa.image.dense_image_warp(res_map, preds[:, :, :, 14:16] * height * warp_scale)

            loss = loss_object(maps, res_map) * maps_loss_weight + loss_object(res_targets_, targets) * targets_loss_weight + loss_object(res_origins_, origins) * origins_loss_weight

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    maps = create_grid(height, width)
    maps = np.concatenate((maps, origins * 0.1, targets * 0.1), axis=-1).astype(np.float32)

    template = 'Epoch {}, Loss: {}'
    for i in range(TRAIN_EPOCHS):
        epoch = i + 1
        train_step(maps, origins, targets)

        if epoch % 100 == 0:
            print(template.format(epoch, train_loss.result()))

        if (epoch < 100 and epoch % 10 == 0) or \
                (epoch < 1000 and epoch % 100 == 0) or \
                (epoch % 1000 == 0):
            print("Epoch: ", epoch)
            preds = model(maps, training=False)[:1]
            preds = tf.image.resize(preds, [height, width])

            if binary_images:
                res_targets, res_origins = warp(origins, targets, preds[:, :, :, 0:4], preds[:, :, :, 4:8])
            else:
                res_targets, res_origins = warp(origins, targets, preds[:, :, :, 0:8], preds[:, :, :, 8:16])

            # Clip images for values out of [-1,1]
            res_targets = tf.clip_by_value(res_targets, -1, 1)[0]
            res_origins = tf.clip_by_value(res_origins, -1, 1)[0]
            # Convert the images to uint8
            res_targets = ((res_targets.numpy() - integer_to_float_bias) * integer_to_float_scaling).astype(np.uint8)
            res_origins = ((res_origins.numpy() - integer_to_float_bias) * integer_to_float_scaling).astype(np.uint8)
            if binary_images:
                # Binarize the images. Mininum value:0, maximum value: value_if_greater_than_threshold
                retVal, res_targets = cv2.threshold(res_targets, binary_threshold, value_if_greater_than_threshold, cv2.THRESH_BINARY)
                retVal, res_origins = cv2.threshold(res_origins, binary_threshold, value_if_greater_than_threshold, cv2.THRESH_BINARY)
                # Save the images
                cv2.imwrite("train/a_to_b_" + str(epoch) + ".png", res_targets)
                cv2.imwrite("train/b_to_a_" + str(epoch) + ".png", res_origins)
            else:
                # Save the image converting from RGB order to BGR order
                cv2.imwrite("train/a_to_b_" + str(epoch) + ".jpg", cv2.cvtColor(res_targets, cv2.COLOR_RGB2BGR))
                cv2.imwrite("train/b_to_a_" + str(epoch) + ".jpg", cv2.cvtColor(res_origins, cv2.COLOR_RGB2BGR))

            np.save('preds.npy', preds.numpy())

def use_warp_maps(origins, targets):

    # Load predictions file
    preds = np.load('preds.npy')
    # apply maps and save results

    org_strength = tf.reshape(tf.range(images_to_generate, dtype=tf.float32), [images_to_generate, 1, 1, 1]) / (images_to_generate - 1)
    trg_strength = tf.reverse(org_strength, axis=[0])

    # Set the video format
    fourcc = cv2.VideoWriter_fourcc(* video_format)
    # Settings for the video. width must be the first argument for the shape
    if binary_images:
        video = cv2.VideoWriter('morph/morph.mp4', fourcc, video_fps, (width, height), isColor=False)
    else:
        video = cv2.VideoWriter('morph/morph.mp4', fourcc, video_fps, (width, height), isColor=True)

    # Create the folder for the images
    try:
        os.makedirs("morph/morph_images")
        print("Creation of the folder morph/morph/images")
    except FileExistsError:
        print("The folder morph/morph_images has not been created because it already exists")


    for i in range(images_to_generate):
        preds_org = preds * org_strength[i]
        preds_trg = preds * trg_strength[i]

        if binary_images:
            res_targets, res_origins = warp(origins, targets, preds_org[:, :, :, 0:4], preds_trg[:, :, :, 4:8])
        else:
            res_targets, res_origins = warp(origins, targets, preds_org[:, :, :, 0:8], preds_trg[:, :, :, 8:16])
        res_targets = tf.clip_by_value(res_targets, -1, 1)
        res_origins = tf.clip_by_value(res_origins, -1, 1)

        results = res_targets * trg_strength[i] + res_origins * org_strength[i]
        res_numpy = results.numpy()


        # Convert to uint8
        img = ((res_numpy - integer_to_float_bias) * integer_to_float_scaling).astype(np.uint8)
        if binary_images:
            # Binarize
            retVal, img = cv2.threshold(np.squeeze(img), binary_threshold, value_if_greater_than_threshold, cv2.THRESH_BINARY)
            # Save image

            cv2.imwrite("morph/morph_images/image_" + str(i) + ".png", img)
            # Write image to video
            video.write(img)
        else:
            # Save image converting from RGB order to BGR order
            cv2.imwrite("morph/morph_images/image_" + str(i) + ".jpg", cv2.cvtColor(np.squeeze(img), cv2.COLOR_BGR2RGB))
            # Scale image for video
            if scale_images_for_video:
                img_scaled_for_video = img * video_scale_ratio
            # Write image to video converting from RGB order to BGR order
            video.write(cv2.cvtColor(np.squeeze(img_scaled_for_video), cv2.COLOR_RGB2BGR))

    shutil.make_archive("morph/morph_images", 'zip', "morph/morph_images/")
    print("Images zip folder created")
    cv2.destroyAllWindows()
    # Close video writer
    video.release()
    print('Result video saved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="Source file name", default=None)
    parser.add_argument("-t", "--target", help="Target file name", default=None)
    parser.add_argument("-e", "--train_epochs", help="Number of epochs to train network", default=TRAIN_EPOCHS,
                        type=int)
    parser.add_argument("-a", "--add_scale", help="Scaler for addition map", default=add_scale, type=float)
    parser.add_argument("-m", "--mult_scale", help="Scaler for multiplication map", default=mult_scale, type=float)
    parser.add_argument("-w", "--warp_scale", help="Scaler for warping map", default=warp_scale, type=float)
    parser.add_argument("-add_first", "--add_first", help="Should you add or multiply maps first", default=add_first,
                        type=bool)

    args = parser.parse_args()

    if not args.source:
        print("No source file provided!")
        exit()

    if not args.target:
        print("No target file provided!")
        exit()

    TRAIN_EPOCHS = args.train_epochs
    add_scale = args.add_scale
    mult_scale = args.mult_scale
    warp_scale = args.warp_scale
    add_first = args.add_first
    # Load the images
    if binary_images:
        dom_a = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)
        dom_b = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)
    else:
        dom_a = cv2.imread(args.source, cv2.IMREAD_COLOR)
        dom_b = cv2.imread(args.target, cv2.IMREAD_COLOR)
        # Since the color order in cv2.imread is BGR (blue, green, red) the order of the ndarray is changed in RGB
        dom_a = cv2.cvtColor(dom_a, cv2.COLOR_BGR2RGB)
        dom_b = cv2.cvtColor(dom_b, cv2.COLOR_BGR2RGB)
    # Scale the images
    dom_a = dom_a / integer_to_float_scaling + integer_to_float_bias
    dom_b = dom_b / integer_to_float_scaling + integer_to_float_bias
    # Reshape and convert to float32
    if binary_images:
        # Convert shape from (height, width) to (height, width, 1)
        dom_a = np.expand_dims(dom_a, axis=2)
        dom_b = np.expand_dims(dom_b, axis=2)
        # Convert shape from (heigth, width, 1) to (1, height, width, 1)
        origins = np.expand_dims(dom_a, axis=0).astype(np.float32)
        targets = np.expand_dims(dom_b, axis=0).astype(np.float32)
    else:
        # Convert shape from (heigth, width, 3) to (1, height, width, 3)
        origins = np.expand_dims(dom_a, axis=0).astype(np.float32)
        targets = np.expand_dims(dom_b, axis=0).astype(np.float32)

    produce_warp_maps(origins, targets)
    use_warp_maps(origins, targets)
