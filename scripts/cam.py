import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def cam(img_array, model, layer_name, cam_path="cam.jpg"):

    #Get the 128 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]

    get_output = K.function(
        [model.input],
        [model.get_layer(layer_name).output, model.output])

    [conv_outputs, predictions] = get_output([img_array])
    conv_outputs = conv_outputs[0, :, :, :]

    #Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

    for i, w in enumerate(class_weights[:, 0]):
            cam += w * conv_outputs[:, :, i]

    cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)

    return cam.numpy()