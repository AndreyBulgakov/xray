import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, shape):
  return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 1, 1, 1], padding='SAME')

def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def block(x, size, phase_train):
    W = weight_variable([3, 3, size, size])
    b = bias_variable([size])

    return tf.nn.relu(batch_norm(conv2d(x, W) + b, size, phase_train))

def create_net(x, y, phase_train):
    x_image = x
    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])

    h_conv1 = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1) + b_conv1, 16, phase_train))

    h_conv2 = block(h_conv1, 16, phase_train)
    h_conv3 = block(h_conv2, 16, phase_train)
    h_conv4 = block(h_conv3, 16, phase_train)
    h_conv5 = block(h_conv4, 16, phase_train)
    h_conv6 = block(h_conv5, 16, phase_train)
    h_conv7 = block(h_conv6, 16, phase_train)
    h_conv8 = block(h_conv7, 16, phase_train)
    h_conv9 = block(h_conv8, 16, phase_train)
    h_conv10 = block(h_conv9, 16, phase_train)
    h_conv11 = block(h_conv10, 16, phase_train)

    W_convOut = weight_variable([1, 1, 16, 1])
    b_convOut = bias_variable([1])

    h_convOut = tf.nn.relu(conv2d(h_conv11, W_convOut) + b_convOut)
    y_out = h_convOut

    loss = tf.nn.l2_loss(tf.sub(y_out, y))
    return loss, y_out

def conv_block(x, size):
    W1 = weight_variable([3, 3, size, size])
    b1 = bias_variable([size])
    #W2 = weight_variable([3, 1, size, size])
    #b2 = bias_variable([size])
    return conv2d(x, W1) + b1 #conv2d(conv2d(x, W1) + b1, W2) + b2

def res_block(x, size, phase_train):
    inp = x
    inp = tf.nn.relu(batch_norm(conv_block(inp, size), size, phase_train))
    inp = batch_norm(conv_block(inp, size), size, phase_train)

    return tf.nn.relu(inp + x)

def create_resnet(x, y, phase_train):
    x_image = x

    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])

    h = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1) + b_conv1, 16, phase_train))

    for i in range(7):
        h = res_block(h, 16, phase_train)

    W_convOut = weight_variable([1, 1, 16, 1])
    b_convOut = bias_variable([1])

    h_convOut = tf.nn.relu(conv2d(h, W_convOut) + b_convOut)
    y_out = tf.minimum(h_convOut, tf.constant(1, dtype='float32'))

    loss = tf.nn.l2_loss(y - y_out)
    return loss, y_out

def create_preview_net(x, y, phase_train):
    x_image = x

    x_preview = tf.image.resize_images(x, 32, 32)

    W_conv_p = weight_variable([3, 3, 1, 16])
    b_conv_p = bias_variable([16])

    h = tf.nn.relu(batch_norm(conv2d(x_preview, W_conv_p) + b_conv_p, 16, phase_train))

    for i in range(7):
        h = res_block(h, 16, phase_train)

    W_conv_p = weight_variable([1, 1, 16, 4])
    b_conv_p = bias_variable([4])

    h = tf.nn.relu(conv2d(h, W_conv_p) + b_conv_p)
    h1 = tf.image.resize_images(h, x.get_shape()[1], x.get_shape()[2])

    W_conv1 = weight_variable([3, 3, 1, 12])
    b_conv1 = bias_variable([12])

    h = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1) + b_conv1, 12, phase_train))

    h = tf.concat(3, [h, h1])

    for i in range(3):
        h = res_block(h, 16, phase_train)

    W_convOut = weight_variable([1, 1, 16, 1])
    b_convOut = bias_variable([1])

    h_convOut = tf.nn.relu(conv2d(h, W_convOut) + b_convOut)
    y_out = tf.minimum(h_convOut, tf.constant(1, dtype='float32'))

    loss = tf.nn.l2_loss(y - y_out)
    return loss, y_out

def cascade_block(x, channels, out_channels, depth, phase_train):
    h = x
    for i in range(depth):
        h = res_block(h, channels, phase_train)

    W_conv_p = weight_variable([1, 1, channels, out_channels])
    b_conv_p = bias_variable([out_channels])

    h = tf.nn.relu(conv2d(h, W_conv_p) + b_conv_p)
    return h

def resize(x, channels, size, phase_train):
    x_preview = tf.image.resize_images(x, size, size)

    W_conv_p = weight_variable([3, 3, 1, channels])
    b_conv_p = bias_variable([channels])

    h = tf.nn.relu(batch_norm(conv2d(x_preview, W_conv_p) + b_conv_p, channels, phase_train))
    return h

def create_cascade_net(x, y, phase_train):
    x_image = batch_norm(x, 1, phase_train)

    x_preview = resize(x_image, 16, 16, phase_train)
    h = x_preview
    h = cascade_block(h, 16, 8, 4, phase_train)

    h = tf.image.resize_images(h, 32, 32)
    x_preview = resize(x_image, 8, 32, phase_train)
    h = tf.concat(3, [h, x_preview])
    h = cascade_block(h, 16, 8, 4, phase_train)

    h = tf.image.resize_images(h, 64, 64)
    x_preview = resize(x_image, 8, 64, phase_train)
    h = tf.concat(3, [h, x_preview])
    h = cascade_block(h, 16, 8, 3, phase_train)

    h = tf.image.resize_images(h, 128, 128)
    x_preview = resize(x_image, 8, 128, phase_train)
    h = tf.concat(3, [h, x_preview])
    h = cascade_block(h, 16, 8, 2, phase_train)

    h = tf.image.resize_images(h, x.get_shape()[1], x.get_shape()[1])
    x_preview = resize(x_image, 8, x.get_shape()[1], phase_train)
    h = tf.concat(3, [h, x_preview])
    h = cascade_block(h, 16, 1, 2, phase_train)

    y_out = tf.minimum(h, tf.constant(1, dtype='float32'))

    loss = tf.nn.l2_loss(y - y_out)
    return loss, y_out