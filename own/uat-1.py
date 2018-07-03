import tensorflow as tf

def univAprox(x, hidden_dim=50):
    # The simple case is f: R -> R
    input_dim = 1
    output_dim = 1

    with tf.variable_scope('UniversalApproximator'):
        ua_w = tf.get_variable('ua_w', shape=[input_dim, hidden_dim], initializer=tf.random_normal_initializer(stddev=.1))
        ua_b = tf.get_variable('ua_b', shape=[hidden_dim], initializer=tf.constant_initializer(0.))
        z = tf.matmul(x, ua_w) + ua_b
        a = tf.nn.relu(z) # we now have our hidden_dim activations

        ua_v = tf.get_variable('ua_v', shape=[hidden_dim, output_dim], initializer=tf.random_normal_initializer(stddev=.1))
        z = tf.matmul(a, ua_v)

    return z
