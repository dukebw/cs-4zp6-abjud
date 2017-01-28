import tensorflow as tf

class LogisticRegression(object):

    def __init__(self,input_imgs, output_target, batch_size=64):
        # our input
        self.input_imgs = input_imgs
        # our output
        self.output_target = output_target
        # Hyperparameters
        self.batch_size = batch_size
        # the resulting operator
        self.train_op = self._build_trainop()
    
    def 


    def _build_trainop(self):
        # simple model
        w = tf.get_variable("w1", [28*28, 10])

        y_pred = tf.matmul(self.input_imgs, w)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, self.output_target)

        # for monitoring
        loss_mean = tf.reduce_mean(loss)

        return tf.train.AdamOptimizer().minimize(loss)
