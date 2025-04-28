import datetime
import tensorflow as tf
import numpy as np
import random
import time

# Constants
BATCH_SIZE = 1

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback and stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory 
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


# sigmoid function
def sigmoid(a):
  if 0 <= a:
    return 1 / (1 + np.exp(-a))
  else:
    # to avoid overflow
    return 1. - 1 / (1 + np.exp(a))

# soft-max function
def softmax(a):
    ea = np.exp(a - np.max(a))
    return ea / ea.sum()

class ClassificationModel:

  def __init__(self):
    # weights matrix
     self.W = []
     self.lr = 0.001
     self.time_per_epoch = 0

  # updating weights using Maximum Likelihood Estimation
  def logistic_regression(self, x, y, num_of_classes, batch_size=BATCH_SIZE):

    # start measuring time
    start_time = time.time()

    # initialize weights with zeros
    """ W.shape = (10,784)
        img.shape = (784, 1)
        output_shape = (10, 1)"""
    self.W = np.array([[0.]*784 for i in range(num_of_classes) ])

    # preprocess the data
    x = preprocess(x)

    # count probabilities
    for id, img in enumerate(x):
      # y_true vector
      y_true = np.zeros(shape=(num_of_classes,))
      # a = Wx (x -> img)
      a = self.W@img.flatten()
      p = softmax(a)

      # set value to one 
      y_true[y[id]] = 1
      """ W.shape = (10,784)
        p.shape = y_true.shape = (10,1)
        img.shape = (784, 1)
        output_shape = (10, 1)"""

      # W = W + lr*(y-true -p)x
      """batch size -> hyperparameter that determines the number of samples
       to use in each iteration when updating the model's weights
      """
      # set the gradient for updating weights
      gradient = np.outer(self.lr*(y_true-p), img)

      # keep track of the gradient over the batch
      if id % batch_size == 0:
        batch_gradient = gradient
      else:
        batch_gradient += gradient
      # update the weights when the batch size is reached
      if (id + 1) % batch_size == 0:
        self.W += batch_gradient / batch_size
    
    # end measuring time
    end_time = time.time()
    self.time_per_epoch = end_time - start_time
      

  def predict(self, x):
    
    y_preds = np.zeros(shape=(len(x),))
    for id, img in enumerate(x):
      # dot product
      a = self.W@img.flatten()
      # count prob
      p = softmax(a)
      # predict y
      y_preds[id] = np.argmax(p)
    
    return y_preds


  def plot_preds(self, x, y):
    
    # view 4 random samples of training data
    for num in range(1,5):
      # pick random index
      i = random.randint(0, len(x))
      # dot product
      a = self.W@x[i].flatten()
      # count prob
      p = softmax(a)
      # predict y
      y_pred = np.argmax(p)
      # plot
      plt.subplot(2,2, num)
      
      if (y[i]==y_pred):
        text_color = 'g'
      else:
        text_color = 'r'
      plt.title("""Original Label: {},
      Predicted label: {}""".format(y[i], y_pred), c=text_color)
      plt.imshow(x[i], cmap='binary')
      
      plt.axis("off")
