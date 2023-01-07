import tensorflow as tf

def load_and_prep_img(file_name, img_shape):
  """reads-in an img from a file name, turns it into tensor and reshapes it to (img_shape,img_shape,color channels) """
  
  # load img
  img = tf.io.read_file(file_name)

  # decode the file to tensor
  img = tf.image.decode_image(img)

  # resize an img
  img = tf.image.resize(img, size=[img_shape,img_shape])

  # rescale the img -> get all values between 0 and 1
  img = img/255.

  return img
  
 
def pred_and_plot(model, filename, classnames):
  """ Imports an image located at filename and makes a prediction using the model
  and plots the img with the predicted class as the title
  """

  # import the target img
  img = load_and_prep_img(filename)

  # make a prediction
  pred = model.predict(tf.expand_dims(img, axis =0))

  # set the predicted class
  pred_class = classnames[int(tf.round(pred))]

  # plot the img and predicted class
  plt.imshow(img)
  plt.title("Prediction: "+pred_class)
  
  plt.axis(False)
  
  

import matplotlib.image as mpimg
import random
import os

def view_sample(target_dir, target_class):
  """ view random image from a dataset """
  
  # setup the target directory
  target_folder = target_dir+target_class

  # get a random image path
  random_img = random.sample(os.listdir(target_folder), 1)

  # read-in the image
  img = mpimg.imread(target_folder+ "/"+random_img[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  # show the shape of the img
  print(f'Image shape: {img.shape}')
  
  
def plot_random_sample(model, dataset, y_true, classes):
  """ function that picks a random sample from a dataset and plots it with its true and predicted labels"""

  # set up a random integer
  i = random.randint(0, len(dataset))

  # create predictions and targets
  target_sample = dataset[i]
  y_prob = model.predict(target_sample.reshape(1, 28, 28))

  y_pred = classes[y_prob.argmax()]
  y_true = classes[y_true[i]]

  # plot the sample
  plt.imshow(target_sample, cmap=plt.cm.binary)

  # change the color of the titles depending on if the prediction is right or wrong
  if y_pred == y_true:
    color = "green"
  else:
    color= "red"

  # label info
  plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(y_pred, 100*tf.reduce_max(y_probs), y_true), color=color)

  
def plot_decision_boundary(model, X, y):
  
  """ Plot decision boundary """

  x_min, x_max = X[:,0].min(), X[:,0].max()+0.1
  y_min, y_max = X[:,1].min(), X[:,1].max()+0.1

  xx, yy = np.meshgrid(np.linspace(x_min,x_max,100), np.linspace(y_min, y_max,100))


  # change a 2-dimensional array or a multi-dimensional array into a contiguous flattened array
  x_in = np.c_[xx.ravel(),yy.ravel()]
  y_pred = model.predict(x_in)

  # check for multiclass classification
  if len(y_pred[0])>1:
    print("Multiclass-classification")
    y_pred=np.argmax(y_pred, axis=1).reshape(xx.shape)

  else:
    print("Binary classification")
    y_pred=np.round(y_pred).reshape(xx.shape)


  # plot the decision-surface with a two-color colormap
  plt.contourf(xx,yy,y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap=plt.cm.RdYlBu)

  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())



import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, figsize, classes=None, text_size=20):
  """ creates a confusion matrix """
  
  cm = confusion_matrix(y_true, tf.round(y_pred))
  
  # normalize the confusion matrix
  cm_norm = cm.astype("float")/ cm.sum(axis=1)[:, np.newaxis]

  n_classes = cm.shape[0]

  fig, ax = plt.subplots(figsize=figsize)

  # create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  # set labels to be classes
  if classes:
    labels = classes
  else:
    labels= np.arange(cm.shape[0])

  # label the axes
  ax.set(title="confusion matrix", 
        xlabel="predicted_label", 
        ylabel="true_label", 
        xticks=np.arange(n_classes), 
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        )

  # set x -axis labels to the bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # set threshold for different colors
  threshold =(cm.max()+cm.min())/2

  # plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(i,j,f"{cm[i,j]} ({cm_norm[i,j]*100: .1f}%)",
    horizontalalignment="center",
    color= "white" if cm[i,j] > threshold else "black",
    size=text_size
    )

  # adjust the lable size
  ax.yaxis.label.set_size(text_size)
  ax.xaxis.label.set_size(text_size)
  ax.title.set_size(text_size)
  
   
  
