import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import random
import os

def load_and_prep_img(file_name, img_shape):
  """ Reads-in an img from a file name, turns it into tensor 
  and reshapes it to (img_shape,img_shape,color channels) """
  
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
  

def plot_pred(model, test_data, class_names):
  """
    data --> tf.data.Dataset object -> (image , label) tuples 
  """
  image_batch , label_batch = test_data.as_numpy_iterator().next() 
  batch_prob = [model.predict(tf.expand_dims(img , axis = 0)) for img in image_batch]
  batch_preds = [class_names[np.argmax(prob)] for prob in batch_prob]

  plt.figure(figsize= (10 , 10))
  for i in range(4):
    ax = plt.subplot(2 , 2 , i + 1)
    if class_names[np.argmax(label_batch[i])] == batch_preds[i]:
      title_color = 'g'
    else:
      title_color = 'r'
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(f"""Original label: {class_names[np.argmax(label_batch[i])]},
     predicted label: {batch_preds[i]}, 
     probability: {batch_prob[i].max():.2f}""" , c = title_color)
    plt.axis('off')

  
def plot_decision_boundary(model, X, y):
  
  """ Plots decision boundary """

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


def plot_confusion_matrix(y_true, y_pred, figsize, classes=None, text_size=20):

  # create confusion matrix
  cm = confusion_matrix(y_true, tf.round(y_pred))
  
  # original confusion matrix divided by the number of samples in the corresponding class
  cm_norm = cm.astype("float")/ cm.sum(axis=1)[:, np.newaxis]

  # get the number of classes
  n_classes = cm.shape[0]

  # count accuracy
  correct = sum(cm[i][i] for i in range(len(cm)))
  acc = correct / len(y_pred)

  # create subplots
  fig, ax = plt.subplots(figsize=figsize)

  # create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  # set labels for classes
  if classes:
    labels = classes
  else:
    labels= np.arange(cm.shape[0])

  # label the axes
  ax.set(title=f"Confusion Matrix (overall accuracy:{acc*100: .1f}%)", 
        xlabel="Predictions", 
        ylabel="Classes", 
        xticks=np.arange(n_classes), 
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels
        )

  # set x-axis labels to the bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # set threshold for different colors
  threshold = np.mean(cm)

  # iterate over all the elements in the confusion matrix and
  # plot text on each cell : number of correctly classified samples and accuracy
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    # get all possible pairs of indices for the columns and rows of the confusion matrix
    plt.text(i,j,f"""{cm[i,j]} 
    ({cm_norm[i,j]*100: .1f}%)""",
    horizontalalignment="center",
    color= "white" if cm[i,j] > threshold else "black",
    size=text_size
    )

  # adjust the lable size
  ax.yaxis.label.set_size(text_size+10)
  ax.xaxis.label.set_size(text_size+10)
  ax.title.set_size(text_size+10)
