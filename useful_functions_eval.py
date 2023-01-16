import matplotlib.pyplot as plt

def compare_histories(history_before, history_after, initial_epochs):
  """
    Compares two tensorflow History objects.
  """

  # original
  acc = history_before.history["accuracy"]
  loss = history_before.history["loss"] 
  val_acc = history_before.history["val_accuracy"]
  val_loss = history_before.history["val_loss"]

  # after fine-tuning
  total_acc = acc + history_after.history["accuracy"]
  total_loss = loss + history_after.history["loss"] 
  total_val_acc = val_acc + history_after.history["val_accuracy"]
  total_val_loss = val_loss + history_after.history["val_loss"]

  # plot
  plt.figure(figsize=(8,10))
  

  plt.subplot(2,1,1)
  plt.title("Accuracy curve")
  plt.plot(total_acc, label="acc")
  plt.plot(total_val_acc, label="val_acc")
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start fine-tuning")
  plt.xlabel("Epochs")
  plt.legend(loc="lower right")

  plt.subplot(2,1,2)
  plt.title("Loss curve")
  plt.plot(total_loss, label="loss")
  plt.plot(total_val_loss, label="val_loss")
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start fine-tuning")
  plt.xlabel("Epochs")
  plt.legend()
  
  
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();
