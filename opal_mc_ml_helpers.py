# Import of the packages. numpy, matplotlib, pandas and datetime are included in Anaconda.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Scikit-learn is used only for performing test-train-splits. It is included in Anaconda
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Scikit-image is used for dataset augmentation. It is included in Anaconda
from skimage.transform import rotate
from skimage import exposure

# The machine learning part is done with TensorFlow and Keras as frontend.
# Can be installed with "pip install tensorflow"
import tensorflow as tf
from tensorflow.keras import layers, models

# Paths to the category file and the images.
cat_filename = "events_list.csv"
picture_archive_filename = "events-images.zip"
picture_filepath = "events-images"

# Random seed
random_seed = None

def check_files():
    """Checks if category file in images are at the specified path. If not, they will be downloaded.
    """    
    import os, platform
    download_command_prefix = ""
    if platform.system() == "Linux":
        download_command_prefix = "wget -q -O "
    elif platform.system() == "Darwin":
        download_command_prefix = "curl -s -o "

    if picture_filepath in os.listdir():
        # Images available
        print("Images were found")
    else:
        # Images not available, searching for archive of images
        if picture_archive_filename in os.listdir():
            # Archive of images available
            print("Archive of images already available")
        else:
            # Archive of images unavailable, downloading
            if download_command_prefix != "":
                print("Downloading images...", end="")
                os.system(download_command_prefix + picture_archive_filename + " \"https://raw.githubusercontent.com/NTW-Muenster/opal-mc-ml/main/events-images.zip\"")
                print(" done!")
            else:
                raise Exception("No images found and no download possible on Windows. Please download all files manually.")
        # Unpacking images
        print("Unpacking images...", end="")
        os.system("unzip -q " + picture_archive_filename)
        print(" done!")

    if cat_filename in os.listdir():
        print("Category list was found")
    else:    
        if download_command_prefix != "":
            print("Downloading category list...", end="")
            os.system(download_command_prefix + cat_filename + " \"https://raw.githubusercontent.com/NTW-Muenster/opal-mc-ml/main/events_list.csv\"")
            print(" done!")
        else:
            raise Exception("No category list found and no automatic download possible on Windows. Please download all files manually.")

def set_random_seeds(seed):
    """Sets random seeds for Numpy and Tensorflow. The same seed is used for both modules.
    With given seed, the whole process is not random anymore, but will always produce the same results.
    The functions split_events_random, augment_events and MLModel.init use this seed. Within these functions, the global seed of Numpy or Tensorflow may be reset.

    Args:
        seed (number): Seed to be passed.
    """
    global random_seed
    random_seed = seed

class Event(object):
    """An event consists of a filename, an image and a category. Additionally, probabilities from predictions can be stored.

    Attributes:
        filename (text): Filename of the event.
        image (numpy array): Image of the event as numpy array.
        category (text or number): Category as 1, 2, 3, 4 or q, e, m, t.
        probabilities (list of numbers, optional): Prediction probabilities for the classes "q", "e", "m" and "t". Defaults to None.
    """    

    def __init__(self, filename, image, category, probabilities=None):
        """Constructor for an event-object.

        Args:
            filename (text): Filename of the event.
            image (numpy array): Image of the event as numpy array.
            category (text or number): Category as 1, 2, 3, 4 or q, e, m, t.
        """        
        self.filename = filename
        self.image = image
        self.category = category
        self.probabilities = probabilities

    def show_image(self, show_category=False, show_probability=False):
        """Displays the image of the event.

        Args:
            show_category (bool, optional): Sets if the category will be shown below the image. Defaults to False.
            show_probability (bool, optional): Shows the probabilites of classification in case there are any. Defaults to False.
        """        
        show_image(self, show_category, show_probability=show_probability)

def show_image(event, show_category=False, show_probability=False):  
    """Displays the image of the event.

    Args:
        show_category (bool, optional): Sets if the category will be shown below the image. Defaults to False.
        show_probability (bool, optional): Shows the probabilites of classification in case there are any. Defaults to False.
    """        
    plt.imshow(event.image, cmap=plt.cm.binary)
    plt.show()
    if show_category:
        print("Name:", event.filename, "\t Category:", event.category)
    else:
        print("Name:", event.filename)
    if show_probability:
        if event.probabilities is None:
            print("The event is fixed. There is no probability.")
        else:
            print("Probabilities for event being in class:")
            print("q\t{:.2f}%".format(event.probabilities[0]*100))
            print("e\t{:.2f}%".format(event.probabilities[1]*100))
            print("m\t{:.2f}%".format(event.probabilities[2]*100))
            print("t\t{:.2f}%".format(event.probabilities[3]*100))


def filenames_from_eventlist(eventlist):
    """Generates a list of filenames from a list of events.

    Args:
        eventlist (list of events): Input list.

    Returns:
        List: List of filenames.
    """    
    out=[]
    for event in eventlist:
        out.append(event.filename)
    return out

def images_from_eventlist(eventlist):
    """Generates a list of images from a list of events.

    Args:
        eventlist (list of events): Input list.

    Returns:
        List: List of images. 

    Warning: This is a list of 3D numpy arrays and not a 4D numpy array yet!
    """  
    out=[]  
    for event in eventlist:
        out.append(event.image)
    return out

def categories_from_eventlist(eventlist, numbers=False):
    """Generates a list of categories from a list of events.

    Args:
        eventlist (list of events): Input list.
        numbers (bool, optional): Sets if the categories are given as numbers (True) or letters (False). Defaults to False.

    Returns:
        List: List of categories
    """  
    out=[]
    for event in eventlist:
        if numbers:
            out.append(sign_to_number(event.category))
        else:
            out.append(event.category)
    return out



# Calculate visible branching ratios
# Source: https://pdg.lbl.gov/2020/listings/rpp2020-list-z-boson.pdf
br_ee = 3.3632
br_mm = 3.3662
br_tt = 3.3696
br_invisible = 20.000
br_hadron = 69.911

# Fraction of all visible decays (excluding neutrinos)
br_vis_total = br_ee + br_mm + br_tt + br_hadron
# Fraction of electronic decays in all visible decays
br_ee_vis = br_ee / br_vis_total
# Fraction of muonic decays in all visible decays
br_mm_vis = br_mm / br_vis_total
# Fraction of tau decays in all visible decays
br_tt_vis = br_tt / br_vis_total
# Fraction of hadronic decays in all visible decays
br_hadron_vis = br_hadron / br_vis_total

def show_overview(names, *arrays, show_theory=True):
    """Shows an overview of all visible branching ratios.

    Args:
        names (liste of strings): List with column headings.
        *arrays (list of events): One or multiple list of events. Separate by comma when using multiple lists.
        show_theory (bool, optional): Sets if the theoretical branching ratio is shown. Defaults to True.
    """    
    def tts(value):
        return "{:.2f}".format(value)

    # calculate entries and branching ratios in category arrays
    def count_entries(categories):
        total=len(categories)
        factor=100./total

        unique, counts = np.unique(categories, return_counts=True)
        counts_out=[]
        ratios_out=[]
        for i in range(4):
            if i in unique:
                location=np.where(unique==i)[0]
                counts_out.append(counts[location][0])
                ratios_out.append(counts[location][0]*factor)
            else:
                counts_out.append(0)
                ratios_out.append(0)
        return counts_out, ratios_out

    rs=[]
    cs=[]

    for imageset in arrays:
        iset = categories_from_eventlist(imageset, numbers=True)
        c, r = count_entries(iset)
        cs.append(c)
        rs.append(r)  
    
    labels = ["q", "e", "m", "t"]
    theory = np.array([br_hadron_vis, br_ee_vis, br_mm_vis, br_tt_vis])*100

    line1 = "\t"
    line2 = "#Evt:\t"
    for iset in range(len(arrays)):
        line1 += names[iset] + "\t\t\t"
        line2 += str(len(arrays[iset])) + "\t\t\t"
    if show_theory:
        line1 += "Theory\t\t\t"
    print(line1)
    print(line2)
    print("")

    for itype in range(len(labels)):
        output=labels[itype]+":\t"
        for iset in range(len(rs)):
            output += str(cs[iset][itype]) + "\t" + tts(rs[iset][itype]) + "%\t\t"
        if show_theory:
            output += tts(theory[itype]) + "%"
        print(output)

class Overview(object):
    """An object of this class can create an overview of visible branching ratios. 
    The method show always prints the current state of the given eventlists and not the state at the time, the list was added.
    """

    def __init__(self):
        """Contructor for an overview object.
        """
        self.tuplelist=[]

    def add_entry(self, name, eventlist, overwrite=False):
        """Adds a new entry to the overview. 
        There can not be two columns with the same name. In this case nothing will be added.
        An entry with a certain name can be overwritten.

        Args:
            name (string): Column heading of the entry.
            eventlist (list): Eventlist of the entry.
            overwrite (boolean): If True, a possible entry with the same name will be overwritten with a new eventlist.
        """
        found = False
        for i in range(len(self.tuplelist)):
            if self.tuplelist[i][0] == name:
                if overwrite == True:
                    self.tuplelist[i] = (name, eventlist)
                found = True
        if not found:
            self.tuplelist.append((name, eventlist))

    def delete_entry(self, index):
        """Deletes an entry from the overview.

        Args:
            index (number): Index of item to be deleted.

        Raises:
            IndexError: The given index is too low or too high.
        """
        if index<0 or index>=len(self.tuplelist):
            raise IndexError("List index out of range.")
        del self.tuplelist[index]

    def show(self, show_theory=True):
        """Prints the overview.

        Args:
            show_theory (bool, optional): Sets if the theoretical branching ratio is shown. Defaults to True.
        """
        names = []
        eventlists = []
        for tuple in self.tuplelist:
            names.append(tuple[0])
            eventlists.append(tuple[1])
        show_overview(names, *eventlists, show_theory=True)

def sign_to_number(sign):
    """Converts decay categories signs to numbers.

    Args:
        sign (letter): Input category.

    Raises:
        RuntimeError: Letter does not match any category.

    Returns:
        Number: Output category.
    """    
    if sign=="q":
        return 0
    elif sign=="e":
        return 1
    elif sign=="m":
        return 2
    elif sign=="t":
        return 3
    else:
        raise RuntimeError("\"" + str(sign) + "\" is not a valid identifier for decay classes")

def number_to_sign(number):
    """Converts decay categories numbers to signs.

    Args:
        number (number): Input category.

    Raises:
        RuntimeError: Number does not match any category.

    Returns:
        Letter: Output category.
    """    
    if number==0:
        return "q"
    elif number==1:
        return "e"
    elif number==2:
        return "m"
    elif number==3:
        return "t"
    else:
        raise RuntimeError("\"" + str(number) + "\" is not a valid identifier for decay classes")


def show_confusion_matrix(true_eventlist, predicted_eventlist):
    """Shows confusion matrix.

    Args:
        true_eventlist (list): List of events with true categories.
        predicted_eventlist (list): List of events with predicted categories.
    """  

    # also see https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/_plot/confusion_matrix.py#L12
    true=np.array(categories_from_eventlist(true_eventlist, numbers=True))
    predicted=np.array(categories_from_eventlist(predicted_eventlist, numbers=True))
    
    import matplotlib.colors as colors
    cmat=tf.math.confusion_matrix(true, predicted).numpy()
    fig, ax = plt.subplots()

    cm = cmat
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm+1, interpolation='nearest', cmap='viridis', norm=colors.LogNorm(vmin=1, vmax=np.max(cmat)))
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    from itertools import product
    text_ = np.empty_like(cm, dtype=object)
    thresh = (cm.max()+1 + cm.min()+1) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_cm = format(cm[i, j], '.0f')
        text_[i, j] = ax.text(j, i, text_cm, ha="center", va="center", color=color)

    display_labels = ["q", "e", "m", "t"]
    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True category",
            xlabel="Predicted category")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="horizontal")
    figure_ = fig
    ax_ = ax
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.show()
    print("Total prediction accuracy: {:.4f}".format(calculate_prediction_accuracy(true_eventlist, predicted_eventlist)))

def calculate_prediction_accuracy(true_eventlist, predicted_eventlist):
    """Calculates the prediction accuracy for given eventlists of true and predicted events.

    Args:
        true_eventlist (list): List of events with true categories.
        predicted_eventlist (list): List of events with predicted categories.

    Returns:
        number: Prediction accuracy.
    """
    true=np.array(categories_from_eventlist(true_eventlist, numbers=True))
    predicted=np.array(categories_from_eventlist(predicted_eventlist, numbers=True))
    cmat=tf.math.confusion_matrix(true, predicted).numpy()
    return np.trace(cmat)/np.sum(cmat)

def plot_metrics(historylist, show_accuracy=True, show_loss=False):
    """Shows the learning curve with respect to accuracy or loss. By default, only the accuracy learning curve is shown.

    Args:
        historylist (list of TF history objects): List of learning histories of the model.
        show_accuracy (bool, optional): Shows the accuracy. Defaults to True.
        show_loss (bool, optional): Shows the loss. Defaults to False.
    """       
    if show_loss:
        prev_epochs=0
        for run, history in enumerate(historylist):
            plt.plot(np.array(history.epoch)+1+prev_epochs, history.history['loss'], color="C"+str(run%10), label='Training, Run ' + str(run+1), marker="^")
            plt.plot(np.array(history.epoch)+1+prev_epochs, history.history['val_loss'], color="C"+str(run%10), linestyle="--", label='Validation, Run ' + str(run+1), marker=".")
            prev_epochs+=np.max(history.epoch)+1
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim([0, plt.ylim()[1]])
        plt.legend(loc=1)
        plt.show()
    
    if show_accuracy:
        prev_epochs=0
        for run, history in enumerate(historylist):
            plt.plot(np.array(history.epoch)+1+prev_epochs, history.history['accuracy'], color="C"+str(run%10), label='Training, Run ' + str(run+1), marker="^")
            plt.plot(np.array(history.epoch)+1+prev_epochs, history.history['val_accuracy'], color="C"+str(run%10), linestyle="--", label='Validation, Run ' + str(run+1), marker=".")
            prev_epochs+=np.max(history.epoch)+1
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0,1])
        plt.legend(loc=4)
        plt.show()



def load_events():
    """Loads all images from the folder and combines them with category information from .csv file.

    Returns:
        List: List of events.
    """    
    df = pd.read_csv(cat_filename, delimiter=";", header=None)
    eventlist=[]
    for filename in df[0]:
        sign=df[df[0].str.match(filename)].iat[0,1]
        eventlist.append(Event(filename, plt.imread(picture_filepath + "/" + filename), sign))
    print(len(eventlist), "events loaded")
    return eventlist



# Augment the dataset by copying the pictures with an integer factor rotating and/or flipping them randomly

# factor is integer: all categories are augmented with same factor
# factor is sequence with length 3: last 3 categories are augmented with given factors
# factor is sequence with length 4: all categories are augmented with given factors
def augment_events(eventlist, factor):
    """The dataset is augmented by copying the images and rotating and/or flipping the randomly. The output is shuffled.
    Resets Numpy random seed with current value of global variable random_seed.

    Args:
        eventlist (list): List of events
        factor (number or list): If factor is a number, all decay channels are augmented by this factor. If factor is a list of four entries, the decay channels are augmented according to the order "q, e, m, t".
    Raises:
        ValueError: Factor has unexpected structure.

    Returns:
        List: List of events.
    """    
    if hasattr(factor, "__len__") and (not isinstance(factor, str)):
        if len(factor) == 4:
            pass
        elif len(factor) == 3:
            factor = [1, factor[0], factor[1], factor[2]]
        else:
            raise ValueError("Dimension of factor is not equal to 4.")
    else:
        factor = [factor, factor, factor, factor]

    np.random.seed(random_seed)

    # rotate images by random angle and flip randomly
    def rot_mirr_eq(event):
        angle = np.random.randint(0, 360)
        mirror = np.random.randint(0, 2)
        image_out = rotate(event.image, angle)
        if mirror==1:
            image_out = np.fliplr(image_out)
        image_out = exposure.rescale_intensity(image_out)
        return Event(event.filename, image_out, event.category)

    # loop over the images until enough copies are created
    eventlist_out=[]
    for event in eventlist:
        # for i_factor in range(factor[categories[i_events]]):
        for i_factor in range(factor[sign_to_number(event.category)]):
            eventlist_out.append(rot_mirr_eq(event))

    # ensure random distribution
    eventlist_out = shuffle(eventlist_out, random_state=random_seed)
    return eventlist_out

def split_events_random(eventlist, fraction_first_block):
    """Splits a list into two lists randomly.
    Uses global random seed.

    Args:
        eventlist (list): List of events.
        fraction_first_block (number): Fraction of items in the first new list.

    Returns:
        tuple: tuple containing:
            List: List of events (first block),
            List: List of events (second block)
    """    
    return train_test_split(eventlist, test_size=1-fraction_first_block, random_state=random_seed)

def show_false_predictions(true_eventlist, predicted_eventlist, count=5, show_probability=True):
    """Generates an overview with all false predicted events.

    Args:
        true_eventlist (list): True events.
        predicted_eventlist (list): Predicted events.
        count (number, optional): Number of events to be shown. Defaults to 5.
        show_probability (boolean, optional): Sets if classification probabilities are shown. Defaults to True.
    """    
    true = np.array(categories_from_eventlist(true_eventlist, numbers=True))
    predicted = np.array(categories_from_eventlist(predicted_eventlist, numbers=True))
    cmat = tf.math.confusion_matrix(true, predicted, num_classes=None, weights=None, dtype=tf.dtypes.int32, name=None).numpy()
    wrong_count = np.sum(cmat) - np.trace(cmat)
    print("Total of", wrong_count, "wrong predictions")
    if wrong_count > count:
        print("showing first", count, "...")
        stop=count
    else:
        stop=wrong_count
    print("")
    found = 0
    index = 0
    while found < stop:
        if true[index]!=predicted[index]:
            found+=1
            predicted_eventlist[index].show_image(show_probability=show_probability)
            print("Was predicted as", predicted_eventlist[index].category, "but is", true_eventlist[index].category, "\n")
        index+=1

def filter_by_category(eventlist, category):
    """This function creates a new eventlist, where only events of one category appear.

    Args:
        eventlist (list): Input eventlist.
        category ([type]): Category to be filtered for, given as "q", "e", "m" or "t".

    Returns:
        list: Filtered eventlist.
    """
    outlist=[]
    for event in eventlist:
        if event.category == category:
            outlist.append(event)
    return outlist

def show_only_one_category(eventlist, category, count=5):
    """Show all (or a few) events of one category in an eventlist.

    Args:
        eventlist (list): Input eventlist.
        category (string): Category given as letter ("q", "e", "m" or "t").
        count (int, optional): Number of events to be shown. Defaults to 5.
    """
    filtered_eventlist = filter_by_category(eventlist, category)
    print("Total of", len(filtered_eventlist), "events in category", category)
    if len(filtered_eventlist) > count:
        print("showing first", count, "...")
        stop=count
    else:
        stop=len(filtered_eventlist)
    for i in range(stop):
        filtered_eventlist[i].show_image(show_category=True)


class MLModel:
    """This class is a wrapper for the machine learning model. It simplifies the use of Tensorflow.
    Resets Tensorflow random seed with value of global variable random_seed.
    
    Attributes:
        model (tf.keras.models): Underlying tensorflow model.
        historylist (list of tf.keras.callbacks): List of all training outputs.
        training (list of Events): List containing the training data as Event objects.
        validation (list of Events): List containing the validation data as Event objects.
        train_time (datetime.timedelta): Stores the summed duration of the training operation.
    """    
    def __init__(self):
        """Constructor.
        """        
        self.model=None
        self.historylist=None
        self.training=None
        self.validation=None
        self.train_time=None
        tf.random.set_seed(random_seed)

    def load_structure_default(self):
        """Loads the default model structure. These are three convolution-pooling-layers and two fully connected layers.
        """        
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(4))

    def new_layer_convolution(self, count_filter=32, size_filter=(3, 3), activation='relu'):
        """Adds a convolutional layer.

        Args:
            count_filter (number, optional): Defines the number of individual filters. Defaults to 32.
            size_filter (tuple with two numbers, optional): Defines the size of the filter kernels. Defaults to (3, 3).
            activation (string, optional): Sets the activation function. Defaults to 'relu'.
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(count_filter, size_filter, activation=activation, input_shape=(200, 200, 3)))
        else:
            self.model.add(layers.Conv2D(count_filter, size_filter, activation=activation))

    def new_layer_pooling(self, size_filter=(3, 3)):
        """Adds a pooling layer.

        Args:
            size_filter (tuple with two numbers, optional): Defines the size of the neighboorhood to be pooled. Defaults to (3, 3).
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.MaxPooling2D(size_filter, input_shape=(200, 200, 3)))
        else:
            self.model.add(layers.MaxPooling2D(size_filter))

    def new_layer_flatten(self):
        """Serializes all neurons. This is used for switching from convolutional layers to fully connected layers.
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.Flatten(input_shape=(200, 200, 3)))
        else:
            self.model.add(layers.Flatten())

    def new_layer_dense(self, count_neurons=64, activation='relu'):
        """Adds a fully connected (dense) layer.

        Args:
            counts_neurons (number, optional): Defines the count of neurons in this layer. Defaults to 64.
            activation (string, optional): Sets the activation function. Defaults to 'relu'.
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.Dense(count_neurons, activation=activation, input_shape=(200, 200, 3)))
        else:
            self.model.add(layers.Dense(count_neurons, activation=activation))

    def new_layer_final(self):
        """Adds the last layer of a network. This is a fully connected layer with four outputs.
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.Dense(4, input_shape=(200, 200, 3)))
        else:
            self.model.add(layers.Dense(4))

    def delete_structure(self):
        """Deletes current model structure.
        """        
        self.model = None

    def show_structure(self):
        """Shows an overview of all model layers.

        Raises:
            RuntimeError: No model has been loaded or created.
        """        
        if self.model is None:
            raise RuntimeError("No model has been loaded or created")
        self.model.summary()

    def load_training_eventlist(self, training_eventlist):
        """Loads the eventlist for training.

        Args:
            training_eventlist (list): Eventlist with training events.
        """        
        self.training = training_eventlist   
        
    def load_validation_eventlist(self, validation_eventlist):
        """Loads the eventlist for validation.

        Args:
            validation_eventlist (list): Eventlist with validation events.
        """        
        self.validation = validation_eventlist

    def train(self, show_time=True, count_epochs=3):
        """Starts the training of the model.

        Args:
            show_time (bool, optional): Defines if the total time is shown after training. Defaults to True.
            counts_epochs (number, optional): Defines how many iterations over all training and validation data should be performed.

        Raises:
            RuntimeError: No model or no data.
        """        
        if self.model is None:
            raise RuntimeError("No model loaded or created")
        if self.training is None:
            raise RuntimeError("No training data loaded")
        if self.validation is None:
            raise RuntimeError("No validation data loaded")

        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        train_start=datetime.datetime.now()
        print("Starting training...")
        train_npcat=np.array(categories_from_eventlist(self.training, numbers=True))
        train_nppic=np.array(images_from_eventlist(self.training))
        vali_npcat=np.array(categories_from_eventlist(self.validation, numbers=True))
        vali_nppic=np.array(images_from_eventlist(self.validation))
 
        history_temp = self.model.fit(train_nppic, train_npcat, epochs=count_epochs, validation_data=(vali_nppic, vali_npcat))
        if self.historylist is None:
            self.historylist = []
        self.historylist.append(history_temp)

        train_end=datetime.datetime.now()
        if self.train_time is None:
            self.train_time=train_end-train_start
        else:
            self.train_time=train_end-train_start+self.train_time
        if show_time:
            print("Training finished, took", self.train_time)

    def show_learning_curve(self, **args):        
        """Shows the learning curve.

            Args:
                historylist (List of TF objects): List of learning histories of the model.
                show_accuracy (bool, optional): Shows the accuracy. Defaults to True.
                show_loss (bool, optional): Shows the loss. Defaults to False.

            Raises:
                RuntimeError: Model not trained.
        """    

        if self.train_time is None:
            raise RuntimeError("Model was not trained")
        plot_metrics(self.historylist, **args)

    def predict(self, test_eventlist):
        """Predict categories of given events using the learned model. Categories of input data will be ignored and overwritten with the prediction.

        Args:
            test_eventlist (list): List of events for testing.

        Raises:
            RuntimeError: Model not trained.

        Returns:
            List: Eventlist with predictions.
        """        

        if self.train_time is None:
            raise RuntimeError("Model was not trained")
        predictions = self.model.predict(np.array(images_from_eventlist(test_eventlist)))
        score = tf.nn.softmax(predictions)
        max = np.argmax(score, axis=1)
        prediction_eventlist=[]
        for i in range(len(max)):
            prediction_eventlist.append(Event(test_eventlist[i].filename, test_eventlist[i].image, number_to_sign(max[i]), score.numpy()[i]))
        return prediction_eventlist

# ########uncomment for debugging in notebook
# import os
# check_files()
