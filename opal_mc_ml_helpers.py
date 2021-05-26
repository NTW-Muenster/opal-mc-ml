# Hier werden alle Bibliotheken importiert. numpy, matplotlib, pandas und datetime sind Standardpakete in Anaconda.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Scikit-learn wird nur zum erstellen der Trainings-Test-Aufteilungen benötigt und ist Anaconda-Standardpaket.
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Scikit-image wird benötigt, um die Trainingsdaten zu verfielfältigen und ist ebenfalls Anaconda-Standardpaket.
from skimage.transform import rotate
from skimage import exposure

# Der Teil des maschinellen Lernens läuft in Tensorflow mit Keras als Schnittstelle.
# Installation mit "pip install tensorflow"
import tensorflow as tf
from tensorflow.keras import layers, models

# Pfade für die Datei mit den Kategorien und den Ordner mit Bilder.
cat_filepath = "all_events.csv"
picture_filepath = "all-events/"

def ueberpruefe_dateien():
    """Überprüft ob Bilder und Kategorienliste schon vorhanden sind. Ansonsten werden die Dateien heruntergeladen.
    """    
    import os
    if "all-events" in os.listdir():
        print("Bilder schon vorhanden")
    else:
        print("Lade Bilder...", end="")
        os.system("wget -O all-events.zip \"https://uni-muenster.sciebo.de/s/GHZpTpV0q8LYQjM/download\"")
        print(" abgeschlossen!")
        print("Entpacke Bilder...", end="")
        os.system("unzip all-events.zip")
        print(" abgeschlossen!")

    if "all_events.csv" in os.listdir():
        print("Kategorienliste schon vorhanden")
    else:    
        print("Lade Kategorienliste...", end="")
        os.system("wget -O all_events.csv \"https://uni-muenster.sciebo.de/s/cvZBNrEDIf3MMwn/download\"")
        print(" abgeschlossen!")


class Bilddatei(object):
    """Eine Bilddatei beinhaltet den Dateinamen, das Bild als Array und die Kategorie.
    """    

    def __init__(self, dateiname, bild, kategorie):
        """Konstruktor für ein Bilddatei-Objekt.

        Args:
            dateiname (Text): Dateiname des Bildes
            bild (Numpy Array): Bilddaten als Numpy Array
            kategorie (Text oder Zahl): Kategorie als 1, 2, 3, 4 oder q, e, m, t
        """        
        self.dateiname = dateiname
        self.bild = bild
        self.kategorie = kategorie

    def zeige_bild(self, zeige_kategorie=False):
        """Zeige dieses Bilddatei-Objekt.

        Args:
            zeige_kategorie (bool, optional): Legt fest, ob die Kategorie angezeigt werden soll. Standardmäßig False.
        """        
        zeige_bild(self, zeige_kategorie)

def zeige_bild(bilddatei, zeige_kategorie=False):  
    """Zeige gegebenes Bilddatei-Objekt.

    Args:
        bilddatei (Bilddatei): Anzuzeigende Bilddatei
        zeige_kategorie (bool, optional): Legt fest, ob die Kategorie angezeigt werden soll. Standardmäßig False.
    """    
    plt.imshow(bilddatei.bild, cmap=plt.cm.binary)
    plt.show()
    if zeige_kategorie:
        print("Name:", bilddatei.dateiname, "\t Kategorie:", bilddatei.kategorie)
    else:
        print("Name:", bilddatei.dateiname)

def dateinamen_aus_liste(bilddaten):
    """Generiert eine Liste der Dateinamen aus einer Liste von Bilddateien.

    Args:
        bilddaten (Liste von Bilddateien): Eingabeliste

    Returns:
        Liste: Liste der Dateinamen
    """    
    out=[]
    for bilddatei in bilddaten:
        out.append(bilddatei.dateiname)
    return out

def bilder_aus_liste(bilddaten):
    """Generiert eine Liste der Bilddaten aus einer Liste von Bilddateien.

    Args:
        bilddaten (Liste von Bilddateien): Eingabeliste

    Returns:
        Liste: Liste der Bilddaten. Achtung: Noch kein Numpy Array!
    """  
    out=[]  
    for bilddatei in bilddaten:
        out.append(bilddatei.bild)
    return out

def kategorien_aus_liste(bilddaten, zahlen=False):
    """Generiert eine Liste der Kategorien aus einer Liste von Bilddateien.

    Args:
        bilddaten (Liste von Bilddateien): Eingabeliste
        zahlen (bool, optional): Legt fest, ob die Kategorien als Ziffern angezeigt werden sollen (True). Standardmäßig Buchstaben (False)

    Returns:
        Liste: Liste der Kategorien als Buchstaben oder Ziffern
    """  
    out=[]
    for bilddatei in bilddaten:
        if zahlen:
            out.append(sign_to_number(bilddatei.kategorie))
        else:
            out.append(bilddatei.kategorie)
    return out



# Berechne theoretische Verzweigungsverhältnisse in Prozent
# Quelle: https://pdg.lbl.gov/2020/listings/rpp2020-list-z-boson.pdf
br_ee = 3.3632
br_mm = 3.3662
br_tt = 3.3696
br_invisible = 20.000
br_hadron = 69.911

# Anteil aller sichtbaren Zerfälle (alle außer reine Neutrinozerfälle)
br_vis_total = br_ee + br_mm + br_tt + br_hadron
# Anteil der Elektron-Zerfälle an allen sichtbaren Zerfällen
br_ee_vis = br_ee / br_vis_total
# Anteil der Elektron-Zerfälle an allen sichtbaren Zerfällen
br_mm_vis = br_mm / br_vis_total
# Anteil der Tau-Zerfälle an allen sichtbaren Zerfällen
br_tt_vis = br_tt / br_vis_total
# Anteil der hadronischen Zerfälle an allen sichtbaren Zerfällen
br_hadron_vis = br_hadron / br_vis_total

def zeige_uebersicht(names, *arrays, show_theory=True):
    """Zeige eine Übersichten der Anzahlen und Verzweigungsverhältnisse in mehreren Datensätzen.

    Args:
        names (Liste aus Texten): Liste mit den Spaltenübrschriften
        *arrays (Liste aus Bilddaten): Liste der Bilddateien. Bei mehreren Listen einzelne Listen mit Komma trennen.
        show_theory (bool, optional): Legt fest, ob die Theoriewerte mit angezeigt werden sollen. Standardmäßig True.
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
        iset = kategorien_aus_liste(imageset, zahlen=True)
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
        line1 += "Theorie\t\t\t"
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


def sign_to_number(sign):
    """Konvertiert Kategorien von Buchstaben in Ziffern.

    Args:
        sign (Buchstabe): Eingabekategorie

    Raises:
        RuntimeError: Buchstabe gehört zu keiner Kategorie.

    Returns:
        Ziffer: Ausgabekategorie
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
    """Konvertiert Kategorien von Ziffern in Buchstaben.

    Args:
        number (Ziffer): Eingabekategorie

    Raises:
        RuntimeError: Ziffer gehört zu keiner Kategorie

    Returns:
        Buchstabe: Ausgabekategorie
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


def zeige_confusion_matrix(echte_daten, vorhergesagte_daten):
    """Zeige confusion matrix

    Args:
        echte_daten (Liste): Liste mit Bilddateien aus den echten Kategorien
        vorhergesagte_daten (Liste): Liste mit Bilddateien aus der Vorhersage
    """    

    # angelehnt an https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/_plot/confusion_matrix.py#L12
    true=np.array(kategorien_aus_liste(echte_daten, zahlen=True))
    predicted=np.array(kategorien_aus_liste(vorhergesagte_daten, zahlen=True))
    
    import matplotlib.colors as colors
    cmat=tf.math.confusion_matrix(true, predicted, num_classes=None, weights=None, dtype=tf.dtypes.int32, name=None).numpy()
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
            ylabel="Echte Kategorie",
            xlabel="Vorhergesagte Kategorie")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="horizontal")
    figure_ = fig
    ax_ = ax
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.show()

def plot_metrics(history, zeige_treffergenauigkeit=True, zeige_verlust=False):
    """Zeigt die Lernkurve.

    Args:
        history (TF Objekt): Lernhistorie aus dem Modell
        zeige_treffergenauigkeit (bool, optional): Plotte Treffergenauigkeit. Standardmäßig True.
        zeige_verlust (bool, optional): Plotte Verlustfunktion. Standardmäßig False.
    """        
    if zeige_verlust:
        plt.plot(history.epoch, history.history['loss'], color="C0", label='Training')
        plt.plot(history.epoch, history.history['val_loss'], color="C0", linestyle="--", label='Validation')
        plt.xlabel("Epoche")
        plt.ylabel("Verlustfunktion")
        plt.ylim([0, plt.ylim()[1]])
        plt.legend()
        plt.show()
    
    if zeige_treffergenauigkeit:
        plt.plot(history.epoch, history.history['accuracy'], color="C0", label='Training')
        plt.plot(history.epoch, history.history['val_accuracy'], color="C0", linestyle="--", label='Validation')
        plt.xlabel("Epoche")
        plt.ylabel("Treffergenauigkeit")
        plt.ylim([0,1])
        plt.legend()
        plt.show()



def lade_bilder():
    """Lädt alle Bilder aus dem Bildordner zusammen mit den Kategorien in eine Liste aus Bilddatei-Objekten.

    Returns:
        Liste: Bilddateien
    """    
    df = pd.read_csv(cat_filepath, delimiter=";", header=None)
    bilddaten=[]
    for filename in df[0]:
        sign=df[df[0].str.match(filename)].iat[0,1]
        bilddaten.append(Bilddatei(filename, plt.imread(picture_filepath+filename), sign))
    print(len(bilddaten), "Bilder geladen")
    return bilddaten



# Augment the dataset by copying the pictures with an integer factor rotating and/or flipping them randomly

# factor is integer: all categories are augmented with same factor
# factor is sequence with length 3: last 3 categories are augmented with given factors
# factor is sequence with length 4: all categories are augmented with given factors
def vervielfaeltige_daten(bilddaten, factor):
    """Der gegebene Datensatz wird erweitert indem Bilder kopiert und die Kopien zufällig gedreht und/oder gespiegelt werden. Die Ausgabeliste wird
        zufällig angeordnet.

    Args:
        bilddaten (Liste): Liste der Bilddateien
        factor (Zahl oder Liste): Wenn factor eine Zahl ist werden alle Kategorien um diesen Faktor erweitert. Wenn factor eine Liste mit 4 Einträgen 
            ist werden die Bilder der Kategorien entsprechend der Reihenfolge q, e, m, t mit dem Faktor aus der Liste erweitert.

    Raises:
        ValueError: Faktor hat nicht die erwartete Struktur.

    Returns:
        Liste: Bilddateien
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

    # rotate images by random angle and flip randomly
    def rot_mirr_eq(bilddatei):
        angle = np.random.randint(0, 360)
        mirror = np.random.randint(0, 2)
        image_out = rotate(bilddatei.bild, angle)
        if mirror==1:
            image_out = np.fliplr(image_out)
        image_out = exposure.rescale_intensity(image_out)
        return Bilddatei(bilddatei.dateiname, image_out, bilddatei.kategorie)

    # loop over the images until enough copies are created
    bilddaten_out=[]
    for bilddatei in bilddaten:
        # for i_factor in range(factor[categories[i_events]]):
        for i_factor in range(factor[sign_to_number(bilddatei.kategorie)]):
            bilddaten_out.append(rot_mirr_eq(bilddatei))

    # ensure random distribution
    bilddaten_out = shuffle(bilddaten_out)
    return bilddaten_out

def trenne_bilddaten_zufaellig(bilddaten, anteil_erster_block):
    """Trennt einen Datensatz zufällig in zwei einzelne Datensätze auf.

    Args:
        bilddaten (Liste): Liste der Bilddatei-Objekte
        anteil_erster_block (Zahl): Anteil des ersten Blocks

    Returns:
        Liste: Bilddatei-Objekte aus dem ersten Block
        Liste: Bilddatei-Objekte aus dem zweiten Block
    """    
    return train_test_split(bilddaten, test_size=1-anteil_erster_block)

def zeige_falsche_vorhersagen(echte_daten, vorhergesagte_daten, anzahl=5):
    """Generiert eine Übersicht mit falsch vorhergesagten Bildern.

    Args:
        echte_daten (Liste): Echte Bilddaten
        vorhergesagte_daten (Liste): Vohergesagte Bilddaten
        anzahl (Zahl, optional): Anzahl der Bilder, die angezeigt werden sollen. Standardmäßig 5.
    """    
    true = np.array(kategorien_aus_liste(echte_daten, zahlen=True))
    predicted = np.array(kategorien_aus_liste(vorhergesagte_daten, zahlen=True))
    cmat = tf.math.confusion_matrix(true, predicted, num_classes=None, weights=None, dtype=tf.dtypes.int32, name=None).numpy()
    wrong_count = np.sum(cmat) - np.trace(cmat)
    print("Insgesamt", wrong_count, "falsche Vorhersagen")
    if wrong_count > anzahl:
        print("zeige erste", anzahl, "...")
        stop=anzahl
    else:
        stop=wrong_count
    print("")
    found = 0
    index = 0
    while found < stop:
        if true[index]!=predicted[index]:
            found+=1
            echte_daten[index].zeige_bild()
            print("wurde als", vorhergesagte_daten[index].kategorie, "erkannt, ist aber", echte_daten[index].kategorie, "\n")
        index+=1



class Modell:
    """Diese Klasse verarbeitet das Machine Learning Modell.
    """    
    def __init__(self):
        """Konstruktor.
        """        
        self.model=None
        self.history=None
        self.train=None
        self.vali=None
        self.train_time=None

    def lade_modellstruktur_standard(self):
        """Lade die standardmäßige Struktur des Modells. Dies ist ein faltendes neuronales Netz mit drei faltenen Ebenen und 2 voll verbundenen Ebenen.
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

    def neue_ebene_convolution(self, anzahl_filter=32, groesse_filter=(3, 3), aktivierung='relu'):
        """Füge eine faltende Ebene (convolutional layer) hinzu.

        Args:
            anzahl_filter (Zahl, optional): Definiert die Anzahl der Filter. Standardmäßig: 32.
            groesse_filter (Tupel mit zwei Zahlen, optional): Definiert die Größe der Filter. Standardmäßig: (3, 3).
            aktivierung (Text, optional): Definiert die Aktivierungsfunktion. Standardmäßig: 'relu'.
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(anzahl_filter, groesse_filter, activation=aktivierung, input_shape=(200, 200, 3)))
        else:
            self.model.add(layers.Conv2D(anzahl_filter, groesse_filter, activation=aktivierung))

    def neue_ebene_pooling(self, groesse_filter=(3, 3)):
        """Füge eine neue zusammenfassende Ebene (pooling layer) hinzu.

        Args:
            groesse_filter (Tupel mit zwei Zahlen, optional): Definiert die Größe der Nachbarschaft, in der zusammengefasst wird. Standardmäßig: (3, 3).
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.MaxPooling2D(groesse_filter), input_shape=(200, 200, 3))
        else:
            self.model.add(layers.MaxPooling2D(groesse_filter))

    def neue_ebene_flatten(self):
        """Reiht alle Elemente der 2D-Struktur aneinander.
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.Flatten(), input_shape=(200, 200, 3))
        else:
            self.model.add(layers.Flatten())

    def neue_ebene_dense(self, anzahl_neuronen=64, aktivierung='relu'):
        """Fügt eine neue voll verbundene Ebene (dense / fully connected layer) hinzu.

        Args:
            anzahl_neuronen (Zahl, optional): Definiert die Anzahl der Neuronen in dieser Ebene. Standardmäßig: 64.
            aktivierung (Text, optional): Definiert die Aktivierungsfunktion. Standardmäßig: 'relu'.
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.Dense(anzahl_neuronen, activation=aktivierung), input_shape=(200, 200, 3))
        else:
            self.model.add(layers.Dense(anzahl_neuronen, activation=aktivierung))

    def neue_ebene_abschluss(self):
        """Fügt eine neue letzte Ebene mit 4 Neuronen hinzu.
        """        
        if self.model is None:
            self.model = models.Sequential()
            self.model.add(layers.Dense(4), input_shape=(200, 200, 3))
        else:
            self.model.add(layers.Dense(4))

    def loesche_modellstruktur(self):
        """Löscht die aktuelle Modellstruktur
        """        
        self.model = None

    def zeige_modelluebersicht(self):
        """Zeige eine Übersicht der einzelnen Modellebenen.

        Raises:
            RuntimeError: Es wurde noch kein Modell geladen
        """        
        if self.model is None:
            raise RuntimeError("Modell wurde noch nicht geladen")
        self.model.summary()

    def lade_trainingsdaten(self, trainingsdaten):
        """Stellt die Trainingsdaten bereit.

        Args:
            trainingsdaten (Liste): Trainings-Bilddateien
        """        
        self.train = trainingsdaten    
        
    def lade_validierungsdaten(self, validierungsdaten):
        """Stellt die Validierungsdaten bereit.

        Args:
            validierungsdaten (Liste): Validierungs-Bilddateien
        """        
        self.vali = validierungsdaten

    def trainiere(self, zeige_zeit=True, anzahl_epochen=3):
        """Starte das Modelltraining.

        Args:
            zeige_zeit (bool, optional): Wählt aus, ob die Trainingszeit angezeigt werden soll. Standardmäßig True.
            anzahl_epochen (Zahl, optional): Gibt an, wie viele Epochen das Training umfassen soll

        Raises:
            RuntimeError: Modell oder Daten fehlen
        """        
        if self.model is None:
            raise RuntimeError("Modell wurde (noch) nicht geladen")
        if self.train is None:
            raise RuntimeError("Trainingsdaten wurden noch nicht geladen")
        if self.vali is None:
            raise RuntimeError("Validierungsdaten wurden noch nicht geladen")

        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        train_start=datetime.datetime.now()
        if zeige_zeit:
            print("Starte Training...")
        train_npcat=np.array(kategorien_aus_liste(self.train, zahlen=True))
        train_nppic=np.array(bilder_aus_liste(self.train))
        vali_npcat=np.array(kategorien_aus_liste(self.vali, zahlen=True))
        vali_nppic=np.array(bilder_aus_liste(self.vali))
 
        self.history = self.model.fit(train_nppic, train_npcat, epochs=anzahl_epochen, validation_data=(vali_nppic, vali_npcat))
        train_end=datetime.datetime.now()
        self.train_time=train_end
        if zeige_zeit:
            print("Training abgeschlossen, hat", train_end-train_start, "gedauert")

    def zeige_lernkurve(self, **args):        
        """Zeigt die Lernkurve an.

        Args:
            history (TF Objekt): Lernhistorie aus dem Modell
            zeige_treffergenauigkeit (bool, optional): Plotte Treffergenauigkeit. Standardmäßig True.
            zeige_verlust (bool, optional): Plotte Verlustfunktion. Standardmäßig False.

        Raises:
            RuntimeError: Modell wurde noch nicht trainiert
        """       
        if self.train_time is None:
            raise RuntimeError("Modell wurde noch nicht trainiert")
        plot_metrics(self.history, **args)

    def erstelle_vorhersage(self, testdaten):
        """Erstelle Vorhersagen mit dem Modell.

        Args:
            testdaten (Liste): Liste mit Bilddateien. Die Kategorien der Eingabe werden ignoriert.

        Raises:
            RuntimeError: Modell wurde noch nicht trainiert

        Returns:
            Liste: Bilddateien mit Kategorien entsprechend der Vorhersage
        """        
        if self.train_time is None:
            raise RuntimeError("Modell wurde noch nicht trainiert")
        predictions = self.model.predict(np.array(bilder_aus_liste(testdaten)))
        score = tf.nn.softmax(predictions)
        max = np.argmax(score, axis=1)
        bilddaten_vorhersage=[]
        for i in range(len(max)):
            bilddaten_vorhersage.append(Bilddatei(testdaten[i].dateiname, testdaten[i].bild, number_to_sign(max[i])))
        return bilddaten_vorhersage

    # def zeige_gesamtuebersicht(self, testdaten, *arr):
    #     if self.history is None:
    #         raise RuntimeError("Modell wurde noch nicht trainiert")

    #   #hier noch train time
    #     self.zeige_modelluebersicht()
    #     zeige_uebersicht(*arr)
    #     self.zeige_lernkurve(self.history, zeige_treffergenauigkeit=True, zeige_verlust=True)
    #     bilddaten_vorhersage = self.erstelle_vorhersage(testdaten)
    #     zeige_confusion_matrix(testdaten, bilddaten_vorhersage)


# ########uncomment for debugging in notebook
# import os
# ueberpruefe_dateien()