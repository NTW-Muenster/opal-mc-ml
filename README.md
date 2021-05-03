# Masterclass zum OPAL-Experiment mit maschinellem Lernen
Diese Masterclass basiert auf Daten des OPAL-Experiments am, mittlerweile demontierten, Large Electron-Positron Collider (LEP). Die einzelnen Ereignisse sind Kollisionen zwischen Elektronen und Positronen.

![Beispielererignis](http://www.hep.manchester.ac.uk/u/masterclass/masterclass2019/events/challenge1/events/x7708_23918.gif "Beispielereignis")

Diese Masterclass basiert auf einer [anderen Masterclass](http://www.hep.manchester.ac.uk/u/masterclass/masterclass2019/events/) ([deutsche Version](https://physicsmasterclasses.org/exercises/manchester/de/home.html)). Ursprunglich wurden die Ereignisse in Gruppen aufgeteilt und "per Hand" ausgewertet. Hier wird ein anderer Ansatz gewählt. Nur ein kleiner Teil der Gesamtmenge wird manuell ausgewertet. Mit diesen Daten wird ein [künstliches neuronales Netz](https://de.wikipedia.org/wiki/K%C3%BCnstliches_neuronales_Netz) trainiert, welches dann selbstständig die restlichen Events klassifizieren kann.

Die Eingabe besteht dabei nur aus den (leicht vorverarbeiteten) Ereignisbildern der ursprünglichen Masterclass. Aktuell können nur Bilder in Frontalaufnahme verarbeitet werden. Dies sind ca. 620 der insgesamt 1000 Ereignisbilder. Darüber hinaus wird eine Liste mit den Dateinamen und der manuell ermittelten Kategorie geladen um das Training und die anschließende Bewertung des Modells zu ermöglichen.

## Programm ausführen
Das Programm besteht aus einem IPython-Notebook und einer Python-Datei. Folgende Bibliotheken werden benötigt:
Name|[Anaconda](https://www.anaconda.com/)|[mybinder.org](https://mybinder.org)|[Google Colab](https://colab.research.google.com/)|[WWU JupyterHub](https://jupyterhub.wwu.de/)
---|---|---|---|---
|numpy|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|matplotlib|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|pandas|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|scikit-learn|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|scikit-image|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:
|tensorflow|:x:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:

### Anaconda
Auf lokalen Anaconda-Installationen kann das Notebook ohne Probleme ausgeführt werden. Dazu muss in der *Anaconda Prompt* der Befehl `pip install tensorflow` ausgeführt werden, um Tensorflow zu installieren.

### mybinder.org
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ntiltmann/opal-mc-ml/HEAD?filepath=opal_mc_ml.ipynb)

Bei mybinder.org ist die Masterclass **nicht** lauffähig, da zu wenig Arbeitsspeicher verfügbar ist.

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntiltmann/opal-mc-ml/blob/main/opal_mc_ml.ipynb)

Bei Google Colab lässt sich alles einwandfrei ausführen. Ein Google-Account ist erforderlich.

### WWU JupyterHub
[WWU JupyterHub](https://jupyterhub.wwu.de/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fntiltmann%2Fopal-mc-ml&urlpath=lab%2Ftree%2Fopal-mc-ml%2Fopal_mc_ml.ipynb&branch=main)

Im JupyterHub der [WWU](https://uni-muenster.de) muss scikit-image mit dem Befehl `pip install --user scikit-image` installiert werden. Das Trainieren funktioniert nur einwandfrei, wenn die Option mit 8GB Arbeitsspeicher gewählt wird. Eine WWU-Kennung wird benötigt.

## Technische Details
Die Bilder sind vorsortiert. Der Datensatz besteht nur noch aus Ereignisbildern in der Frontalansicht. Die Anzeigen des Event-Displays sind abgeschnitten um sicherzustellen, dass diese Bilddetails nicht mitgelernt werden. Außerdem sind alle Logos und Legenden abgeschnitten. Die Bilder sind auf eine Größe von 200x200px verkleinert, um das Lernen zu vereinfachen. Der Datensatz umfasst ca. 620 Ereignisse. Diese Ereignisse verteilen sich so auf die Zerfallskanäle, wie man dies messen würde (ca. 88% hadronisch, jeweils ca. 4% Elektron, Myon, Tau, siehe auch [hier](https://pdg.lbl.gov/2020/listings/rpp2020-list-z-boson.pdf)). Die Bilder können bei [sciebo](https://uni-muenster.sciebo.de/s/GHZpTpV0q8LYQjM) heruntergeladen werden. Bei Ausführen des Notebooks geschieht dies aber auch automatisch.

Um das Training robuster zu machen werden die Trainingsdaten verfielfacht. Dazu werden Kopien der Ereignisbilder erstellt und beliebig rotiert sowie zufällig gespiegelt (dies ändert die Aussage der Bilder in diesem Fall nicht). Die Anzahl aller Bilder wird mit 5 multipliziert. Darüber hinaus wird die Anzahl der Ereignisse mit leptonischen Zerfall zusätzlich mit 12 multipliziert. Dadurch sind die Eingabedaten gleichmäßiger auf die Kategorien verteilt.

Das Training selbst wird von einem [Convolutional Neural Network](https://de.wikipedia.org/wiki/Convolutional_Neural_Network) übernommen. Das Netzwerk bestitzt drei faltende Ebenen (convolutional layer) und ein voll verbundenes (fully connected) Netzwerk mit einer versteckten Ebenen (hidden layer). Die Eingabeobjekte sind 200x200-Matrizen mit je drei Farbwerten pro Pixel. Die einzelnen Ebenen sind:

| Nr. | Bezeichnung    | englisch      | Hyperparameter                                                  | Ausgabe               | Anzahl Parameter                              |
|-----|----------------|---------------|-----------------------------------------------------------------|-----------------------|-----------------------------------------------|
| 1.  | falten         | convolutional | 32 Filter in einer 3x3-Nachbarschaft                            | 198x198px * 32 Filter | (3x3px * 3 Farbwerte + 1) * 32 = 896          |
| 2.  | zusammenfassen | pooling       | Maximum in einer 4x4-Nachbarschaft                              | 49x49px * 32 Filter   | 0                                             |
| 3.  | falten         | convolutional | 64 Filter in einer 3x3-Nachbarschaft                            | 47x47px * 64 Filter   | (3x3px * 32 Filter von 1. + 1) * 64 = 18.496  |
| 4.  | zusammenfassen | pooling       | Maximum einer 3x3-Nachbarschaft                                 | 15x15px * 64 Filter   | 0                                             |
| 5.  | falten         | convolutional | 64 Filter in einer 3x3-Nachbarschaft                            | 13x13px * 64 Filter   | (3x3px * 64 Filter von 3. + 1) * 64 = 36.928  |
| 6.  | zusammenfassen | pooling       | Maximum einer 3x3 Nachbarschaft                                 | 4x4px * 64 Filter     | 0                                             |
| 7.  | serialisieren  | flatten       | Aneinanderreihen aller Pixel                                    | 1024 Einzelwerte      | 0                                             |
| 8.  | voll verbunden | dense         | 64 Neuronen, jedes ist mit jedem aus vorheriger Ebene verbunden | 64 Einzelwerte        | (1024 + 1) * 64 = 65.600                      |
| 9.  | voll verbunden | dense         | 4 Neuronen, jedes ist mit jedem aus vorheriger Ebene verbunden  | 4 Einzelwerte         | (64 + 1) * 4 = 260                            |

Insgesamt werden also 122.180 Parameter trainiert. Als Aktivierungsfunktion wir ReLU genutzt. Die Werte der vier Ausgabeneuronen werden mit Softmax normalisiert. Die Kategorie mit dem größten Wert wird als Vorhersage verwendet.