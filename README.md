# Masterclass zum OPAL-Experiment mit maschinellem Lernen
[![Documentation Status](https://readthedocs.org/projects/opal-mc-ml/badge/?version=latest)](https://opal-mc-ml.readthedocs.io/en/latest/?badge=latest)

## Einführung
Diese Masterclass basiert auf Daten des OPAL-Experiments am, mittlerweile demontierten, Large Electron-Positron Collider (LEP). Die einzelnen Ereignisse sind Kollisionen zwischen Elektronen und Positronen.

![Beispielererignis](https://www.hep.manchester.ac.uk/u/masterclass/masterclass2019/events/challenge2/events/x7626_14982.gif "Beispielereignis")

Diese Masterclass basiert auf einer [anderen Masterclass](http://www.hep.man.ac.uk/u/events/) ([deutsche Version](https://physicsmasterclasses.org/exercises/manchester/de/home.html)). Ursprünglich wurden die Ereignisse in Gruppen aufgeteilt und "per Hand" ausgewertet. Hier wird ein anderer Ansatz gewählt. Nur ein kleiner Teil der Gesamtmenge wird manuell ausgewertet. Mit diesen Daten wird ein [Convolutional Neural Network](https://de.wikipedia.org/wiki/Convolutional_Neural_Network) trainiert, welches dann selbstständig die restlichen Events klassifizieren kann.

Die Eingabe besteht dabei nur aus den (leicht vorverarbeiteten) Ereignisbildern der ursprünglichen Masterclass. Aktuell können nur Bilder in Frontalaufnahme verarbeitet werden. Dies sind ca. 620 der insgesamt 1000 Ereignisbilder. Darüber hinaus wird eine Liste mit den Dateinamen und der manuell ermittelten Kategorie geladen um das Training und die anschließende Bewertung des Modells zu ermöglichen.

[Hier](howto_masterclass_de.md) gibt es eine Übersicht über Themen, die angesprochen werden können. Auf der Übersicht basierend existiert eine Präsentation.

Die Dokumentation der verwendeten Klassen und Funktionen kann hier eingesehen werden: [Read the Docs](https://opal-mc-ml.readthedocs.io/en/latest/) (englisch)

## Programm ausführen
Das Programm besteht immer aus einem IPython-Notebook und einer Python-Datei. Das volle Programm ist in der Datei [mc_ml_loesung.ipynb](mc_ml_loesung.ipynb) vorhanden. Das Notebook [mc_ml_minimal.ipynb](mc_ml_minimal.ipynb) importiert nur alle Hilfsfunktionen aus der Python-Datei. Der Rest kann dann von Hand erarbeitet werden. Die Datei [mc_ml_sus.ipynb](mc_ml_sus.ipynb) ist vorbereitet für die Anwendung mit Schülerinnen und Schülern und besitzt ein Grundgerüst an Code kombiniert mit Programmieraufgaben.

Folgende Bibliotheken werden benötigt:
|Name|[Anaconda](https://www.anaconda.com/)|[mybinder.org](https://mybinder.org)|[Google Colab](https://colab.research.google.com/)|[WWU JupyterHub](https://jupyterhub.wwu.de/)
|---|---|---|---|---
|numpy|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|matplotlib|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|pandas|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|scikit-learn|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|scikit-image|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:
|tensorflow|:x:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:

### Anaconda
Auf lokalen [Anaconda](https://www.anaconda.com/products/individual#Downloads)-Installationen kann das Notebook ohne Probleme ausgeführt werden. Dazu muss in der *Anaconda Prompt* der Befehl `pip install tensorflow` ausgeführt werden, um TensorFlow zu installieren. Der automatische Dateidownload funktioniert bislang nicht, wenn Windows als Betriebssystem verwendet wird. In diesem Fall müssen die Dateien einzeln heruntergeladen werden. Alternativ kann der gesamte Inhalt dieses Repositories heruntergeladen werden. Alle benötigten Dateien sind enthalten. Lediglich [events-images.zip](events-images.zip) muss manuell entpackt werden.

### mybinder.org


Bei mybinder.org ist die Masterclass **nicht** lauffähig, da zu wenig Arbeitsspeicher verfügbar ist.

### Google Colab

Bei Google Colab lässt sich alles einwandfrei ausführen. Google stellt den Nutzern auch wahlweise GPU-Ressourcen zur Verfügung. Bei entsprechender Auswahl der Notebookeinstellungen wird das Training automatisch auf GPUs ausgeführt. Ein Google-Account ist erforderlich.

### WWU JupyterHub

Im JupyterHub der [WWU](https://uni-muenster.de) muss scikit-image für jeden User einzeln mit dem Befehl `pip install --user scikit-image` installiert werden. Das Trainieren funktioniert nur einwandfrei, wenn die Option mit 8GB Arbeitsspeicher gewählt wird. Eine WWU-Kennung wird benötigt.

### Links
|Normal|Minimal|Lösung|
|---|---|---|
|[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NTW-Muenster/opal-mc-ml/HEAD?filepath=mc_ml_sus.ipynb)|[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NTW-Muenster/opal-mc-ml/HEAD?filepath=mc_ml_minimal.ipynb)|[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NTW-Muenster/opal-mc-ml/HEAD?filepath=mc_ml_loesung.ipynb)|
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NTW-Muenster/opal-mc-ml/blob/main/mc_ml_sus.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NTW-Muenster/opal-mc-ml/blob/main/mc_ml_minimal.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NTW-Muenster/opal-mc-ml/blob/main/mc_ml_loesung.ipynb)
[WWU JupyterHub](https://jupyterhub.wwu.de/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNTW-Muenster%2Fopal-mc-ml&urlpath=tree%2Fopal-mc-ml%2Fmc_ml_sus.ipynb&branch=main)|[WWU JupyterHub](https://jupyterhub.wwu.de/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNTW-Muenster%2Fopal-mc-ml&urlpath=tree%2Fopal-mc-ml%2Fmc_ml_minimal.ipynb&branch=main)|[WWU JupyterHub](https://jupyterhub.wwu.de/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FNTW-Muenster%2Fopal-mc-ml&urlpath=tree%2Fopal-mc-ml%2Fmc_ml_loesung.ipynb&branch=main)


## Technische Details
Die Bilder sind vorsortiert. Der Datensatz besteht nur noch aus Ereignisbildern in der Frontalansicht. Die Anzeigen des Event-Displays sind abgeschnitten um sicherzustellen, dass diese Bilddetails nicht mitgelernt werden. Außerdem sind alle Logos und Legenden abgeschnitten. Die Bilder sind auf eine Größe von 200x200px verkleinert, um Ressourcen zu sparen. Der Datensatz umfasst ca. 620 Ereignisse. Diese Ereignisse verteilen sich so auf die Zerfallskanäle, wie man dies messen würde (ca. 88% hadronisch, jeweils ca. 4% Elektron, Myon, Tau, siehe auch [hier](https://pdg.lbl.gov/2020/listings/rpp2020-list-z-boson.pdf)). Sowohl der Zuschnitt der Bilder, als auch die Einordnung in die Kategorien ist nicht perfekt, aber ausreichend genug zum trainieren. Die Bilder sind in der Datei [events-images.zip](events-images.zip) gepackt und müssen in den Ordner des Jupyter Notebooks entpackt werden. Bei Ausführen des Notebooks geschieht dies aber auch automatisch.

Die Bibliotheken pandas, scikit-image und scikit-learn werden nur für die Vorverarbeitung der Bilder benötigt. Das eigentliche Training geschieht vollständig mit TensorFlow und Keras als Frontend. Scikit-learn bietet auch grundlegende Methoden des maschinellen Lernens, jedoch nicht die für Bilder sehr gut geeigneten Convolutional Neural Networks.

Um das Training robuster zu machen werden die Trainingsdaten verfielfacht. Dazu werden Kopien der Ereignisbilder erstellt und beliebig rotiert sowie zufällig gespiegelt (dies ändert die Aussage der Bilder in diesem Fall nicht). Die Anzahl aller Bilder wird mit 3 multipliziert. Darüber hinaus wird die Anzahl der Ereignisse mit leptonischen Zerfall zusätzlich mit 20 multipliziert. Dadurch sind die Eingabedaten gleichmäßiger auf die Kategorien verteilt.

Das Training selbst wird von einem [Convolutional Neural Network](https://de.wikipedia.org/wiki/Convolutional_Neural_Network) übernommen. Das Netzwerk bestitzt drei faltende Ebenen (convolutional layer) und ein voll verbundenes (fully connected) Netzwerk mit einer versteckten Ebenen (hidden layer). Die Eingabeobjekte sind 200x200-Matrizen mit je drei Farbwerten pro Pixel. Die einzelnen Ebenen sind:

| Nr. | Bezeichnung    | englisch      | Hyperparameter                                                  | Ausgabe               | Anzahl Parameter                              |
|-----|----------------|---------------|-----------------------------------------------------------------|-----------------------|-----------------------------------------------|
| 1.  | falten         | convolution   | 32 Filter in einer 3x3-Nachbarschaft                            | 198x198px * 32 Filter | (3x3px * 3 Farbwerte + 1) * 32 = 896          |
| 2.  | zusammenfassen | pooling       | Maximum in einer 3x3-Nachbarschaft                              | 66x66px * 32 Filter   | 0                                             |
| 3.  | falten         | convolution   | 64 Filter in einer 3x3-Nachbarschaft                            | 64x64px * 64 Filter   | (3x3px * 32 Filter von 1. + 1) * 64 = 18.496  |
| 4.  | zusammenfassen | pooling       | Maximum einer 3x3-Nachbarschaft                                 | 21x21px * 64 Filter   | 0                                             |
| 5.  | falten         | convolution   | 64 Filter in einer 3x3-Nachbarschaft                            | 19x19px * 64 Filter   | (3x3px * 64 Filter von 3. + 1) * 64 = 36.928  |
| 6.  | zusammenfassen | pooling       | Maximum einer 3x3 Nachbarschaft                                 | 6x6px * 64 Filter     | 0                                             |
| 7.  | serialisieren  | flatten       | Aneinanderreihen aller Pixel                                    | 2304 Einzelwerte      | 0                                             |
| 8.  | voll verbunden | dense         | 64 Neuronen, jedes ist mit jedem aus vorheriger Ebene verbunden | 64 Einzelwerte        | (2304 + 1) * 64 = 147.520                     |
| 9.  | voll verbunden | dense         | 4 Neuronen, jedes ist mit jedem aus vorheriger Ebene verbunden  | 4 Einzelwerte         | (64 + 1) * 4 = 260                            |

Insgesamt werden also 204.100 Parameter trainiert. Als Aktivierungsfunktion wir [ReLU](https://de.wikipedia.org/wiki/Rectifier_(neuronale_Netzwerke)) genutzt. Die Werte der vier Ausgabeneuronen werden mit [Softmax](https://de.wikipedia.org/wiki/Softmax-Funktion) normalisiert. Die Kategorie mit dem größten Wert wird als Vorhersage verwendet.

Die verwendete Struktur und Hyperparameters des Modells orientieren sich grob an anderen Beispielen für Bilderkennung und Klassifikation und stellen sich als brauchbar heraus. Es ist jedoch möglich, dass die Vorhersage sich mit weiterer Optimierung der Hyperparameter verbessert. Neben dem Standardmodell von oben können auch andere Modelle verwendet werden.

## Acknowledgements

The OPAL Machine Learning Masterclass is widely based on the [original OPAL Masterclass](http://www.hep.man.ac.uk/u/events/) from 1997 by Terry Wyatt, Pezouna Pieri, Akram Khan, David Ward, Nigel Watson and Andrew McNab. 

The machine learning "upgrade" is based on an idea by Christian Klein-Bösing. This Masterclass is written by Johanna Rätz and Nicolas Tiltmann.

Feel free to report any issues and ideas on this GitHub page.
