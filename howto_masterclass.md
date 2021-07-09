# How to Masterclass

Diese Anleitung soll eine Übersicht geben, wie eine Masterclass mit Daten des OPAK Experimentes und maschinellem Lernen aussehen **kann**. An vielen Stellen lassen sich Dinge auch anders oder besser gestalten. Die Präsentation orientiert sich an dieser Übersicht.

## Einführung in das Standardmodell

Der Vortrag beginnt mit einer kurzen Einführung in das Standardmodell der Teilchenphysik. Insbesondere sollten hier folgende Stichpunkte angesprochen werden:

- Drei Generationen der Materie
- Teilchen und Antiteilchen
- Wechselwirkungen und Botenteilchen
    - Z-Boson als ein Botenteilchen der schwachen Wechselwirkung

## Experimente in der Teilchenphysik

- Beschleuniger Large Electron-Positron-Collider (LEP)
    - Lage und Größe
    - Elektronen und Positronen
    - Vorgänger des LHC
- Detektor: OPAL
    - Zwiebelschalen, speziell:
        - Spurdetektoren
        - Elektromagnetische Kalorimeter
            - (elektromagnetische Schauer)
        - Hadronische Kalorimeter
        - Myonendetektoren
    - Welche Teilchen erzeugen wo Signale?
        - Elektron / Positron
        - Photon
        - Myon
        - Proton
        - Neutron
## Zerfall des Z-Boson

- Feynman-Diagramm
- Erzeugung aus Elektron und Positron
- Zerfall in Teilchen-Antiteilchen-Paar des gleichen Typs
    - Elektron
    - Myon
    - Tau
    - Hadronen
    - Neutrinos
        - nicht detektierbar mit OPAL
- Zerfall der Taus
    - Elektronen, Myonen oder Quarks
    - Beide Taus sind unabhängig -> Eine Seite Positron, andere Myon möglich
    - Aufnahme des Event-Displays
    - Beobachtbare Verzweigungsverhältnisse
    - Quarks:
        - mindestens zwei Jets mit vielen Teilchenspuren
        - diverse Einträge in beiden Kalorimetern
    - Elektronen:
        - Nur zwei Teilchenspuren
        - Nur Beiträge in e.m.-Kalorimeter
    - Myonen:
        - Nur zwei Teilchenspuren
        - kleine Einträge in hadronischem Kalorimeter
        - Hits im Myonendetektor
    - Tau:
        - Bei unterschiedlichem Zerfall auf beiden Seiten (z.B. links Positron, rechts Myon) sicher Tau-Zerfall
        - Bei gleichem Zerfall auf beiden Seiten (z.B. links Positron, rechts Elektron) nicht optisch unterscheidbar von direktem Zerfall in Elektronen
            - Energiebetrachtung kann das Problem lösen, da bei Zerfall der Tau Neutrinos Energie davontragen
            - Kann hier ignoriert werden, da die Energiedaten fehlen

## Machine Learning

- Künstliche Intelligenz vs. Machine Learning vs. Deep Learning
    - Einordnung künstliche Neuronale Netze in das Schema
- Einführung neuronale Netze
    - z.B. via Film (englisch)
- Erweiterung **Convolutional Neural Networks**
    - **Convolution**/Faltung allgemein
        - reduziert Anzahl der Gewichte
        - reduziert Overfitting
        - Robust gegenüber Verschiebungen
    - **Pooling** (=Subsampling)
        - reduziert Anzahl der Gewichte in nachfolgenden Ebenen
        - reduziert Overfitting
- Strategie
    - Kombiniere mehrere Convolutional Layer und Pooling Layer zur Merkmalserkennung
    - Anschließend voll verbundenes Netzwerk zur Klassifikation
- Unterschied
    - **Parameter** werden gelernt, z.B. Gewichte im Filter
    - **Hyperparameter**: werden vorgegeben, z.B. Filtergrößen

## Code

- ggf. Einführung in Python
- kurze Einführung in Jupyter Notebooks
- je nnach Vorkenntnissen entweder mit minimalen Code oder mit ganzem code starten
    - ganzer Code gibt schnelle Erfolge
        - anschließend können Hyperparameter angepasst werden uvm.
    - minimaler Code gibt das Gefühl, mehr selbst programmiert zu haben
- kurze Erklärungen:
    - Bild als Matrix
    - Klasse Event mit Dateiname, Bildmatrix und Kategorie
- Code wird Block für Block aufgebaut
- An passender Stelle eingehen auf
    - Test-tv-Split
    - Data Augmentation durch Drehungen und Spiegelungen
    - Training-Valierung-Split
    - Modellübersicht des Default-Modells
    - Lernkurve
    - Confusion Matrix
    - gemessene Verzweigungsverhältnisse
- Wenn noch Zeit: Hyperparameter können angepasst werden
    - Aufteilen der Events in Training, Validierung, Test
    - Faktor für die Verfielfältigung
        - Alle
        - Leptonen
    - (Wieviel Einfluss hat die Farbe? Reicht ein Schwarz-Weiß Bild auch?)
    - Anzahl der Epochen (Wie lange wird trainiert?)
    - **Einfaches neuronales Netz vs. Convolutional Neural Network (CNN)**
    - Modellstruktur:
        - Convolutional Layer
            - Anzahl
            - Filtergröße
        - Mit Pooling vs. Ohne Pooling
        - Dense Layer
            - Anzahl 
            - Anzahl Neuronen pro Ebene
        - (Aktivierungsfunktionen)
- Fragestellungen zum Ende/Fazit:
    - Wie gut hat die automatische Erkennung geklappt?
    - Kann man bestimmte Merkmale identifizieren, die dazu führen, dass bestimmte Bilder falsch einsortiert wurden?
    - Das Problem lässt sich auch ganz einfach klassisch lösen mit den Eventdaten oben im Bild. Welche Vorteile und welche Nachteile haben Machine Learning verfahren?
        - Vorteile:
            - Man muss sich selbst keine Gedanken machen, welche Kriterien zur Einteilung in welche Klasse führen (auch wenn das in diesem Fall nicht schwierig ist)
            - Weiter gedacht: Man hat nicht immer zusätzliche Daten zu den Bildern
        - Nachteile:
            - Training kostet Zeit und Ressourcen
            - Training funktioniert nur, wenn genügend Traingsdaten vorhanden sind
            - Vorhersage liegt manchmal aus nicht ersichtlichen Gründen falsch




