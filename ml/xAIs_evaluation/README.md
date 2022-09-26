# Quantitative Evaluation der xAI Erklärungen - Notebook für die Interviews mit den Experten aus der Finanzbranche


In der Arbeit wurde für ein intrinsisches Modell DT verwendet, weil er alle Anforderungen an ein
transparentes Modell erfüllt, indem die Entscheidungswege anschaulich und nachvollziehbar dargestellt
werden können. In dieser Arbeit wurde XGBoost Modell (eXtreme Gradient Boosting) in Rolle des
Black-Box-Modells angewendet. XGBoost ist eine effiziente Implementierung des stochastischen
Gradient-Boosting-Algorithmus und bietet eine Reihe von Hyperparametern, die eine feine Kontrolle
über das Modell-Trainingsverfahren ermöglichen. Obwohl der Algorithmus im Allgemeinen auch bei
unbalancierten Klassifizierungsdatensätzen gut funktioniert, bietet er eine Möglichkeit, den Trainings-
algorithmus so abzustimmen, dass er bei Datensätzen mit einer starken Unterrepräsentation der Klasse
der Minderheitenklasse mehr Fokus schenkt. Für die Erklärung des Post-hoc-Modells wurde SHAP
verwendet, da die Verwendung dieser Erklärung auf viele Methoden anwendbar ist

in diesem Ordner befindet sich der jupyter-Notebook für die Evaluation der xAI Erklärungen. 

Das Interview wurde so präsentiert, dass die Bankexperten die Ergebnisse leicht verstehen können,
damit die Bankexperten nicht irritiert werden. D.h. der Code wurde ausgelagert und nur die nötigsten
Methodenaufrufe sind zu sehen. Zusätzlich sind solche interaktive Komponente wie Selektion, Knöpfe und Visualisierung der Bilder umgesetzt worden. Anstelle von anonymen Namen wurde der Generator
für zufällige Namen verwendet, was zu einer besseren Vorstellung des Anwendungsfalls führt.


**evaluation.ipynb** - ist der Notebook der bei den Interviews benutzt wurde. Die restlichen Files sind ausgelagerte Funktionen oder Bilder, die in dem genannten jupyter-Notebook benutzt werden.