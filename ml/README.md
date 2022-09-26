# Erklärbare Künstliche Intelligenz in den Finanzwissenschafte

Erklärbare künstliche Intelligenz (Explainable Artificial
Intelligence - XAI) wird als Lösung gesehen, um KI-Systeme vertraulicher und weniger zu einer
„Black-Box“ zu machen. XAI ist essenziell, um Transparenz, Vertrauen und Verantwortlichkeit
zu gewährleisten - was insbesondere im Finanzsektor von größter Bedeutung ist. Um dieser Frage
nachzugehen, wurden in dieser Arbeit die Methoden der XAI untersucht und analysiert. 

Das Repository repräsentiert den Code, der in dieser Arbeit entstand.

es gibt folgende Ordner:

- data - dient zur Speicherung aller verwendeten Datensätze (ist in .gitignore vermerkt, dass die Datensätze nicht hochgelanden werden, sieh github policies)
  - Statlog German Dataset - https://archive.ics.uci.edu/ml/citation_policy.html
  - Credit Card Fraud (ULB) - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  - Credit Card Fraud (IITTB) - https://www.kaggle.com/datasets/mishra5001/credit-card
- models - speichert jegliche Modelle, die durch Experimente entstanden (ist ebenfalls in .gitignore vermerkt, dass die nicht hochgeladen werden, sieh github policies) 
- models_evaluation - speichert alle Dateien/Skripte, die sich mit der Evalaution der Modelle befassen. In dem Ordner existiert ebenfalls ein README-File mit weiteren Informationen
- xAIs_evaluatuib - beinhaltet Dateien, die erstellt wurden um die Interviews mit den Experten aus der Finanzbranche durchzuführen. Der Ordner beinhaltet ebenfalls ein README mit weiteren Informationen zu den Dateien/Skripten.
- Die Dateien in diesem Ordner sind unterschiedliche Tests. Hier ist die kurze Beschreibung:
  - create_datesets.ipynb - Eine Untersuchung des creditcard (ULB) Datensatzes und Anwendung der SMOTE Methode
  - dataset_examination_onehot.ipynb - Eine Untersuchung weiterer zwei Sätze, sowie das Mapping der Merkmale mit der One-Hot-Kodierung
  - decision_tree.ipynb, nn.ipynb, xgboost - sind jupyter Notebooks, die unterschiedliche ML-Modelle erproben 
  - nn_shap.ipynb, shap_xgboost - SHAP Anwendung auf XGboost und NN und Visualisierung der Attribute
  - random_names* - sind die Files für die Erstellung künstlich generierten Vor- und Nachnamen für die spätere Evaluation, damit der Datensatz realer erscheint

Für Fragen und Anmerkungen gern Issues in Github erstellen, vielen Dank!
