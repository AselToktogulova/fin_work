# Quantitavie Evaluation der Modelle

In der Arbeit wurde für ein intrinsisches Modell DT verwendet, weil er alle Anforderungen an ein
transparentes Modell erfüllt, indem die Entscheidungswege anschaulich und nachvollziehbar dargestellt
werden können. In dieser Arbeit wurde XGBoost Modell (eXtreme Gradient Boosting) in Rolle des
Black-Box-Modells angewendet.

Hier werden die Modelle auf drei unterschiedlichen Datensätzen evauliert.

Datensätze:

- Statlog German Dataset - https://archive.ics.uci.edu/ml/citation_policy.html

- Credit Card Fraud (ULB) - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Credit Card Fraud (IITTB) - https://www.kaggle.com/datasets/mishra5001/credit-card

Die Tests erfolgen auf einer 5-Kreuzvalidierung

Verwendete Metriken sind:

- Precision, Recall, F1-Score, FP-Rate
- ROC, AUC, Precision-Recall Curve

Information zu den Dateien in dem Ordner:

- evaluate_xgboost.ipynb - evaluiert XGboost Methode auf den oben genannten Datensätzen 
- evaluate_xgboost.py - Evaluation als pures .py File wegen der Zeitdauer auf dem großen Datensatz (Credit Card Fraud IITTB)
- evaluate_DT.ipynb - evaluiert DTs auf den oben genannten Datensätzen
- die restlichen Dateien werden durch die obenen genannten Skirpte erzeugt, es sind nähmlich Graphiken, die in der schriftlichen Ausarbeitung benutzt wurden



