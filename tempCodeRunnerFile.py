from sklearn.metrics import accuracy_score
y_pred_forest = forest_clf.predict(X_test)
print(accuracy_score(y_pred_forest, y_test))