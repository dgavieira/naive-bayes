import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from models.naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr


def main():
    file_path = './data/iris_dataset.xlsx'
    df = pd.read_excel(file_path)
    
    X = df[['meas1', 'meas2', 'meas3', 'meas4']].values
    y = df['species'].values

    # Supondo que o DataFrame tem as colunas ['meas1', 'meas2', 'meas3', 'meas4', 'species']
    X = df[['meas1', 'meas2', 'meas3', 'meas4']].values
    y = df['species'].values

    # Codificar os valores de y (species) para valores numéricos
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    nb_classifier = NaiveBayes()

    # Configurar validação cruzada com 5 pastas
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        'Fold': [],
        'Accuracy': [],
        'Pearson Coefficient': [],
        'MSE': []
    }

    fold_number = 1

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        nb_classifier.fit(X_train, y_train)
        y_pred = nb_classifier.predict(X_test)
        
        # Calcular acurácia
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calcular coeficiente de Pearson
        pearson_coef, _ = pearsonr(y_test, y_pred)
        
        # Calcular erro médio quadrático
        mse = mean_squared_error(y_test, y_pred)
        
        # Armazenar resultados
        results['Fold'].append(fold_number)
        results['Accuracy'].append(accuracy)
        results['Pearson Coefficient'].append(pearson_coef)
        results['MSE'].append(mse)
        
        fold_number += 1

    # Calcular valores médios e adicionar ao DataFrame
    mean_accuracy = np.mean(results['Accuracy'])
    mean_pearson = np.mean(results['Pearson Coefficient'])
    mean_mse = np.mean(results['MSE'])

    results['Fold'].append('Mean')
    results['Accuracy'].append(mean_accuracy)
    results['Pearson Coefficient'].append(mean_pearson)
    results['MSE'].append(mean_mse)

    # Criar o DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('./data/results.csv', index=False)

    # Exibir os resultados
    print(results_df)

if __name__ == '__main__':
    main()