import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Generación de datasets de ejemplo
def generar_datasets():
    # Esta función debe crear y devolver los datasets N1, N2, N3
    # Aquí hay un ejemplo simplificado:
    X1 = np.random.rand(100, 10)  # Dataset 1
    y1 = np.ones(100)  # Etiquetas para clase 1
    X2 = np.random.rand(100, 10)  # Dataset 2
    y2 = np.ones(100) * 2  # Etiquetas para clase 2
    X3 = np.random.rand(100, 10)  # Dataset 3
    y3 = np.ones(100) * 3  # Etiquetas para clase 3
    
    X = np.vstack((X1, X2, X3))
    y = np.hstack((y1, y2, y3))
    
    return X, y

# Funciones para la etapa de entrenamiento usando SVM
def entrenar_svm(X_train, y_train):
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    return clf

# Funciones para la etapa de pruebas
def etapa_pruebas(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    return y_pred, y_prob, report, matrix

# Ejecución del programa
X, y = generar_datasets()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento
clf = entrenar_svm(X_train, y_train)

# Pruebas
y_pred, y_prob, report, matrix = etapa_pruebas(clf, X_test, y_test)

# Resultados
print("Reporte de Clasificación:")
print(report)

print("Matriz de Confusión:")
print(matrix)

# Calcular las normativas y otros valores
def calcular_normativas(X, y, clf):
    clases = np.unique(y)
    resultados = {}
    
    for clase in clases:
        indices = np.where(y == clase)
        datos_clase = X[indices]
        
        centroide = np.mean(datos_clase, axis=0)
        norm_centroide = np.linalg.norm(centroide)
        norm_datos = np.linalg.norm(datos_clase, axis=1)
        
        resultados[clase] = {
            'centroide': centroide,
            'norm_centroide': norm_centroide,
            'norm_datos': norm_datos
        }
        
    return resultados

resultados_entrenamiento = calcular_normativas(X_train, y_train, clf)

for clase, valores in resultados_entrenamiento.items():
    print(f"Clase {clase}:")
    print(f"Centroide: {valores['centroide']}")
    print(f"Norma del Centroide: {valores['norm_centroide']}")
    print(f"Normas de los datos: {valores['norm_datos']}")

# Etapa de pruebas adicional
for i, x in enumerate(X_test):
    proyeccion = clf.decision_function([x])
    P_i = y_prob[i]
    F_i = y_pred[i]
    print(f"Proyección del ejemplo {i}: {proyeccion}")
    print(f"Probabilidad del ejemplo {i}: {P_i}")
    print(f"Predicción del ejemplo {i}: {F_i}")
