import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from collections import Counter

class ModeloDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.modelo = None
        self.scaler = None
        self.etiquetas_significativas = {}

    def cargar_datos(self, ruta_excel, hoja=0):
        df = pd.read_excel(ruta_excel, sheet_name=hoja)
        datos_numericos = df.select_dtypes(include=['float64', 'int64'])
        return datos_numericos, df

    def entrenar(self, datos):
        self.scaler = StandardScaler()
        datos_normalizados = self.scaler.fit_transform(datos)
        self.modelo = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        etiquetas = self.modelo.fit_predict(datos_normalizados)
        self.etiquetas_significativas = self.asignar_significados_por_tamaño(etiquetas)
        return etiquetas

    def guardar_modelo(self, ruta_modelo='modelo_dbscan.pkl'):
        if self.modelo is not None and self.scaler is not None:
            joblib.dump({
                'modelo': self.modelo,
                'scaler': self.scaler,
                'etiquetas_significativas': self.etiquetas_significativas
            }, ruta_modelo)
            print(f'Modelo guardado en {ruta_modelo}')
        else:
            print("Primero debes entrenar el modelo.")

    def cargar_modelo(self, ruta_modelo='modelo_dbscan.pkl'):
        datos = joblib.load(ruta_modelo)
        self.modelo = datos['modelo']
        self.scaler = datos['scaler']
        self.etiquetas_significativas = datos.get('etiquetas_significativas', {})
        print(f'Modelo cargado desde {ruta_modelo}')

    def predecir_nuevos(self, nuevos_datos):
        if self.modelo is None or self.scaler is None:
            raise Exception("El modelo no ha sido cargado o entrenado.")
        datos_norm = self.scaler.transform(nuevos_datos)
        return self.modelo.fit_predict(datos_norm)

    def graficar_clusters(self, datos, etiquetas, etiquetas_significativas=None):
        if datos.shape[1] != 2:
            print("La gráfica solo se puede generar con 2 características.")
            return

        plt.figure(figsize=(8, 6))
        colores = ['red', 'orange', 'green', 'blue', 'purple', 'brown']
        etiquetas_unicas = set(etiquetas)

        for cluster_id in etiquetas_unicas:
            indice = etiquetas == cluster_id
            color = 'gray' if cluster_id == -1 else colores[cluster_id % len(colores)]
            label = 'Ruido' if cluster_id == -1 else f'Cluster {cluster_id}'
            if etiquetas_significativas and cluster_id in etiquetas_significativas:
                label = etiquetas_significativas[cluster_id]
            plt.scatter(datos.iloc[indice, 0], datos.iloc[indice, 1], c=color, label=label, s=60, edgecolors='k')

        plt.title("Clusters DBSCAN")
        plt.xlabel("Característica 1")
        plt.ylabel("Característica 2")
        plt.legend()
        plt.grid(True)
        plt.show()

    def asignar_significados_por_tamaño(self, etiquetas):
        conteo = Counter(etiquetas)
        if -1 in conteo:
            conteo.pop(-1)

        if not conteo:
            return {}

        ordenados = sorted(conteo.items(), key=lambda x: x[1], reverse=True)
        etiquetas_significativas = {}

        for i, (cluster_id, _) in enumerate(ordenados):
            if i == 0:
                etiquetas_significativas[cluster_id] = 'Precaución (Normal)'
            elif i == 1:
                etiquetas_significativas[cluster_id] = 'Advertencia'
            else:
                etiquetas_significativas[cluster_id] = 'Falla'

        etiquetas_significativas[-1] = 'Anomalía o Ruido'
        return etiquetas_significativas

# Uso de ejemplo
if __name__ == "__main__":
    modelo = ModeloDBSCAN(eps=0.3, min_samples=4)
    ruta_excel = "tus_datos.xlsx"
    datos, df_original = modelo.cargar_datos(ruta_excel)
    etiquetas = modelo.entrenar(datos)
    df_original['Cluster'] = etiquetas
    modelo.graficar_clusters(datos, etiquetas, modelo.etiquetas_significativas)
    modelo.guardar_modelo("modelo_dbscan.pkl")
