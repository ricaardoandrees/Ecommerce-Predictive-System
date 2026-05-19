import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos(ruta_archivo):
    """Carga el dataset y muestra información básica."""
    print("--- CARGANDO DATASET ---")
    df = pd.read_csv(ruta_archivo)
    print(df.head())
    print("\n--- INFORMACIÓN GENERAL ---")
    print(df.info())
    print("\n--- ESTADÍSTICAS DESCRIPTIVAS ---")
    print(df.describe())
    return df

def visualizar_datos(df):
    print("\n--- GENERANDO GRÁFICOS ---")
    
   # --- CARGA DE DATOS Y ANÁLISIS ---
    print(f"Dataset cargado con {df.shape[0]} registros.\n")
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df['VisitorType'].describe())

    print("Valores nulos por columna:\n", df.isnull().sum(), end='\n')

    # Gráficos de Caja para conocer de los valores atípicos
    cols_numericas = df.select_dtypes(include=['number']).columns
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(cols_numericas):
        sns.boxplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()
    
    print("Gráficos generados. Cierra la ventana del gráfico para continuar.")

    # --- MATRIZ DE CORRELACIÓN ---
    correlacion = df[cols_numericas].corr()
    print("\n--- MATRIZ DE CORRELACIÓN ---")
    print(correlacion)

    plt.figure(figsize=(12, 11))
    sns.heatmap(correlacion, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
    plt.title('Mapa de Calor de Correlaciones')
    plt.show()

    # --- SCATTERPLOTS (Relaciones) ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # PageValues vs ExitRates
    sns.scatterplot(data=df, x='PageValues', y='ExitRates', hue='Revenue', ax=axes[0], palette='viridis', alpha=0.5)
    axes[0].set_title('PageValues vs ExitRates (Impacto en Revenue)')

    # ExitRates vs BounceRates
    sns.scatterplot(data=df, x='ExitRates', y='BounceRates', hue='Revenue', ax=axes[1], palette='magma', alpha=0.5)
    axes[1].set_title('ExitRates vs BounceRates (Redundancia)')

    # ProductRelated vs ProductRelated_Duration
    sns.scatterplot(data=df, x='ProductRelated', y='ProductRelated_Duration', hue='Revenue', ax=axes[2], palette='coolwarm', alpha=0.5)
    axes[2].set_title('Cantidad vs Duración en Productos Related')

    plt.tight_layout()
    plt.show()

def ejecutar_investigacion():
    # Asegúrate de que el nombre del csv sea el correcto
    dataset = 'Dataset/dataset shop.csv' 
    
    try:
        df = cargar_datos(dataset)
        visualizar_datos(df)
        print("\n¡Investigación finalizada con éxito!")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {dataset}. Asegúrate de tenerlo en la misma carpeta.")

if __name__ == "__main__":
    ejecutar_investigacion()