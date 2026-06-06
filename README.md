# Desaprendizaje Secuencial y Degradación de Utilidad en LLMs

Este repositorio contiene la implementación experimental y los resultados de la investigación sobre desaprendizaje secuencial en modelos de lenguaje de gran escala (LLMs), utilizando el framework **OpenUnlearning** y modelos de la familia **Llama-3.2**.

El objetivo principal es evaluar cómo la partición del conocimiento en lotes (*batches*) secuenciales mitiga el "olvido catastrófico" y la degradación de utilidad en comparación con el desaprendizaje global.

## 🚀 Entorno de Ejecución
La experimentación fue realizada en un laboratorio de alto rendimiento con las siguientes especificaciones:
- **GPU:** NVIDIA RTX 6000 Ada (48 GB VRAM).
- **Entorno:** Conda (`tesis_unlearning`).
- **Arquitectura Base:** Meta-Llama-3.2-1B-Instruct.
- **Técnica de Adaptación:** LoRA (Low-Rank Adaptation).

## 🛠️ Instalación y Requisitos

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/imwaltdiaz/tesis-unlearning-llm.git](https://github.com/imwaltdiaz/tesis-unlearning-llm.git)
   cd tesis-unlearning-llm

    ```

2. **Recrear el entorno Conda:**
    ```bash
    conda create -n tesis_unlearning python=3.11
    conda activate tesis_unlearning
    pip install -r requirements.txt
    ```

3. **Autenticación en Hugging Face (Requerido para Llama-3.2):**
    Como Llama-3.2-1B-Instruct es un modelo protegido (*gated*), necesitas configurar tu token de Hugging Face. Genera un token en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) y corre:
    ```bash
    python -c "from huggingface_hub import login; login('TU_TOKEN_DE_HF')"
    ```

4. **Nota Importante para Usuarios de Windows (UTF-8):**
    Para evitar el error `UnicodeDecodeError` provocado por las plantillas de `trl` al decodificarse en codificación no-UTF8 por defecto en Windows (como `cp1252`), debes activar el modo UTF-8 de Python antes de correr cualquier script:
    * **PowerShell:**
        ```powershell
        $env:PYTHONUTF8=1
        ```
    * **CMD:**
        ```cmd
        set PYTHONUTF8=1
        ```
    * **Alternativa de ejecución directa:**
        ```bash
        python -X utf8 scripts/03_train_base_model.py
        ```




## 📈 Pipeline de Reproducibilidad

Para ejecutar el experimento completo de literatura desde cero, siga el orden de los scripts:

### Fase 1: Preparación

1. **Descarga de Datos:** Obtiene el corpus de Project Gutenberg.
```bash
python scripts/01_download_data.py

```


2. **Procesamiento y Splits:** Genera los conjuntos *Forget* y *Retain*.
```bash
python scripts/02_prepare_datasets.py

```



### Fase 2: Entrenamiento del Oráculo (M0)

3. **Fine-Tuning Base:** Entrena el modelo Llama-3.2 con el conocimiento literario completo.
```bash
python scripts/03_train_base_model.py

```



### Fase 3: Desaprendizaje y Evaluación

4. **Orquestador de Experimentos:** Ejecuta los algoritmos **GA, WGA, NPO y SimNPO** para 1, 3 y 5 lotes secuenciales.
```bash
python scripts/06_run_experiments.py

```


*Nota: Este script utiliza internamente `04_sequential_unlearn.py` para el entrenamiento iterativo y `05_evaluate.py` para el cálculo de ROUGE y Perplejidad.*

### Fase 4: Análisis Visual

5. **Generación de Gráficas:** Crea las curvas de comparación en `results/figures/`.
```bash
python scripts/07_plot_results.py

```



## 📂 Estructura del Proyecto

* `scripts/`: Código fuente del pipeline de entrenamiento y evaluación.
* `data/`: (Ignorado en Git) Datasets crudos y procesados.
* `results/runs/`: Archivos JSON con las métricas de cada experimento (Eficacia y Utilidad).
* `results/figures/`: Gráficas finales en formato PNG/PDF.
* `checkpoints/`: (Ignorado en Git) Pesos de los adaptadores LoRA.

## 📊 Algoritmos Evaluados

* **Gradient Ascent (GA):** Línea base destructiva.
* **Weighted GA (WGA):** GA con regularización por peso.
* **NPO (Negative Preference Optimization):** Optimización basada en preferencias.
* **SimNPO:** Variante optimizada de NPO.

## 🤝 Contribuciones y Modificaciones

Si desea aportar a este proyecto o modificar el dominio de estudio:

1. Modifique la lista `DOMAINS` en `scripts/06_run_experiments.py`.
2. Ajuste los hiperparámetros en los mapas de configuración de los scripts 04 y 06.
3. Para nuevos algoritmos, regístrelos en el `algo_map` de la lógica de entrenamiento.

## 📜 Cita

Si utiliza este código en su investigación, por favor cite:

> Diaz, W. (2026). *Desaprendizaje Secuencial y Degradación de Utilidad en LLMs*. Universidad de Lima.

