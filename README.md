# Simulación del Hantavirus Andino (ANDV) usando Reinforcement Learning
### *Proyecto final para las clases “Introducción a Optimización” y “Matemáticas del Machine Learning”, semestre 2026-1S, Universidad Nacional de Colombia*
### Integrantes del grupo: Francisco Gutiérrez, Juan Soto, Camilo Acuña, Tuli Peña

---

<img width="426" height="280" alt="ANDV" src="https://github.com/user-attachments/assets/04820ad2-53b8-4248-a35f-e1b584fbd500" />

El **Andes Orthohantavirus (ANDV)** representa una amenaza para nuestro continente debido a su alta tasa de letalidad y debido a que actualmente no existe una vacuna para este virus. Sin embargo, la escasez de datos epidemiológicos dificulta desarrollar modelos tradicionales capaces de identificar cuáles son las estrategias más efectivas para controlar su propagación.

En este proyecto proponemos utilizar **Reinforcement Learning** para simular el virus y optimizar **Políticas de Intervención No Farmacéutica (NPI)**, con el objetivo de reducir la transmisión del virus y el número de muertes. De esta manera, el proyecto busca ser una herramienta de apoyo para los tomadores de decisiones, dándoles información sobre cuáles podrían ser las políticas de intervención más efectivas para reducir el impacto de este virus altamente mortal sobre el cual aún existe una cantidad de datos muy limitada.

## *Experimento 1:* Simulación local de la transmisión del Hantavirus Andino

<img width="1600" height="830" alt="Pantallazo experimento 1" src="https://github.com/user-attachments/assets/e76da0bc-1d8c-4997-acdd-725745ead148" />

Como primer acercamiento al problema, modelamos la transmisión del virus en un espacio reducido con el fin de analizar cómo se comporta un grupo pequeño de personas y cómo ocurre la propagación del contagio a nivel local. Para ello realizamos un experimento de *Reinforcement Learning* utilizando el algoritmo de optimización **PPO** y a partir de una versión modificada de la librería **ContagionRL**.

## *Experimento 2:* Optimización de Políticas de Intervención No Farmacéutica (NPI) para reducir el impacto del Hantavirus Andino

<img width="1600" height="900" alt="Diagrama experimento 2" src="https://github.com/user-attachments/assets/599a7678-9015-449e-966a-aedf3915d7d2" />

Con este experimento entramos a abordar el objetivo del proyecto: *el agente es un “ente gubernamental”* que ejecuta políticas que se aplican a toda la población, y su objetivo es reducir la transmisión del virus y la cantidad de muertes. Mediante este experimento queremos descubrir cuáles son las mejores políticas gubernamentales para reducir el impacto del virus. Para este experimento de *Reinforcement Learning* utilizamos el algoritmo **PPO** y además realizamos un **Modelado Basado en Agentes (ABM).**
