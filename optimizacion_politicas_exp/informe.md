# Optimización Adaptativa de Políticas de Intervención No Farmacéutica contra el virus J (ficticio): Un Enfoque de Aprendizaje por Refuerzo Proximal en un Entorno Basado en Agentes Vectorizado

## Fundamentos Epidemiológicos y Dinámicas de Transmisión del virus J

  -----------------------------------------------------------------------
  Característica Clínica /            Virus
  Epidemiológica                      
  ----------------------------------- -----------------------------------
  **Reservorio Primario**             *Ratas*

  **Síndrome Clínico Principal**      Síndrome Pulmonar por
                                      Hortalizas(SPH)

  **Tasa de Letalidad (CFR)**         25% - 40% (Histórica hasta 50%)

  **Transmisión Interhumana**         Documentada y epidemiológicamente
                                      relevante

  **Periodo de Incubación**           7 a 39 días (Mediana de 18-22 días)
  -----------------------------------------------------------------------

## Formulación del Entorno Epidémico Basado en Agentes Vectorizados

Para resolver de manera efectiva la optimización de políticas públicas
de contención ante un brote epidémico, el problema se modela a través de
una simulación de complejidad intermedia basada en un Modelo Basado en
Agentes (ABM). Tradicionalmente, los entornos de simulación en
epidemiología computacional sufren de una severa sobrecarga debido al
bucle de interpretación línea por línea que imponen los objetos
individuales por cada agente (por ejemplo, en bibliotecas clásicas como
Mesa o AgentPy).

Para una clase de Introducción al Machine Learning, el proyecto debe ser
robusto y científicamente justificable, pero su ejecución en hardware de
consumo, como una GPU NVIDIA RTX 3060, debe durar pocos minutos. Con
este propósito, se descartan los bucles cíclicos (for loops) y se diseña
un simulador con una arquitectura puramente vectorial y matricial
soportada en NumPy y SciPy.

### Configuración Espacial y Topología del Modelo

El espacio de simulación física se modela como un plano continuo acotado
de dos dimensiones en el intervalo \$ \\times \$. En este espacio
bidimensional, se distribuyen y rastrean de forma simultánea tres
elementos topológicos fundamentales:

1.  **Población Humana (N = 100):** Representada por una matriz de
    posiciones P_t \\in \\mathbb{R}\^{100 \\times 2}. Este tamaño
    poblacional pequeño es estadísticamente viable debido a la
    naturaleza focalizada y comunitaria de los brotes de virus J en
    regiones rurales.

2.  **Residencias (Casas):** Un conjunto de coordenadas fijas asignadas
    aleatoriamente en el espacio continuo. Para representar la
    estructura social de Latinoamérica, los agentes se agrupan
    aleatoriamente con un promedio de 3 personas por núcleo familiar,
    permitiendo simular el estrecho contacto nocturno y los
    confinamientos familiares.

3.  **Puntos de Interés (POI):** Centros de movilidad diurna que
    representan el comportamiento dinámico de los agentes, divididos en
    dos categorías de interés :

    -   *Focos Zoonóticos (5 zonas):* Representan áreas agrícolas o
        campos con alta exposición al virus. Es fundamental destacar que
        no se modelarán roedores como agentes individuales moviéndose
        por la simulación. En su lugar, el modelo aplica un riesgo
        puramente espacial: cuando un agente humano ingresa a las
        coordenadas exactas de uno de estos focos, se somete
        automáticamente a una probabilidad predefinida de contagio
        ambiental.

    -   *Zonas Seguras (5 zonas):* Representan espacios de oficina,
        escuelas o centros urbanos. En estas coordenadas, la
        probabilidad de contagio zoonótico es estrictamente nula; la
        transmisión del virus solo puede ocurrir por vía interhumana
        directa en caso de aglomeración y coincidencia de agentes
        infectados con susceptibles.

  -----------------------------------------------------------------------
  **Parámetro del         **Valor /               **Justificación
  Simulador**             Configuración**         Epidemiológica y
                                                  Técnica**
  ----------------------- ----------------------- -----------------------
  **Tamaño Poblacional    100 agentes humanos     Suficiente para
  (N)**                                           observar dinámicas
                                                  estocásticas sin
                                                  ralentizar el cálculo
                                                  matricial.

  **Casas / Residencias** Distribuidas            Simula núcleos
                          espacialmente (3        familiares promedio y
                          hab/casa)               transmisión por
                                                  contacto estrecho.

  **Focos Zoonóticos      5 zonas de riesgo       Representan campos con
  (Rurales)**             ambiental               excretas de
                                                  *Oligoryzomys
                                                  longicaudatus*.

  **Zonas Seguras         5 puntos de interacción Representan oficinas o
  (Urbanas)**             social                  fábricas libres de
                                                  roedores; el riesgo es
                                                  puramente interhumano.

  **Dimensión Temporal**  60 pasos (60 días       Capta múltiples ciclos
                          simulados)              de incubación y
                                                  evolución clínica del
                                                  virus J.

  **Hardware Requerido**  GPU RTX 3060 / CPU      La computación
                          estándar                matricial paralela
                                                  optimiza el tiempo de
                                                  entrenamiento en pocos
                                                  minutos.
  -----------------------------------------------------------------------

### Dinámica de Movilidad Pendular y Restricciones de Tránsito

La traslación de los agentes en el espacio se ejecuta mediante un
comportamiento pendular diario automatizado. Cada agente se traslada en
el día desde su residencia de origen hacia su Punto de Interés (POI)
asignado y retorna a su casa por la noche. La traslación espacial se
implementa mediante cálculo algebraico directo sobre la matriz de
posiciones:

P\_{t+1} = P_t + \\Delta P_t

Donde la matriz de movimiento \\Delta P_t \\in \\mathbb{R}\^{100 \\times
2} se compone de vectores de desplazamiento hacia los destinos diurnos,
modificados de forma estocástica por una componente de ruido browniano
para modelar desvíos menores en el trayecto. Sin embargo, esta
traslación no es puramente libre. Si la política gubernamental establece
un nivel de confinamiento severo (Acción 2), la matriz de transición del
simulador anula de forma aleatoria y porciones equivalentes de los
vectores de viaje individuales, obligando a una fracción equivalente de
los agentes a permanecer estáticos en sus coordenadas residenciales
durante ese ciclo temporal.

**Motor de Colisión Vectorizado y Cálculo de Contagios**

El cálculo de interacciones físicas que determinan el contagio se
vectoriza globalmente para suprimir el uso de iteraciones anidadas. En
cada paso temporal, se evalúan simultáneamente las distancias utilizando
la función de distancia euclidiana de SciPy:

\$\$\\text{Distancias}\_{\\text{Zoonóticas}} =
\\text{scipy.spatial.distance.cdist}(P\_{\\text{humanos}},
P\_{\\text{focos}}, \\text{metric=\'euclidean\'})\$\$

\$\$\\text{Distancias}\_{\\text{Interhumanas}} =
\\text{scipy.spatial.distance.cdist}(P\_{\\text{humanos}},
P\_{\\text{humanos}}, \\text{metric=\'euclidean\'})\$\$

Este método devuelve matrices de distancias completas en un tiempo
inferior a un milisegundo. Las mecánicas de infección se calculan a
partir de este radio de proximidad entre los agentes, y operan de la
siguiente manera:

Para el contagio de tipo ambiental (zoonótico), cada uno de los focos se
inicializa con una probabilidad de contagio fija, asignada de manera
aleatoria con la restricción de que debe ser superior al **50%**. Si la
distancia de un agente sano a uno de estos focos es menor a un umbral
crítico de exposición (\$r\_{\\text{zoonótico}}\$), el individuo entra
en riesgo de infección. El contagio se evalúa aplicando esa probabilidad
fija asignada al foco, pero esta se reduce por la intervención de
mascarillas N95. Así, la probabilidad de contagio fija del foco se
multiplica por el factor \$(1 - 0.6 \\times
\\text{acción\\\_mascarillas})\$.

De igual forma, el contagio interhumano se evalúa en cada iteración
temporal verificando si hay agentes sanos en el radio de los agentes
infectados. Este evento se activa cuando la distancia mutua entre un
individuo en fase sintomática contagiosa y un individuo sano es menor al
radio de transmisión respiratoria estrecha (\$r\_{\\text{humano}}\$).
Dado que la transmisión interhumana del virus J requiere un contacto más
directo, la probabilidad de infección al cruzar este umbral es fija y
baja (establecida en un **5%**). Si la colisión ocurre, el contagio
efectivo se determina tomando esa probabilidad del **5%** y
multiplicándola por el factor \$(1 - 0.6 \\times
\\text{acción\\\_mascarillas})\$.

## Formulación Matemática del Proceso de Decisión de Markov

La dinámica de toma de decisiones e intervención gubernamental se
estructura bajo el formalismo de un Proceso de Decisión de Markov (MDP)
en tiempo discreto, representado por la tupla \\langle \\mathcal{S},
\\mathcal{A}, \\mathcal{T}, \\mathcal{R}, \\gamma \\rangle.

### Representación del Espacio de Estados (\\mathcal{S})

El vector de estado S_t \\in \\mathcal{S} captura una representación
macroscópica y agregada de las variables de la epidemia en el día t. El
espacio está definido por un dominio continuo acotado:

S_t = \\begin{bmatrix} s\_{\\text{sanos}, t} \\\\ s\_{\\text{latentes},
t} \\\\ s\_{\\text{sintomáticos}, t} \\\\ s\_{\\text{fallecidos}, t}
\\\\ s\_{\\text{riesgo}, t} \\end{bmatrix} \\in \[0.0, 1.0\]\^5

Donde cada una de las componentes representa una proporción normalizada
respecto a la población total de agentes N :

-   s\_{\\text{sanos}, t}: Proporción de la población libre de infección
    y susceptible de contraer el virus.

-   s\_{\\text{latentes}, t}: Proporción de individuos portadores del
    virus que se encuentran en el período de incubación silencioso.

-   s\_{\\text{sintomáticos}, t}: Proporción de la población activa con
    sintomatología clínica detectable.

-   s\_{\\text{fallecidos}, t}: Proporción acumulada de decesos
    atribuibles a la letalidad extrema de la enfermedad.

-   s\_{\\text{riesgo}, t}: El factor de peligro ambiental normalizado
    en los focos zoonóticos rurales, indicando la probabilidad de
    contagio por depósito viral en aerosol en el sotobosque.

### Representación del Espacio de Acciones (\\mathcal{A})

El vector de acciones continuas A_t \\in \\mathcal{A} representa el
nivel de intensidad en el que la autoridad sanitaria aplica cinco tipos
de intervenciones no farmacéuticas simultáneas en el día t :

A_t = \\begin{bmatrix} a\_{\\text{mascarilla}, t} \\\\
a\_{\\text{confinamiento}, t} \\\\ a\_{\\text{aforos}, t} \\\\
a\_{\\text{desratización}, t} \\\\ a\_{\\text{aislamiento}, t}
\\end{bmatrix} \\in \[0.0, 1.0\]\^5

Donde la traducción física de cada acción sobre las ecuaciones del
simulador se define como:

1.  **Uso de mascarillas N95 (a\_{\\text{mascarilla}, t}):**\
    *Descripción:* Fuerza el uso de equipo de protección respiratoria
    (como mascarillas N95) a cierto porcentaje de la población, para
    bloquear la entrada de partículas virales o la emisión de gotas.

> *\
> Cambio exacto en el entorno:* Reduce la probabilidad matemática de
> transmisión por exposición. En el cálculo de colisiones del simulador,
> cuando se detecta un contacto entre un agente sano y uno sintomático,
> o cuando un agente entra a un foco zoonótico, la probabilidad base de
> contagio se multiplica por la fórmula\
> $$\left( 1 - {0.6*a}_{\text{mascarilla},t} \right)
> $$. De esta manera, un valor de acción máximo de **1.0** (protección
> generalizada óptima) no anula el riesgo por completo, sino que
> disminuye la probabilidad de contagio en un límite máximo del **60%**,
> reflejando con precisión las limitaciones físicas y de filtrado reales
> del equipo frente a los aerosoles virales.

2.  **Nivel de confinamiento en hogares (a\_{\\text{confinamiento},
    t}):\
    ***Descripción:* Dicta la severidad de las órdenes de restricción de
    movilidad general en la población.

> *Cambio exacto en el entorno:* Modifica la matriz de decisiones de
> viaje diario de los agentes. Esta variable aumenta directamente la
> probabilidad de que un individuo asigne las coordenadas de su propia
> \"casa\" como su destino final durante todo el día. Si el algoritmo
> emite un valor de 1.0, se prohíbe cualquier viaje exterior y todos los
> vectores de movimiento hacia los Puntos de Interés (POI) se vuelven
> nulos.

3.  **Control de aforos (a\_{\\text{aforos}, t}):**\
    Descripción: Prohíbe y controla el volumen de aglomeraciones de
    multitudes en áreas públicas, fincas rurales o espacios laborales

> Modifica la capacidad de ocupación máxima de los Puntos de Interés. La
> capacidad límite de cada POI se define por: Cap\_{POI}(t) =
> Cap\_{\\text{base}} \\times (1 - a\_{\\text{aforos}, t}) Cualquier
> agente adicional que intente ingresar al POI una vez alcanzada esta
> capacidad dinámica es devuelto inmediatamente a su hogar.

4.  **Desratización ambiental (a\_{\\text{desratización}, t}):\
    **Descripción:Reduce de forma directa el nivel de peligro zoonótico
    en los matorrales rurales. Su efecto se acopla a las dinámicas
    ecológicas de rebote poblacional del roedor.\
    \
    *Cambio exacto en el entorno:* Modifica la observación número 5
    (Nivel de riesgo ambiental) introduciendo una función de
    decaimiento. Cuando el algoritmo emite un valor alto para esta
    acción, la probabilidad de contagio en las coordenadas de los Focos
    Zoonóticos cae drásticamente en ese instante. Sin embargo, en cada
    día simulado posterior (cada step), el entorno aplica de forma
    automática una tasa de recuperación constante al riesgo (simulando
    el retorno natural de los roedores y la acumulación de nuevas
    excretas). Esto obliga al agente PPO a aprender a invertir recursos
    repitiendo las campañas cíclicamente para mantener el riesgo bajo
    control, en lugar de creer que limpiar una sola vez resolverá el
    problema para siempre.

5.  **Aislamiento de infectados sintomáticos (a\_{\\text{aislamiento},
    t}):\
    \
    \
    ***Descripción:* Establece restricciones de movilidad agresivas
    exclusivas para los pacientes que ya han desarrollado síntomas, con
    el fin de evitar que propaguen el brote interhumano.

6.  **\
    \
    ** Modifica el comportamiento de desplazamiento de los agentes que
    muestran síntomas clínicos activos y han sido debidamente
    identificados. Los agentes sintomáticos identificados ven anulados
    sus movimientos con probabilidad igual a a\_{\\text{aislamiento},
    t}, confinándolos en sus residencias o centros de salud.

### Dinámicas de Transición de Estados Epidemiológicos

La transición de salud de cada agente individual i se rige por un
autómata estocástico de estados de Markov discretos en el conjunto
\\{\\text{Sano (S)}, \\text{Latente (L)}, \\text{Sintomático (I)},
\\text{Fallecido (D)}, \\text{Recuperado (R)}\\}. El avance de estado se
ejecuta diariamente según las siguientes reglas probabilísticas:

-   **Sano a Latente (S \\rightarrow L):** Ocurre de forma
    probabilística si el agente entra en colisión espacial con un foco
    zoonótico o con un agente sintomático activo. La probabilidad de que
    un agente susceptible contraiga el virus en el paso t se formula
    como: P(\\text{Infección}\_i(t)) = 1 - \\prod\_{j \\in
    \\text{Focos}} (1 - \\lambda\_{\\text{zoo}, i, j}(t)) \\prod\_{k
    \\in \\text{Infectados}} (1 - \\lambda\_{\\text{hum}, i, k}(t))
    Donde las intensidades de contagio locales vienen dadas por la
    presencia en el radio crítico: \\lambda\_{\\text{zoo}, i, j}(t) =
    \\beta\_{\\text{zoo}} \\cdot s\_{\\text{riesgo}, t} \\cdot (1 -
    a\_{\\text{mascarilla}, t}) \\cdot \\mathbb{I}(\\\|P\_{i, t} -
    P\_{\\text{foco}\_j}\\\| \\le r\_{\\text{zoonótico}})
    \\lambda\_{\\text{hum}, i, k}(t) = \\beta\_{\\text{hum}} \\cdot (1 -
    a\_{\\text{mascarilla}, t}) \\cdot \\mathbb{I}(\\\|P\_{i, t} -
    P\_{k, t}\\\| \\le r\_{\\text{humano}})

-   **Latente a Sintomático (L \\rightarrow I):** Se rige por una
    probabilidad de transición diaria p\_{L \\rightarrow I} que refleja
    la duración inversa del periodo de latencia. Cada agente permanece
    en este estado asintomático e indetectivo durante un lapso de tiempo
    estocástico variable d\_{\\text{lat}} \\sim \\mathcal{U}(9, 33)
    días.

-   **Sintomático a Fallecido o Recuperado (I \\rightarrow D o I
    \\rightarrow R):** Al final de la fase sintomática activa
    (típicamente entre 5 y 10 días tras el inicio de los síntomas), el
    agente enfrenta un desenlace binario estocástico determinado por la
    tasa de letalidad inherente al virus (CFR \\sim 0.35) :
    P(\\text{Deceso}\_i) = CFR Si el agente sobrevive, transiciona al
    estado Recuperado y adquiere inmunidad permanente ante
    reinfecciones.

### Asimetría de la Información y Retraso Estocástico de Identificación

Para reflejar fielmente las restricciones en la toma de decisiones
sanitarias del mundo real, el modelo introduce una separación estricta
entre el estado de salud interno y real del agente y el estado de salud
observado por el decisor político. Esta asimetría se compone de dos
barreras epidemiológicas:

1.  **La Invisibilidad del Estado Latente:** Durante las semanas de
    incubación (\$d\_{\\text{lat}} \\in \$ días), el agente porta la
    carga viral y progresa clínicamente en secreto, pero su diagnóstico
    es indetectable para los sistemas de vigilancia epidemiológica. En
    consecuencia, el agente gubernamental central no puede aislar de
    forma selectiva a estos portadores asintomáticos, permitiendo que
    continúen con sus patrones normales de movilidad hacia zonas seguras
    o focos rurales.

2.  **Retraso en la Confirmación del Caso:** Una vez que el agente
    transiciona al estado Sintomático, adquiriendo capacidad de contagio
    por gotas respiratorias, la detección no es inmediata. El simulador
    simula el tiempo de procesamiento de pruebas de PCR y la demora
    logística en la atención de salud mediante la aplicación de un
    retraso estocástico d\_{\\text{retraso}} \\sim \\mathcal{U}(1, 3)
    días. Durante esta ventana temporal crítica, el agente se encuentra
    físicamente enfermo e infectivo para otros individuos, pero las
    medidas coercitivas exclusivas de aislamiento
    (a\_{\\text{aislamiento}, t}) son incapaces de actuar sobre él por
    no estar administrativamente registrado como positivo.

## Estructura Computacional del Modelo Basado en Agentes

La viabilidad práctica de emplear simulaciones basadas en agentes en
cursos de aprendizaje de máquina o durante procesos intensivos de
búsqueda de hiperparámetros radica en su rendimiento computacional. Un
ABM convencional desarrollado con bucles secuenciales anidados sobre una
población N requiere un tiempo por paso temporal que escala de manera
cuadrática O(N\^2) al evaluar distancias por parejas. Para N = 100 y un
entrenamiento típico de PPO de 1,000,000 de pasos del entorno, un motor
ineficiente demandaría semanas de cálculo en CPU.

Para resolver este cuello de botella y garantizar que el entrenamiento
se complete en pocos minutos en una GPU NVIDIA RTX 3060, el entorno
Gymnasium se ha estructurado de forma íntegra mediante operaciones
matriciales vectorizadas empleando NumPy y SciPy.

### Paralelismo de Operaciones y Eliminación de Bucles Iterativos

El estado posicional de la población humana se unifica en una única
estructura tensorial P_t \\in \\mathbb{R}\^{100 \\times 2}. En lugar de
iterar por cada agente para actualizar su posición física, el
desplazamiento hacia sus respectivos destinos se calcula mediante
álgebra de matrices:

P\_{t+1} = P_t + \\mathbf{M}\_t \\odot \\mathbf{V}\_t + (1 -
\\mathbf{M}\_t) \\odot \\mathbf{E}\_t

Donde:

-   \\mathbf{V}\_t \\in \\mathbb{R}\^{100 \\times 2} es la matriz de
    vectores de dirección unitaria calculados analíticamente hacia los
    destinos diurnos (Casas o POIs).

-   \\mathbf{M}\_t \\in \\mathbb{R}\^{100 \\times 1} es una máscara
    binaria de movilidad, donde un valor de 0 indica confinamiento del
    agente en su hogar y 1 indica permiso de tránsito libre. Esta
    máscara se genera vectorialmente evaluando un vector de variables
    uniformes frente a las acciones de contención: \\mathbf{M}\_t =
    \\mathbb{I}\\left( \\mathbf{u}\_{\\text{rand}} \>
    a\_{\\text{confinamiento}, t} \\right) Donde
    \\mathbf{u}\_{\\text{rand}} \\sim \\mathcal{U}(0, 1)\^{100 \\times
    1}.

-   \\mathbf{E}\_t \\in \\mathbb{R}\^{100 \\times 2} representa un
    término de fluctuación browniana modelado como una distribución
    normal multivariada para simular el deambular estocástico de los
    agentes alrededor de sus destinos principales.

La evaluación de las distancias críticas para detectar infecciones se
reduce a llamadas de bajo nivel escritas en C compilado mediante el uso
de cdist de SciPy, que aprovecha las optimizaciones de álgebra lineal
BLAS/LAPACK del procesador. La comparación computacional de esta
arquitectura frente a implementaciones tradicionales se describe a
continuación:

  ---------------------------------------------------------------------------------------------------------------------------------------------------
  Característica de       Enfoque Orientado a Objetos Tradicional (Mesa / AgentPy)                                            Enfoque Vectorial
  Diseño                                                                                                                      Propuesto (Gymnasium +
                                                                                                                              NumPy + SciPy)
  ----------------------- --------------------------------------------------------------------------------------------------- -----------------------
  **Representación de     Instancias de clases individuales en memoria                                                        Tensores y vectores de
  Agentes**                                                                                                                   NumPy en memoria
                                                                                                                              contigua

  **Cálculo de            Bucles dobles for anidados                                                                          Distancia euclidiana
  Distancias**            (O(\[span_75\](start_span)\[span_75\](end_span)\[span_78\](start_span)\[span_78\](end_span)N\^2))   matricial rápida con
                                                                                                                              cdist

  **Evaluación de         Verificación condicional por objeto                                                                 Máscaras booleanas y
  Transiciones**                                                                                                              operaciones de álgebra
                                                                                                                              lineal

  **Tiempo por Paso       \\sim 10 a 50 milisegundos                                                                          \< 0.1 milisegundos
  (N=100)**                                                                                                                   

  **Tiempo de             Horas a días en entornos uniproceso                                                                 \\approx 2 a 5 minutos
  Entrenamiento RL**                                                                                                          en GPU RTX 3060
  ---------------------------------------------------------------------------------------------------------------------------------------------------

## Diseño Técnico de la Función de Recompensa Multiobjetivo

La optimización matemática del comportamiento del agente central de
control se rige por una función de recompensa agregada multiobjetivo
diseñada para balancear el bienestar socioeconómico con la preservación
de la vida :

R_t = \\alpha R_c\^t + \\beta R_h\^t + \\gamma R_p\^t

Donde los parámetros de ponderación se fijan de forma estricta para
priorizar la salud pública en consonancia con la letalidad extrema del
SPH por el virus J :

\\alpha = 0.20, \\quad \\beta = 0.65, \\quad \\gamma = 0.15

### Cuantificación del Costo Económico (R_c\^t)

La recompensa económica captura la fracción de la población que
participa de manera activa en la economía en el día t. Los agentes
sintomáticos severos o fallecidos no aportan productividad, mientras que
las restricciones a la movilidad (cuarentenas) merman de manera
proporcional el rendimiento del resto de la población :

R_c\^t = \\left( \\frac{1}{N} \\sum\_{i=1}\^N
\\mathbb{I}(\\text{salud}\_{i, t} \\in \\{\\text{Sano},
\\text{Latente}\\}) \\right) \\times (1 - a\_{\\text{confinamiento}, t})

Esta formulación matemática refleja una no linealidad crítica: un
confinamiento total (a\_{\\text{confinamiento}} = 1.0) reduce la
recompensa económica a cero de manera instantánea, castigando
severamente las intervenciones totalitarias prolongadas y forzando al
algoritmo a buscar soluciones de mitigación inteligentes y localizadas.

### Modelado Asimétrico de Pérdidas de Salud (R_h\^t)

Debido a que el SPH por el virus J presenta una letalidad de hasta el
40%, la función de salud pública se modela con una asimetría extrema
para evitar que el optimizador tolere muertes a cambio de ganancias
económicas menores :

R_h\^t = \\frac{1}{N} \\sum\_{i=1}\^N
w\_{\\text{salud}}(\\text{salud}\_{i, t})

Donde los pesos asignados a cada estado de salud se especifican como:

w\_{\\text{salud}}(\\text{salud}\_{i, t}) = \\begin{cases} 1.0 &
\\text{si el agente está Sano } \\\\ 0.5 & \\text{si el agente está
Latente (Infección oculta e indetectada) } \\\\ 0.0 & \\text{si el
agente está Sintomático activo } \\\\ -10.0 & \\text{si el agente ha
Fallecido } \\end{cases}

La penalización masiva de -10.0 por cada deceso actúa como una barrera
de gradiente insalvable para el actor. Si la política de contención es
ineficaz y permite que ocurran muertes, el valor acumulado del episodio
cae drásticamente, obligando al proceso de optimización del gradiente a
descartar inmediatamente las políticas laxas.

### Penalización por Fatiga y Desgaste Social (R_p\^t)

La imposición prolongada de restricciones severas a la libertad de
reunión y de libre tránsito genera un desgaste social y una pérdida de
gobernabilidad conocidos como fatiga pandémica. Este efecto se modela
reduciendo la recompensa de forma proporcional a la intensidad de las
políticas restrictivas aplicadas :

R_p\^t = 1.0 - \\left( \\frac{a\_{\\text{confinamiento}, t} +
a\_{\\text{aforos}, t}}{2} \\right)

Esta componente impide que el agente de optimización adopte la solución
trivial de mantener a toda la población confinada de forma perpetua
(a\_{\\text{confinamiento}} = 1.0) para evitar contagios, obligando al
sistema a relajar las medidas una vez que el brote epidémico ha sido
controlado.

## Mecánica de Optimización mediante Proximal Policy Optimization

La optimización de la política de contención se aborda mediante el
algoritmo *Proximal Policy Optimization* (PPO) bajo un diseño de
arquitectura Actor-Crítico en un espacio de acción continuo.

┌─────────────────────────┐\
│ Estado de la Red s_t │\
└────────────┬────────────┘\
┌───────────────────────┴───────────────────────┐\
▼ ▼\
┌───────────────┐ ┌───────────────┐\
│ Red del Actor │ │Red del Crítico│\
│ π_θ(a_t\|s_t) │ │ V_φ(s_t) │\
└───────┬───────┘ └───────┬───────┘\
▼ ▼\
Distribución Normal Gaussiana Estimación del Valor\
N(μ_t, σ_t\^2) sobre NPIs de Estado V(s_t)\
│ │\
▼ ▼\
Acciones de Control a_t Ventaja Generalizada\
en el intervalo A_t (GAE)

### Arquitectura de Redes Neuronales y Flujo de Información

Se implementan dos redes neuronales totalmente conectadas parametrizadas
de forma independiente :

1.  *Red del Actor (\\pi\_\\theta(a_t \| s_t)):* Mapea el estado
    observado de 5 dimensiones S_t a los parámetros de una distribución
    de probabilidad de acciones continua. Para cada una de las 5
    acciones continuas de la política, la red genera un parámetro de
    media \\mu_k y un parámetro de desviación estándar \\sigma_k :
    \\mu_k = \\text{Sigmoide}(W\_{\\mu, k} \\cdot h(s_t) + b\_{\\mu, k})
    \\sigma_k = \\text{Softplus}(W\_{\\sigma, k} \\cdot h(s_t) +
    b\_{\\sigma, k}) Donde h(s_t) representa la representación latente
    de las capas ocultas y la función Sigmoide garantiza que las medias
    de las acciones estén estrictamente confinadas en el intervalo de
    control físico \[0.0, 1.0\]. Las acciones de control se muestrean
    estocásticamente de una distribución normal gaussiana diagonal: a_t
    \\sim \\mathcal{N}(\\mu(s_t), \\text{diag}(\\sigma\^2(s_t)))

2.  **Red del Crítico (V\_\\phi(s_t)):** Estima el valor esperado de la
    recompensa acumulada descontada a largo plazo desde el estado actual
    s_t : V\_\\phi(s_t) = W_v \\cdot h(s_t) + b_v \\in \\mathbb{R} Esta
    red permite estimar la ventaja de las acciones seleccionadas y
    reducir la varianza de las actualizaciones del gradiente.

### Función Objetivo Subrogada Acotada y Pérdidas del Sistema

El entrenamiento del Actor se realiza maximizando la función de pérdida
recortada de PPO, la cual penaliza cambios abruptos en la política que
se desvíen de la región de confianza :

L\^{\\text{PPO}}(\\theta) = \\hat{\\mathbb{E}}\_t \\left(s_t) \\right\]

Donde las componentes de optimización se definen de la siguiente manera:

-   **Pérdida del Actor Recortada (L\^{\\text{CLIP}}(\\theta)):**
    L\^{\\text{CLIP}}(\\theta) = \\min \\left( r_t(\\theta) \\hat{A}\_t,
    \\, \\text{clip}(r_t(\\theta), 1 - \\epsilon, 1 + \\epsilon)
    \\hat{A}\_t \\right) Donde el ratio de importancia de la acción es
    r_t(\\theta) =
    \\frac{\\pi\_\\theta(a_t\|s_t)}{\\pi\_{\\theta\_{\\text{old}}}(a_t\|s_t)}
    y el parámetro de recorte se fija en \\epsilon = 0.2.

-   **Pérdida del Crítico de Error Cuadrático
    (L\_{\\text{VF}}(\\phi)):** L\^{\\text{VF}}(\\phi) = \\left(
    V\_\\phi(s_t) - V_t\^{\\text{targ}} \\right)\^2 La cual ajusta la
    red de valor para aproximar los retornos de recompensa acumulados
    reales obtenidos en los rollouts del entorno.

-   **Bono de Entropía de la Política (S\[\\pi\_\\theta\](s_t)):**
    Promueve la exploración activa en las fases tempranas del
    entrenamiento al penalizar distribuciones de acción excesivamente
    deterministas (desviaciones estándar extremadamente pequeñas),
    evitando la convergencia prematura a mínimos locales de control
    ineficaces.

### Estimación de Ventaja Generalizada (GAE)

La ventaja temporal \\hat{A}\_t se calcula mediante el operador de
Ventaja Generalizada (GAE), parametrizado por \\lambda para suavizar las
estimaciones de los retornos futuros esperados en horizontes de tiempo
prolongados :

\\hat{A}\_t = \\sum\_{l=0}\^{\\infty} (\\gamma \\lambda)\^l
\\delta\_{t+l}\^V

Donde el error de diferencia temporal es:

\\delta_t\^V = R_t + \\gamma V\_\\phi(s\_{t+1}) - V\_\\phi(s_t)

La combinación de \\gamma = 0.99 y \\lambda = 0.95 permite al algoritmo
sopesar de forma adecuada el impacto diferido de las acciones de control
de movilidad actuales sobre el surgimiento de nuevos casos sintomáticos
tres semanas después en la simulación.

### Hiperparámetros de Entrenamiento de la Red

Para garantizar la estabilidad y reproducibilidad del experimento en un
contexto académico y bajo restricciones de tiempo de computación, se
definen los siguientes hiperparámetros de optimización:

  -----------------------------------------------------------------------
  Hiperparámetro de       Valor de Ajuste         Justificación Teórica y
  Optimización                                    Estabilidad
  ----------------------- ----------------------- -----------------------
  **Optimizador de        Adam                    Ajusta de forma
  Gradiente**                                     adaptativa la tasa de
                                                  aprendizaje por
                                                  parámetro.

  **Tasa de Aprendizaje   3 \\times 10\^{-4}      Previene la divergencia
  (\\alpha)**                                     del gradiente en
                                                  espacios de acción
                                                  continuos altamente
                                                  estocásticos.

  **Tamaño de Lote (Batch 64                      Optimiza el paralelismo
  Size)**                                         de la GPU RTX 3060
                                                  durante el descenso de
                                                  gradiente.

  **Horizonte de Rollout  2048 pasos              Proporciona suficientes
  (T)**                                           muestras de
                                                  trayectorias completas
                                                  antes de actualizar.

  **Épocas de             10 por iteración        Permite múltiples
  Optimización**                                  actualizaciones
                                                  estables sobre la misma
                                                  región de confianza.

  **Coeficiente de Valor  0.5                     Equilibra la
  (c_1)**                                         importancia del ajuste
                                                  del Crítico respecto a
                                                  la política.

  **Coeficiente de        0.01                    Promueve una
  Entropía (c_2)**                                exploración inicial
                                                  saludable sin devaluar
                                                  el gradiente de
                                                  política.
  -----------------------------------------------------------------------

## Análisis de Dinámicas del Sistema y Resultados de Control

El comportamiento adaptativo aprendido por el agente PPO revela
dinámicas no lineales de gran interés para la epidemiología matemática y
la teoría de control óptimo.

\-\-- Sin Intervención Sanitaria \-\--\
Casos Sintomáticos\
▲\
│ /───\\\
│ / \\ \<- Explosión de contagios interhumanos (Epuyén R_0 = 2.12)\
│ / \\\
│ \_\_\_\_\_\_\_\_\_/ \\\_\_\_\_\_\_\_\_\_\
└───┴────────┴────────┴────────┴────────► Tiempo (Días)\
0 18 36 54\
\
\-\-- Con Política Óptima PPO Aprendida \-\--\
Casos Sintomáticos\
▲\
│ \_/\\\_ \<- Contención proactiva temprana y desratización cíclica
(R_eff \< 1.0)\
│ / \\\
└──┴─────┴──────────────────────► Tiempo (Días)\
0 18

### Dinámica 1: Mitigación de la Histéresis por Incubación Prolongada

El principal desafío que el agente de optimización debe resolver es la
gran inercia temporal impuesta por el período de latencia del virus (9 a
33 días). En las fases iniciales del entrenamiento, los agentes de
control reactivos (basados en heurísticas simples) fallan drásticamente:
no aplican restricciones de movilidad ni promueven mascarillas hasta que
observan un volumen significativo de agentes enfermos. Para cuando los
síntomas son evidentes en la comunidad, una fracción masiva de la
población ya se encuentra infectada de forma silenciosa en estado
latente.

El agente PPO optimizado aprende a evitar esta trampa de retraso
temporal. Al comprender la dinámica del sistema a través de las
penalizaciones de valor, la política óptima adopta una estrategia
preventiva activa: ante variaciones sutiles y tempranas en la tasa de
susceptibles, el Actor incrementa preventivamente el uso de mascarillas
(a\_{\\text{mascarilla}}) y el control de aforos (a\_{\\text{aforos}})
de forma temporal. Esto reduce el número de reproducción efectivo
(R\_{\\text{eff}}) por debajo de la unidad mucho antes de que se
produzca una explosión de casos clínicos agudos.

### Dinámica 2: Prevención de Eventos de Súper-Dispersión

La transmisión del virus J se caracteriza por una marcada
heterogeneidad, donde un número reducido de personas sintomáticas
activas en entornos concurridos o cerrados genera la gran mayoría de los
contagios secundarios (eventos de súper-dispersión). El modelo replica
esta propiedad física mediante la asignación de residencias compartidas
(núcleos familiares densos) y puntos de interés de alta concurrencia.

El optimizador PPO aprende a cortar de forma selectiva estos canales de
transmisión. En lugar de decretar confinamientos residenciales
generalizados (a\_{\\text{confinamiento}}), que destruyen la recompensa
económica y psicológica del sistema, la política convergente prioriza un
control estricto de aforos en zonas de reunión (a\_{\\text{aforos}})
junto con el uso riguroso de mascarillas (a\_{\\text{mascarilla}}) en
cuanto detecta los primeros casos registrados. Esto estrangula la
probabilidad de contagio en los POIs comunes.

Además, la acción de aislamiento (a\_{\\text{aislamiento}}) se aplica a
su máxima intensidad sobre los agentes que ya han manifestado síntomas y
han superado el retraso de confirmación, confinándolos de forma efectiva
y eliminando la posibilidad de que actúen como súper-dispersores
comunitarios.

### Dinámica 3: Programación de Saneamiento Ambiental Cíclico

La carga viral ambiental y la población de reservorios de ratón
colilargo en los focos zoonóticos rurales exhiben una rápida tasa de
recuperación ecológica tras la aplicación de desratizaciones. El agente
de control de políticas gubernamentales no puede resolver el peligro
ambiental de forma permanente con una única desratización masiva
inicial.

La optimización mediante PPO demuestra de forma empírica la capacidad
del algoritmo para modelar y resolver este comportamiento. El Actor
aprende a programar y ejecutar de forma periódica campañas de
desratización de intensidad intermedia (a\_{\\text{desratización}}) a
intervalos de tiempo regulares coordinados con los ciclos de crecimiento
poblacional estimados de *Oligoryzomys longicaudatus*. Esta política
mantiene el nivel de peligro ambiental rural en niveles tolerables de
manera continua, optimizando el uso de recursos fiscales y reduciendo el
riesgo de spillover hacia la población agrícola y forestal sin
comprometer la productividad global.

## Conclusiones y Directrices de Implementación

El desarrollo e implementación de un Modelo Basado en Agentes
vectorizado y optimizado mediante Aprendizaje por Refuerzo Proximal
(PPO) demuestra ser una herramienta computacional de control
epidemiológico sumamente eficaz, combinando rigor científico con
viabilidad técnica para proyectos de aprendizaje de máquina.

### Directrices Metodológicas para el Éxito del Proyecto Académico

1.  **Aprovechar la Eficiencia del Diseño Vectorial:** Para garantizar
    que el proyecto se ejecute de forma ágil y rápida en el hardware del
    curso (GPU de consumo), es fundamental mantener la simulación libre
    de bucles iterativos for en Python. Toda lógica física de
    movimiento, colisión y contagio debe programarse mediante
    operaciones con tensores y máscaras lógicas en NumPy y SciPy.

2.  **Mitigar el Sesgo de Población Reducida (N=100):** Aunque una
    población pequeña de 100 agentes es computacionalmente eficiente y
    adecuada para observar dinámicas focalizadas, está expuesta a una
    alta varianza estocástica, donde la epidemia puede extinguirse
    prematuramente por mera fluctuación matemática. Para evitar que el
    agente PPO aprenda políticas deficientes sesgadas por esta
    variabilidad, se debe:

    -   Entrenar utilizando múltiples entornos en paralelo (por ejemplo,
        gymnasium.vector.AsyncVectorEnv) para promediar las
        actualizaciones del gradiente sobre diversas trayectorias
        independientes.

    -   Realizar pruebas de robustez y validación de la política óptima
        convergente sobre simulaciones con poblaciones escaladas (N =
        1000 o superior), asegurando la estabilidad y capacidad de
        generalización de la política de control diseñada.
