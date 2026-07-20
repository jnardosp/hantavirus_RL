# hantavirus_RL
## Quantitative modeling of Andes Orthohantavirus Spatial Transmission using Reinforcement Learning

Andes orthohantavirus (ANDV) is a highly lethal **(21-50%)** pathogen endemic to South America (mainly Argentina & Chile).
Understanding how it spreads is vital because of its unique biological and epidemiological characteristics:
* **Dual transmission routes:** Different to other hantaviruses that only spread from rodents to humans (zoonotic spillover), ANDV can also transmit directly from **person to person**.
    * **Rodent to human:** Primary virus reservoir is a long-tailed pygmy rice rat (Oligoryzomys longicaudatus). Humans contract the virus by inhaling aerosolized virions from dry rodent excreta (urine, feces, saliva).
    * **Human to human:** Secondary route of virus propagation happens through close, prolonged contact (within 2 meters or so) and is driven by respiratory droplets or saliva, particularly during the early symptomatic phase of the disease.
* **Environmental mechanics:** 
    Risk of adquisition of the virus is incremented in dry, dusty, poorly ventilated environments (through rodent - human route), social behaviour (human - human route) like wearing masks and distancing is crucial to reduce chances of infection.

Because ANDV transmission is heavily influenced by local environmental exposure and social behavior, traditional aggregate epidemiological models fail to capture how individual movements affect infection risk.

To address this we will model these dynamics using an agent-based spatial grid simulation ([ContagionRL](https://openreview.net/forum?id=yPEASsx3hk) paradigm).

## How to run local_behaviour_experiment:
First create the environment:

```bash
conda env create --file local_behaviour_exp/environment.yml
conda activate ContagionRL
```

Clone ContagionRL repo, to use its functionalities.

*Build* files to run **train_modified.py** (version with video rendering bug solved).

In *ContagionRL/figureTraining/* there are some examples of simulations you can adapt them to work with **train_modified.py**, to run them with video rendering use the flag "--record-video".

```bash
#Example
python "train_figure_visibility_changes.py" --record-video
```

You'll find a videos folder inside the new logs folder to see your current agent behaviour evolution.
