# rouleur: Cycling performance modelling

Makes the physical modelling of cycling trivially easy.

For example, let's try and estimate the power required for Wiggo's current hour record:

```pycon
>>> from rouleur import CyclingParams, calculate_air_density
>>>
>>> record = 54.526          # km/h
>>> record *= 1000 / 60**2   # m/s
>>> rho = calculate_air_density(30, 777, 0.6)  # about right
>>> pars = CyclingParams(
>>>     rider_velocity=record,
>>>     air_density=rho,
>>>     CdA=0.19, Crr=0.0025, 
>>>     chain_efficiency_factor=0.98,
>>>     road_gradient=0,
>>>     mass_total=82)
>>>     
>>> pars.solve_for.power_output()
440.9565671224358
```

That's all there is to it. 

The API consists almost exclusively of the `CyclingParams` class, which holds all the parameters required for modelling a cyclist. The class constructor combines a number of sensible defaults with any (keyword) arguments passed. Details of recognised keyword arguments---i.e. model parameters---can be found in the class docstring (`help(CyclingParams)`).

Instances then have a number of solver methods accessible via `parameters.solve_for.*`. 

# References

This package is an implementation of a number of published algorithms. Important references are:

1. [Martin JC, Milliken DL, Cobb JE, McFadden KL, Coggan AR. Validation of a Mathematical Model for Road Cycling Power. Journal of Applied Biomechanics 14: 276--291, 1998.](http://journals.humankinetics.com/doi/10.1123/jab.14.3.276)

2. [Martin JC, Gardner AS, Barras M, Martin DT. Modeling sprint cycling using field-derived parameters and forward integration. Med Sci Sports Exerc 38: 592--597, 2006.](https://www.ncbi.nlm.nih.gov/pubmed/16540850)

3. [Atkinson G, Peacock O, Passfield L. Variable versus constant power strategies during cycling time-trials: Prediction of time savings using an up-to-date mathematical model. Journal of Sports Sciences 25: 1001--1009, 2007.](https://www.ncbi.nlm.nih.gov/pubmed/17497402)

4. [Wells MS, Marwood S. Effects of power variation on cycle performance during simulated hilly time-trials. European Journal of Sport Science 16: 912--918, 2016.](https://www.ncbi.nlm.nih.gov/pubmed/26949050)
