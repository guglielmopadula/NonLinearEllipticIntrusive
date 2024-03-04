Tests on a linear parabolic equation
$$-\nabla u+\frac{1}{\mu}e^{\mu u-1}=100*sin(2\pi*x)*sin(2\pi*y)$$
with boundary and initial conditions equal to 0 everywhere. Inspired from [this](https://github.com/RBniCS/RBniCS/blob/master/tutorials/07_nonlinear_elliptictutorial_nonlinear_elliptic_exact.ipynb).

|Method                                     |Train error|Test Error|Time   |
|-------------------------------------------|-----------|----------|-------|
|PODGalerkin(Nmax=20)                       |1.9e-05    |1.9e-05   |4.6e+01|
|PODGalerkinDEIM(Nmax=20,DEIM=21)           |4.2e-03    |4.1e-03   |8.8e+00|
|PODGalerkinDEIMGreedy(Nmax=20,DEIM=21)     |3.2e-05    |3.5e-05   |1.2e+01|
|PODGalerkinEIM(Nmax=20,EIM=21)             |3.7e-05    |4.5e-05   |2.5e+01|
|PODGalerkinEIMGreedy(Nmax=20,EIM=21)       |9.4e-05    |8.2e-05   |2.2e+01|
|Tree                                       |0.0e+00    |7.7e-03   |6.1e-04|
|GPR                                        |2.2e-07    |2.9e-04   |2.3e-03|
|RBF                                        |4.8e-16    |1.7e-04   |7.9e-04|