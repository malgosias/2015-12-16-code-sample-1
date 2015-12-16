Generate comprehensive multipanel plots to visualize data,
data products, best fitting models, and fit quality diagnostics;
Figures 2-6, and Figure 8 in Sobolewska et al. 2014, ApJ, 786, 143.

Directory data/ contains Fermi/LAT lightcurves (ascii) and output files
with modeling results: CAR1 (Kelly et al. 2009), supOU (Kelly,
Sobolewska, Siemiginowska 2011).

1. Generate any of Figs. 2-6, e.g. for PKS 1633:

>>> import ou
>>> names = ou.keys
>>> print names
>>> ou.fig_all_fit_examples(names[0])

2. Generate all Figures 2-6:

>>> import ou
>>> names = ou.keys
>>> ou.fig_all_fit_examples(names)

3. Generate Figure 8:

>>> import ou
>>> names = ou.keys
>>> ou.fig_acf(names)
