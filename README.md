## About

Tool to generate comprehensive multipanel plots to visualize data, data products, best fitting models, and fit quality diagnostics: Figures 2-6, and Figure 8 in Sobolewska et al. 2014, ApJ, 786, 143.

Fermi/LAT lightcurves and files with results of modeling (Kelly, Sobolewska, Siemiginowska 2011) stored in ```data/``` directory (not included in this repository).

## Usage

* Generate any of Figs. 2-6, e.g. for PKS 1633:

```
$ import ou
$ names = ou.keys
$ print names
$ ou.fig_all_fit_examples(names[0])
```

* Generate all Figures 2-6:

```
$ import ou
$ names = ou.keys
$ ou.fig_all_fit_examples(names)
```

* Generate Figure 8:

```
$ import ou
$ names = ou.keys
$ ou.fig_acf(names)
```
