'''
    
   Generate comprehensive multipanel plots to visualize data,
   data products, best fitting models, and fit quality diagnostics;
   Figures 2-6, and Figure 8 in Sobolewska et al. 2014, ApJ, 786, 143.

'''

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
                              ScalarFormatter

keys = ['pks1633', 'pks1424', 'b21520', 'pks0454', '3c454', '3c279',
        'pks1510', '3c273', '3c66a', 's50716', 'pks2155', 'bllac',
        'mrk421']

flux_yscale = [2, 1, 2, 4, 4, 2, 1, 1, 1, 2, 2, 1, 2]

flux_ylabel = ['10', '9', '10', '10', '5', '10', '5', '9', '6', '10',
               '7', '9', '10']

name_label = ['B2 1633+38', 'PKS 1424-41', 'B2 1520+31', 'PKS 0454-234',
              '3C 454.3', '3C 279', 'PKS 1510-089', '3C 273', '3C 66A',
              'PKS 0716+714', 'PKS 2155-304', 'BL Lac', 'Mkn 421']

dict1 = {}
dict2 = {}
dict3 = {}
for i, key in enumerate(keys):
    dict1[key] = flux_yscale[i]
    dict2[key] = flux_ylabel[i]
    dict3[key] = name_label[i]


def fig_all_fit_examples(names):
# Generate all Figures (2-6) in Sobolewska et al. (2014)
    for name in names:
        file_list = get_file_names(name)
        fig_fit_example(name)
    return


def fig_fit_example(source):
# Generate one of Figs. 2-6, for a given source, Sobolewska et al. (2014)

    fig = plt.figure(figsize=(9.0,4.2))
    
    factor = np.double("1e-" + dict2[source])

    mL1 = MultipleLocator(500)
    mL2 = MultipleLocator(dict1[source])
    mL3 = MultipleLocator(2)
    mL4 = MultipleLocator(4)
    mL5 = MultipleLocator(2)
    mL6 = MultipleLocator(0.2)
    
    # Anti-clockwise from upper left
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.09, right=0.50, top=0.96, bottom=0.70, wspace=0.05)
    gs2 = gridspec.GridSpec(2, 1)
    gs2.update(left=0.09, right=0.50, top=0.64, bottom=0.12, hspace=0.05)
    gs3 = gridspec.GridSpec(2, 1)
    gs3.update(left=0.58, right=0.99, top=0.64, bottom=0.12, hspace=0.05)
    gs4 = gridspec.GridSpec(1, 2)
    gs4.update(left=0.58, right=0.99, top=0.96, bottom=0.75, wspace=0.25)
    
    # Read parameters of OU and supOU processes fitting the lightcurves
    file_list = get_file_names(source)

    # params = [t, dt, y, yerr, a, logpost, yhat, chi, mumax]
    params = read_car1_lc(file_list[:3])
    
    yh_supou, yhvar = read_yhat(file_list[3])
    chi_supou = ( yh_supou-np.log(params[2]) ) / np.sqrt(yhvar)
    mumax_supou = 0
    
    # lightcurve linear
    ax1 = plt.subplot(  gs1[0,0] )
    plot_lc_lin(params[0], params[2]/factor)
    plt.text(0.05, 0.82, dict3[source], fontsize=11, \
                                            transform=ax1.transAxes)
    ax1.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
    plt.ylabel("Flux / 10$^{-" + dict2[source] +
               "}$\nph cm$^{-2}$ s$^{-1}$")
    ax1.yaxis.set_major_locator(mL2)
    ax1.yaxis.set_label_coords(-0.067, 0.5)
               
    # lightcurve+model ou
    ax2 = plt.subplot( gs2[0, 0] )
    plot_lc_yh(params[0] , params[2], params[3], params[6], params[8])
    plt.ylabel('Ln(Flux)')
    ax2.yaxis.set_major_locator(mL3)
    ax2.yaxis.set_label_coords(-0.12, 0.5)
    plt.text(0.85, 0.1, 'OU', fontsize=11, transform=ax2.transAxes)
               
    # residuals ou
    ax3 = plt.subplot( gs2[1, 0] )
    plot_chi(params[0], params[7])
    plt.ylim(-6, 7.9)
    ax3.yaxis.set_major_locator(mL4)
    ax3.yaxis.set_label_coords(-0.12, 0.5)
    
    # lightcurve+model supou
    ax4 = plt.subplot( gs3[0, 0] )
    plot_lc_yh(params[0], params[2], params[3], yh_supou, mumax_supou)
    plt.ylabel('Ln(Flux)')
    ax4.yaxis.set_major_locator(mL3)
    ax4.yaxis.set_label_coords(-0.12, 0.5)
    plt.text(0.78, 0.1, 'sup-OU', fontsize=11, transform=ax4.transAxes)
        
    # residuals supou
    ax5 = plt.subplot( gs3[1, 0] )
    plot_chi(params[0], chi_supou)
    plt.ylim(-6, 7.9)
    ax5.yaxis.set_major_locator(mL4)
    ax5.yaxis.set_label_coords(-0.12, 0.5)
 
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_locator(mL1)
        ax.set_xlim([54600, 56400])

    # histograms of residuals
    x = np.linspace(-6, 8, 50)
    norm = sp.stats.norm.pdf(x)
    n = 30
    
    # ou
    ax6 = plt.subplot( gs4[0, 0] )
    plot_hist_prob(params[7], n, 'Residuals')
    ax6.set_ylabel('Probability')
    plt.text(0.05, 0.82, 'OU', fontsize=11, transform=ax6.transAxes)

    # supou
    ax7 = plt.subplot( gs4[0, 1] )
    plot_hist_prob(chi_supou, n, 'Residuals')
    ax7.set_ylabel('')
    plt.text(0.05, 0.82, 'sup-OU', fontsize=11, transform=ax7.transAxes)

    for ax in [ax6, ax7]:
        ax.xaxis.set_major_locator(mL5)
        ax.yaxis.set_major_locator(mL6)
        ax.plot(x, norm)

    return


def fig_acf(names):
# Generate Figure 8 from Sobolewska et al. (2014)
    
    mL1 = MultipleLocator(250)
    mL2 = MultipleLocator(0.2)
    
    fig = plt.figure(figsize=(8.5,11.3))
    
    n = np.size(names)
    for i in range(n):
        files = get_file_names(names[i])
        # parset_car1 equals:
        # [t, dt, y, yerr, a, logpost, yhat, chi, mumax]
        parset = read_car1_lc(files[:3])
        y = parset[2]
        chi_ou = parset[7]
        yh_supou, yhvar = read_yhat(files[3])
        chi_supou = ( yh_supou-np.log(y) ) / np.sqrt(yhvar)
        
        gs = gridspec.GridSpec(2, 1)
        gs.update(left=0.33*(i % 3)+0.08, right=0.33*(i % 3)+0.33, \
                  top=0.99-0.20*int(i/3.), bottom=0.84-0.20*int(i/3.), \
                  hspace=0.05)
        
        ax = plt.subplot( gs[0:2, 0:1] )
        ax.set_ylabel("ACF of residuals")
        
        # upper panel
        ax1 = plt.subplot( gs[0, 0] )
        plot_acf(chi_ou, ylabel=False)
        plt.text(0.1, 0.8, dict3[names[i]], fontsize=13, \
                                            transform=ax1.transAxes)
        plt.text(0.83, 0.8, 'OU', fontsize=13, transform=ax1.transAxes)
        ax1.set_ylabel("ACF of residuals")
        ax1.yaxis.set_label_coords(-0.21, -0.1)
        ax1.xaxis.set_ticklabels([])

        # lower panel
        ax2 = plt.subplot( gs[1, 0] )
        plot_acf(chi_supou, ylabel=False)
        plt.text(0.66, 0.8, 'sup-OU', fontsize=13, \
                                            transform=ax2.transAxes)
 
        if (i in [2, 4, 5, 6, 9]):
            x_upper = 990.0
        else:
            x_upper = plt.xlim()[1]

        if (i==0):
            y_upper = 0.39
        else:
            y_upper = 0.29

        for ax in [ax1, ax2]:
            ax.set_xlim(-20, x_upper)
            ax.set_ylim(-0.2, y_upper)
            ax.xaxis.set_major_locator(mL1)
            ax.yaxis.set_major_locator(mL2)

    return


# -------- read data ------------

def read_car1_lc(file_list):
    t, dt, y, yerr = read_lc(file_list[0])
    mu, var, a, b, sigma, logpost, yhat, omega = read_car1(file_list[1:])
    mumax = mu[np.argmax(logpost)]
    chi = (yhat + mumax - np.log(y)) / np.sqrt(omega + (yerr/y)**2)
    return t, dt, y, yerr, a, logpost, yhat, chi, mumax


def read_lc(fname):
    # read lightcurve from file fname
    d = np.loadtxt(fname)
    t, dt, y, yerr = d.T[:4]
    return t, dt, y, yerr


def read_car1(file_list):
    mu, var, a, b, sigma, logpost = read_car1_post(file_list[0])
    yhat, omega = read_yhat(file_list[1])
    return mu, var, a, b, sigma, logpost, yhat, omega


def read_car1_post(fname):
    d = np.loadtxt(fname) # post
    mu, var, a, b, sigma, logpost = d.T[:]
    return mu, var, a, b, sigma, logpost


def read_yhat(fname):
    d = np.loadtxt(fname)
    # yhat, yhvar (or omega)
    yhat, yhvar = d.T[:]
    yhvar = d[:,1]
    return yhat, yhvar


# -------- individual plots ------------

def plot_acf(a, resids="", ylabel=True):
    acf_result = acf(a)
    lag = np.arange(len(acf_result) - 1) + 1
    plt.bar(lag, acf_result[1:])
    cl95 = 1.96 / np.sqrt(len(acf_result))
    plt.hlines([cl95, -cl95], 0, len(a), color='r', linestyle='--')
    plt.hlines([0], 0, len(a), color='r')
    if (ylabel):
        plt.ylabel('ACF of '+resids+'residuals')
    plt.xlabel('Time Lag')
    return

def plot_lc_lin(t, y):
    plt.plot(t, y, 'k-')
    plt.xlabel("Time [ days ]")
    plt.ylabel("Flux [ ph / (s cm^2) ]")
    return


def plot_lc_yh(t, y, yerr, yhat, mu):
    plt.errorbar(t, np.log(y), yerr=yerr/y, marker='o', mfc='none',
                 ms=6, capsize=0, ecolor='black', ls='None')
    plt.plot(t,yhat+mu,'r-')
    plt.xlabel("Time [ days ]")
    plt.ylabel("Ln(Flux) [ ph / (s cm^2) ]")
    return


def plot_chi(t, chi, facecol='none'):
    plt.plot(t, chi, marker='o', mfc=facecol, mec='black', ms=6,
                                                ls='None', zorder=1)
    if (np.size(t) > 2):
        plt.hlines( 0, t[1], t[-2], zorder=np.size(t)+1 )
    plt.xlabel("Time [ days ]")
    plt.ylabel("Residuals (M-D)")
    return


def plot_hist_prob(a, n, l, color='0.85', alpha=1):
    plt.hist(a, n, normed=True, histtype='stepfilled', color=color,
                                                 alpha=alpha)
    plt.xlabel(l)
    plt.ylabel("Posterior p. d.")
    return


# -------- miscellaneous ------------

def acf(a):
    # return ACF function
    # 95% confidence bar = 1.96/np.sqrt(len(acf))
    aunbiased = a - np.mean(a)
    anorm = np.sum(aunbiased**2)
    result = np.correlate(aunbiased, aunbiased, "full") / anorm
    acf = result[len(result)/2:]
    return acf


def get_file_names(name):
    f1 = 'data/lc_' + name + '_adapt_final.dat'
    f2 = 'data/' + name + '_adapt_post.car1.out'
    f3 = 'data/' + name + '_adapt_yhat_var.car1.out'
    f4 = 'data/' + name + '_adapt_yhat.supou.out'
    return [f1, f2, f3, f4]

