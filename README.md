# LiGrad

Transit Model for Gravity-Darkened Oblate Stars

Developed by Eleftherios Voulimiotis

This package can be used for generating transit light-curves for gravity-darkened, oblate stars using a simple vectorized approach.

It features:
- Gravity-darkening calculation
- Limb-darkening (Claret 4-parameter)
- Vectorized method for planet occultation integration
- Example scripts and a LaTeX manual included

LiGrad makes use of the following packages:

* [Numpy](https://numpy.org), [Oliphant 2006](https://archive.org/details/NumPyBook)
* [SciPy](https://www.scipy.org), [Virtanen et al. 2020](https://www.nature.com/articles/s41592-019-0686-2)
* [Pylightcurve](https://github.com/ucl-exoplanets/pylightcurve), [Tsiaras et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...832..202T/abstract)
* [Astropy](https://www.astropy.org), [Astropy Collaboration 2013](https://www.aanda.org/articles/aa/abs/2013/10/aa22068-13/aa22068-13.html)
* [Emcee](https://github.com/dfm/emcee), [Foreman-Mackey et al. 2013](http://iopscience.iop.org/article/10.1086/670067)
* [Matplotlib](https://matplotlib.org), [Hunter 2007](https://ieeexplore.ieee.org/document/4160265)
* [Exoclock](https://www.exoclock.space/), [Kokori et al. 2020](https://ui.adsabs.harvard.edu/abs/2020arXiv201207478K/abstract)
* [Astroquery](https://github.com/astropy/astroquery), [Ginsburg et al. 2019](https://iopscience.iop.org/article/10.3847/1538-3881/aafc33)

as well as the following catalogues:

* Exoplanet Characterisation Catalogue, developed as part of the [ExoClock Project](www.exoclock.space), [Kokori et al. 2020](https://ui.adsabs.harvard.edu/abs/2020arXiv201207478K/abstract)
* [MAST Catalogs](https://mast.stsci.edu/), (MAST 2021, MAST Portal Guide, eds. R.A. Shaw, B. Cherinka, P. Forshay)
* [Exo.MAST](https://exo.mast.stsci.edu/), [DR Rodriguez et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ASPC..527..351R/abstract)
* [ESA Gaia Archive](https://gea.esac.esa.int/archive/), [Gaia Collaboration et al. 2023](https://www.aanda.org/articles/aa/full_html/2023/06/aa43940-22/aa43940-22.html)

# Installation
You can install LiGrad through pip:

```bash
pip install ligrad
```

```bash
git clone https://github.com/evoulimiotis/ligrad.git
cd ligrad
python setup.py install
```

### See MANUAL.pdf in [GitHub](https://github.com/evoulimiotis/LiGrad.git) for detailed information.