# AGN Dust Reverberation Mapping for the Hubble Constant
## Methodological Improvements, Data Expansion, and New Cosmology Constraints
## Mo Spiegel: moshespieg@gmail.com

### Project Background:
This research is ongoing, and began in June 2026. This research focuses on the use of dust reverberation mapping (DRM) of the obscuring tori in active galactic nuclei (AGN), to infer AGN central engine luminosities and therefore extract cosmological distances that can be used to measure the Hubble Constant.

#### Existing Framework:
[Yoshii et al. 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...784L..11Y/abstract) provides a theoretical framework for obtaining distances given time lags from DRM. They utilize an equation of radiative equilibrium to analytically derive the relationship between time lags and luminosity distance given dust sublimation temperature, dust grain size distribution, and a representative power law index for the AGN spectrum population. The above parameters are used to calibrate **g = 10.60**, which is the central calibration constant that is used to obtain distances given time lags.

\$g = 2.5\log_{10}\left(
\frac{
16\pi c^2 \int_{a_{\min}}^{a_{\max}} a^{2-p}
\int_{\mathrm{NIR}} Q_{\nu}(a)B_{\nu}(T_{\mathrm{sub}})\,d\nu\,da
}{
L_0 \int_{a_{\min}}^{a_{\max}} a^{2-p}
\int_{\mathrm{UV}} Q_{\nu}(a)
\left(\frac{\nu}{\nu_V}\right)^{\alpha_{\mathrm{UV}}}\,d\nu\,da
}
\right)\$
