# AGN Dust Reverberation Mapping for the Hubble Constant
## Methodological Improvements, Data Expansion, and New Cosmology Constraints
## Mo Spiegel: moshespieg@gmail.com

### Project Status:
This research is ongoing, and began in June 2026. This research focuses on the use of dust reverberation mapping (DRM) of the obscuring tori in active galactic nuclei (AGN), to infer AGN central engine luminosities and therefore extract cosmological distances that can be used to measure the Hubble Constant.

#### Brief Background:
The dust torus in an AGN, which has also been referred to as the obscuring torus, is a key part of the unified model of different AGN subtypes and provides an explanation for the obscuring of the broad line region (BLR) in Seyfert 2 galaxies. The torus is aligned, for the most part, on the plane of the accretion disk in the central engine (with some theories suggesting that the dusty torus even supplies accreting matter), and is located further from the central engine than the ionized hydrogen clouds of the BLR. The dusty torus is optically thick; as such, it is capable of obscuring emission from the central engine and from the BLR. 

The dust torus exists around the central engine up until the radius where the incident flux received from the central engine causes the dust to reach the temperature at which it sublimates, i.e. the sublimation temperature. In other words, the luminosity of the central engine directly influences the characteristic radius of the inner edge of the dust torus. It therefore becomes possible, in principle, to infer the luminosity given the radius of the dust torus. Indeed, there has been shown to be a tight correlation between AGN luminosity and torus size.

To determine torus size, the principle technique is **reverberation mapping** (although recently interferometric techniques have been used to directly obtain observational sizes). Reverberation mapping is made possible by the radiative response of the dust torus to an increase in incident flux: The inner edge of dust tori reemit incident radiation (which is received predominantly in the UV, X-ray, and visible wavelengths) mostly the near-infrared (calculated by the temperature of the dust at the sublimation radius, which is effectively the sublimation temperature). Therefore, a variation in central engine flux will lead to a variation in the dust thermal emission.

[Yoshii et al. 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...784L..11Y/abstract) (henceforth Y14) provides a theoretical framework for obtaining distances given time lags from DRM. They utilize an equation of radiative equilibrium to analytically derive the relationship between time lags and luminosity distance given dust sublimation temperature, dust grain size distribution, and a representative power law index for the AGN spectrum population. The above parameters are used to calibrate **g = 10.60**, which is the central calibration constant that is used to obtain distances given time lags.

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

*This equation is not given explicitly by Y14, but is rederived by the researcher from Yoshii's descriptions*

Y14 and [Minezaki et al. 2019](https://arxiv.org/html/1910.08722v2) (henceforth M19) give the following to obtain luminosity distances given object apparent apparent magnitude, DRM time lag, and g:
