# p-values for multimodal null distributions

This repository is for exploring approximations to difficult-to-sample or -to-model null distributions of the sort encountered during Monte Carlo (MC) sampling. The focus is deriving an approximation to the CDF such that more precise p-values can be obtained than through direct MC sampling. Although built on PyTorch, we do not require GPU access.

As we are not yet on PyPi, to install please follow this recipe:

```
git clone git@github.com:Genentech/nfnull.git
conda env -n 'nfnull' python=3.11.1
conda activate nfnull
pip install zuko torch torchaudio torchvision scipy numpy pandas
pip install nfnull/NFNull
```

You can then run in Python as follows:

```
import torch
import scipy
import numpy as np

from nfnull import NFNull

def gmm_xdf(x, locs, scales, ws, sf=False):
    """ 
    Gaussian mixture model
    """
    pdfs = np.zeros(len(x))
    cdfs = np.zeros(len(x))
    for i in range(len(locs)):
        pdfs += ws[i] * scipy.stats.norm.pdf(x, loc=locs[i], scale=scales[i])
        if sf:
            cdfs += ws[i] * scipy.stats.norm.sf(x, loc=locs[i], scale=scales[i])
        else:
            cdfs += ws[i] * scipy.stats.norm.cdf(x, loc=locs[i], scale=scales[i])        
    return pdfs, cdfs

## generate trimodal distribution
x = np.concatenate((
    scipy.stats.norm.rvs(loc=-1, scale=0.25, size=150),
    scipy.stats.norm.rvs(loc=0, scale=0.25, size=150),
    scipy.stats.norm.rvs(loc=1.5, scale=0.25, size=200)
))
nfn = NFNull(x)
nfn.fit_pdf(verbose=True, tol=1e-4)

print(f"Analytic p-value: {gmm_xdf([4.9], [-1, 0, 1.5], [0.25]*3, [0.3, 0.3, 0.4], sf=True)[1][0]}")
print(f"Empirical mean from samples {np.mean(x > 4.9)}")
print(f"Neural approx: {nfn.p_value(4.9)}")
```

This should give something close to:

```
Analytic p-value: 8.008668894725593e-43
Empirical mean from samples 0.0
Neural approx: 9.9999999e-09
```

Calling from R, assuming you have named your Python environment 'nfnull':

```
# Load the reticulate library and set up the Python environment
library(reticulate)

# Set the Python environment to 'nfnull'
use_py <- subset(reticulate::conda_list(), name == 'nfnull')$python
Sys.setenv(RETICULATE_PYTHON = use_py)
use_condaenv('nfnull')

# Import necessary Python libraries
np <- import("numpy")
scipy <- import("scipy.stats")
torch <- import("torch")
nfnull <- import("nfnull")

# Define the Gaussian Mixture Model function in R using reticulate
gmm_xdf <- function(x, locs, scales, ws, sf = FALSE) {
  pdfs <- np$zeros(length(x))
  cdfs <- np$zeros(length(x))
  for (i in seq_along(locs)) {
    pdfs <- pdfs + ws[i] * scipy$norm$pdf(x, loc = locs[i], scale = scales[i])
    if (sf) {
      cdfs <- cdfs + ws[i] * scipy$norm$sf(x, loc = locs[i], scale = scales[i])
    } else {
      cdfs <- cdfs + ws[i] * scipy$norm$cdf(x, loc = locs[i], scale = scales[i])
    }
  }
  return(list(pdfs = pdfs, cdfs = cdfs))
}

# Generate the trimodal distribution
x <- np$concatenate(list(
  scipy$norm$rvs(loc = -1, scale = 0.25, size = 150L),
  scipy$norm$rvs(loc = 0, scale = 0.25, size = 150L),
  scipy$norm$rvs(loc = 1.5, scale = 0.25, size = 200L)
))

# Initialize NFNull and fit the model
nfn <- nfnull$NFNull(x)
nfn$fit_pdf(verbose = TRUE, tol = 1e-4)

# Compute and print the p-values
analytic_p_value <- gmm_xdf(c(4.9), c(-1, 0, 1.5), rep(0.25, 3), c(0.3, 0.3, 0.4), sf = TRUE)$cdfs[1]
empirical_mean <- np$mean(x > 4.9)
neural_approx <- nfn$p_value(4.9)

cat("Analytic p-value:", analytic_p_value, "\n")
cat("Empirical mean from samples:", empirical_mean, "\n")
cat("Neural approx:", neural_approx, "\n")
```
