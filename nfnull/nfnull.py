import zuko
import torch
import scipy
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch.distributions as D

from torch import BoolTensor, LongTensor, Size, Tensor
from torch.distributions import Distribution, Transform, constraints, StudentT, Independent
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
from zuko.transforms import MonotonicRQSTransform
from zuko.flows import Flow, MaskedAutoregressiveTransform
from zuko.lazy import UnconditionalDistribution

class ExtendedMonotonicRQSTransform(MonotonicRQSTransform):
    """Extends MonotonicRQSTransform to allow for larger domain without passing argument
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        bound: float = 15.0,
        slope: float = 1e-4,
        **kwargs,
    ):
       super().__init__(
           widths=widths, 
           heights=heights, 
           derivatives=derivatives, 
           bound=bound, 
           slope=slope, 
           **kwargs,
       )
        
class ExtendedNSF(zuko.flows.autoregressive.MAF):
    """Wrapper to use ExtendedMonotonicRQSTransform
    """
    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=ExtendedMonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

class DiagStudentT(Independent):
    """Multivariate Student-T distribution with diagonal covariance matrix"""
    def __init__(self, loc=0., scale=1., ndims=1, nu=5.0):
        self.nu = nu
        super().__init__(StudentT(nu, loc, scale), int(ndims))

    def __repr__(self) -> str:
        return "Diag" + repr(self.base_dist)

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(DiagStudentT, new)
        return super().expand(batch_shape, new)

    def __get_nu__(self):
        return self.nu
        
class TMAF(Flow):
    """Creates a masked autoregressive flow (MAF) with Student-T base distribution.

    References:
        | Tails of Lipschitz Triangular Flows (Jaini et al., 2019)
        | https://arxiv.org/abs/1907.04481

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transformations.
        randperm: Whether features are randomly permuted between transformations or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transformations.
        nu: Degrees of freedom for the Student-T base distribution.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        nu: float = 8.,
        **kwargs,
    ):
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]

        base = UnconditionalDistribution(
            DiagStudentT,
            torch.zeros(features),
            torch.ones(features),
            features,
            torch.tensor(nu),
            buffer=False,
        )

        super().__init__(transforms, base)

class TNSF(TMAF):
    """Neural Spline Flow with Student-T base distribution"""
    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        transforms: int = 3,
        hidden_features: tuple = (64, 64, 64, 64),
        nu: float = 8.,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            transforms=transforms,
            univariate=ExtendedMonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            hidden_features=hidden_features,
            nu=nu,
            **kwargs,
        )

class Rescale():
    """Scales new data according to mean and SD of some data X

    Attributes
    ----------
    mean_x : float
        mean of X
    std_x : float
        standard deviation of X

    Methods
    -------
    forward(x)
        scales data according to properties of X
    reverse(z)
        undoes scaling
    """     
    
    def __init__(self, x):
        self.mean_x = np.mean(x)
        self.std_x = np.std(x) 
        
    def forward(self, x):
        z = (x - self.mean_x) / self.std_x
        return z
    
    def reverse(self, z):
        x = z * self.std_x + self.mean_x
        return x

class NFNull():
    """Methods for using normalizing flows to learn, sample from and compute 
    p-values for distributions.

    Attributes
    ----------
    x : array
        data to be learned
    rescale :
        object of class Rescale, used to center and standardize before training
    mu_x : float
        mean of x
    std_x : float
        standard deviation of x
    min_support : float
        minimum of the domain of x
    max_support : float
        maximum of the domain of x
    features : int
        number of features (dimensionality of x)
    transforms : int
        number of transformations
    hidden_features : tuple
        size of hidden features for one transformation
    bins : int
        number of bins to discretize x into
    passes : int
        number of passes (2 indicates the use of coupling layers)
    min_grid : float
        minimum for grid-based CDF estimation
    max_grid : float
        maximum for grid-based CDF estimation
    grid : array
        grid for grid-based CDF estimation
    grid_points : int
        number of grid points for grid-based CDF estimation
    flow : Zuko flow
        flow to learn x
    pdf : array
        computed probability density function
    cdf : array
        computed cumulative distribution function
    z : array
        x after centering and scaling by standard deviation
    prescaled : bool
        True if the data being modeled has been scaled already
    nu : float
        degrees of freedom for Student-T base distribution

    Methods
    -------
    fit_pdf
        trains the flow, which learns a density function, and estimates a CDF
    get_cdf
        returns the estimated CDF (not advised for tail probabilities)
    log_prob
        returns the log probability under the flow of a given data point
    sample
        sample new data points from trained flow
    p_value
        computes the tail probability for a data point
    """
    
    def __init__(
        self, x, flow='NSF', min_support=float('-inf'), max_support=float('inf'),
        features=1, context=0, transforms=2, hidden_features=(64, 64, 64, 64), bins=16,
        passes=0, min_grid=None, max_grid=None, grid=None, grid_points=100000,
        prescaled=False, nu=8.0
    ):
        self.x = x
        self.rescale = Rescale(x)        
        self.features = features
        self.mu_x = self.x.mean()
        self.std_x = self.x.std()
        self.min_support = np.array(min_support)
        self.max_support = np.array(max_support)
        self.min_grid = np.array(min_grid)
        self.max_grid = np.array(max_grid)
        self.grid = grid        
        self.grid_points = grid_points
        
        # Flow selection
        if flow == 'NSF':
            self.flow = ExtendedNSF(
                features=features, context=context,
                transforms=transforms, hidden_features=hidden_features, 
                bins=bins, passes=passes
            )
        elif flow == 'TNSF':
            self.flow = TNSF(
                features=features, context=context,
                transforms=transforms, hidden_features=hidden_features, 
                bins=bins, nu=nu
            )
        else:
            self.flow = flow
            
        if min_grid is None:
            self.min_grid = np.maximum((x.min() - 5*self.std_x), self.min_support)
            self.max_grid = np.minimum((x.max() + 5*self.std_x), self.max_support)
        if (features == 1) and (self.grid is None):
            one_d_space = np.linspace(self.min_grid, self.max_grid, self.grid_points)            
            self.grid = one_d_space            
        self.pdf = None
        self.cdf = None
        self.z = None
        self.prescaled = prescaled

    def fit_pdf(
        self, batch_size=64, lr=1e-2, n_iter=2000, verbose=False, tol=1e-4, reg_lambda=5e-3,
        tail_lambda=5e-2, t_df=3, weight_decay=0,
    ):
        """Fits Gaussian normalizing flow, which learns a density function. Regularization options
           include l2 penalty on parameters, t-distribution PDF (to encourage longer tails)
           and weight decay

        Parameters
        ----------
        batch_size : int
            batch size for flow training
        lr : float
            learning rate
        n_iter : int
            number of epochs for flow training
        verbose : bool
            whether to print loss per epoch
        tol : float
            tolerance for early stopping
        reg_lambda : float
            magnitude of L1 penalty
        tail_lambda : float
            scaling factor for T prior contribution to loss
        t_df : int
            number of degrees of freedom for T prior
        """
        if not self.prescaled:
            self.z = torch.tensor(self.rescale.forward(self.x))
        flow = self.flow
        trainset = data.TensorDataset(self.z)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
        old_loss = float('inf')
        for epoch in range(n_iter):
            losses = []    
            for hh in trainloader:
                loss = -flow().log_prob(hh[0]).mean() 
                loss += reg_lambda * torch.tensor(
                    [torch.sum(t**2) for t in list(flow.parameters())]
                ).sum()
                loss -= tail_lambda * D.StudentT(df=t_df).log_prob(hh[0]).sum()
                loss.backward()    
                optimizer.step()
                optimizer.zero_grad()    
                losses.append(loss.detach())    
            losses = torch.stack(losses)
            if verbose & (epoch % 100 == 0):
                print(f'({epoch})', losses.mean().item(), '±', losses.std().item())
            if (torch.abs(loss - old_loss)) < tol:
                break
            else:
                old_loss = loss
        self.flow = flow
        if self.prescaled:
            grid_centered = torch.tensor(self.grid)
        else:
            grid_centered = torch.tensor(self.rescale.forward(self.grid))
        if self.features == 1:
            log_pdfs = self.flow().log_prob(grid_centered.reshape(self.grid_points, 1))            
            pdfs = uniform_filter1d(torch.exp(log_pdfs).detach().cpu().numpy(), size=500)
            self.pdf = pdfs
            self.cdf = self.get_cdf(n=self.grid_points)

    def get_cdf(self, n=int(1e5)):
        """Returns CDF

        Parameters
        ----------
        n : int
            number of samples from flow with which to estimate CDF

        Returns
        -------
        xcdf : array
            CDF evaluated along points
        """
        
        x = self.sample(n)
        xcdf = np.zeros(self.grid.shape[0])
        for i in range(self.grid.shape[0]):
            xcdf[i] = (np.sum(x < self.grid[i]) + 1)/(len(x) + 1)

        return xcdf
    
    def log_prob(self, x):
        """Returns log probability

        Parameters
        ----------
        x : float
            data point to score

        Returns
        -------
        flow.log_prob : float
            log of probability density function value for x
        """
        
        return self.flow().log_prob(x)

    def sample(self, n=100000000):
        """Samples points from flow

        Parameters
        ----------
        n : int
            number of samples

        Returns
        -------
        x_hat : array
            n samples from flow
        """
        
        x_hat_centered = self.flow().sample((n,)).detach().cpu().numpy()
        if self.prescaled:
            x_hat = x_hat_centered
        else:
            x_hat = self.rescale.reverse(x_hat_centered)
        x_hat = np.clip(x_hat, self.min_support, self.max_support)
        return x_hat.squeeze()
    
    def p_value(self, x, greater_than=True, n=100000000):
        """Samples points from flow

        Parameters
        ----------
        x : float
            data point for estimating tail probability P(X <= x)
        greater_than : bool
            computes tail probality as greater than (True) or less than (False)
        n : number of samples from flow for computing p-value

        Returns
        -------
        p_value : float
            tail probability P(X <= x)
        """
        
        if self.features > 1:
            raise NotImplementedError(
                'Integration over >1 features not yet supported, please use nfnull.sample() '
                'and define an integration scheme.'
                )
        if greater_than:
            return (np.sum(self.sample(n) > x) + 1)/(n + 1)
        else:
            return (np.sum(self.sample(n) < x) + 1)/(n + 1)
        
