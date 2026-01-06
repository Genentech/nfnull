import zuko
import torch
import scipy
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch.distributions as D
from functools import partial

from torch import BoolTensor, LongTensor, Size, Tensor
from torch.distributions import (
    Distribution,
    Transform,
    constraints,
    StudentT,
    Independent,
)
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
from zuko.transforms import MonotonicRQSTransform
from zuko.flows import Flow, MaskedAutoregressiveTransform
from zuko.lazy import UnconditionalDistribution


class DiagStudentT(Independent):
    """Multivariate Student-T distribution with diagonal covariance matrix"""

    def __init__(self, loc=0.0, scale=1.0, ndims=1, nu=5.0):
        self.nu = nu
        # Expand base distribution to match desired dimensions
        base_dist = StudentT(nu, loc, scale).expand((ndims,))
        super().__init__(base_dist, 1)

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
        nu: float = 8.0,
        **kwargs,
    ):
        # Extract spline-specific parameters from kwargs
        univariate = kwargs.pop("univariate", None)
        shapes = kwargs.pop("shapes", None)

        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        # Create transforms with or without spline functionality
        if univariate is not None and shapes is not None:
            # TNSF case: Create spline transforms
            transforms = [
                MaskedAutoregressiveTransform(
                    features=features,
                    context=context,
                    univariate=univariate,
                    shapes=shapes,
                    order=torch.randperm(features) if randperm else orders[i % 2],
                    **kwargs,
                )
                for i in range(transforms)
            ]
        else:
            # Standard TMAF case: Create basic autoregressive transforms
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
        nu: float = 8.0,
        bound: float = 40.0,
        slope: float = 1e-3,
        **kwargs,
    ):
        # Create a partial function for MonotonicRQSTransform with bound and slope
        RQSTransform = partial(MonotonicRQSTransform, bound=bound, slope=slope)

        super().__init__(
            features=features,
            context=context,
            transforms=transforms,
            univariate=RQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            hidden_features=hidden_features,
            nu=nu,
            **kwargs,
        )


class Rescale:
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
        self.mean_x = np.mean(x, axis=0)
        self.std_x = np.std(x, axis=0)

    def forward(self, x):
        z = (x - self.mean_x) / self.std_x
        return z

    def reverse(self, z):
        x = z * self.std_x + self.mean_x
        return x


class NFNull:
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
    to(device)
        move model to specified device
    """

    def __init__(
        self,
        x,
        flow="NSF",
        min_support=float("-inf"),
        max_support=float("inf"),
        features=1,
        context=0,
        transforms=2,
        hidden_features=(64, 64, 64, 64),
        bins=16,
        passes=0,
        min_grid=None,
        max_grid=None,
        grid=None,
        grid_points=100000,
        prescaled=True,
        nu=0.0,
        bound=8.0,
        slope=1e-4,  # defaults from MonotonicRQSTransform
    ):
        self.x = x
        if not prescaled:
            self.rescale = Rescale(x)
        self.features = features

        # Handle device-specific operations
        if isinstance(x, torch.Tensor):
            self.device = x.device
            self.mu_x = torch.mean(x, axis=0)
            self.std_x = torch.std(x, axis=0)
            self.min_support = torch.tensor(min_support, device=self.device)
            self.max_support = torch.tensor(max_support, device=self.device)
        else:
            self.device = torch.device("cpu")
            self.mu_x = np.mean(x, axis=0)
            self.std_x = np.std(x, axis=0)
            self.min_support = np.array(min_support)
            self.max_support = np.array(max_support)

        # Flow selection - move to correct device
        if flow == "NSF":
            # Create a partial function for MonotonicRQSTransform with bound and slope
            RQSTransform = partial(MonotonicRQSTransform, bound=bound, slope=slope)

            self.flow = zuko.flows.autoregressive.MAF(
                features=features,
                context=context,
                transforms=transforms,
                hidden_features=hidden_features,
                passes=passes,
                univariate=RQSTransform,
                shapes=[(bins,), (bins,), (bins - 1,)],
            ).to(self.device)
        elif flow == "TNSF":
            self.flow = TNSF(
                features=features,
                context=context,
                transforms=transforms,
                hidden_features=hidden_features,
                bins=bins,
                nu=nu,
                bound=bound,
                slope=slope,
            ).to(self.device)
        else:
            self.flow = flow.to(self.device)

        # Grid setup
        if min_grid is None:
            if isinstance(x, torch.Tensor):
                self.min_grid = torch.maximum(
                    x.min() - 5 * self.std_x, self.min_support
                )
                self.max_grid = torch.minimum(
                    x.max() + 5 * self.std_x, self.max_support
                )
            else:
                self.min_grid = np.maximum(np.min(x) - 5 * self.std_x, self.min_support)
                self.max_grid = np.minimum(np.max(x) + 5 * self.std_x, self.max_support)
        self.grid = grid
        if (features == 1) and (self.grid is None):
            if isinstance(x, torch.Tensor):
                self.grid = torch.linspace(
                    self.min_grid, self.max_grid, grid_points, device=self.device
                )
            else:
                self.grid = np.linspace(self.min_grid, self.max_grid, grid_points)

        self.grid_points = grid_points
        self.pdf = None
        self.cdf = None
        self.z = None
        self.prescaled = prescaled

    def fit_pdf(
        self,
        batch_size=64,
        lr=1e-2,
        n_iter=1000,
        verbose=False,
        tol=1e-4,
        reg_lambda=0,
        tail_lambda=0,
        t_df=0,
        weight_decay=1e-3,
        context=None,
        make_grid_estimator=False,
        patience=10,
    ):
        """Fits Gaussian normalizing flow, which learns a density function.

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
            magnitude of L2 penalty
        tail_lambda : float
            scaling factor for T prior contribution to loss
        t_df : int
            number of degrees of freedom for T prior
        context : array-like, optional
            Context variables for conditional density estimation
        make_grid_estimator : bool, optional
            whether to create a grid-based density estimator after training (can be memory intensive)
        patience : int
            number of epochs without improvement before stopping
        """
        # Initialize self.z regardless of prescaling
        # Move to self.device for GPU/MPS support (backwards-compatible: defaults to CPU)
        if self.prescaled:
            self.z = torch.tensor(self.x, dtype=torch.float32, device=self.device).reshape(
                -1, self.features
            )
        else:
            self.z = torch.tensor(
                self.rescale.forward(self.x), dtype=torch.float32, device=self.device
            ).reshape(-1, self.features)

        flow = self.flow

        if context is not None:
            context = torch.tensor(context, dtype=torch.float32, device=self.device)
            self.training_context = context  # Store training context
            trainset = data.TensorDataset(self.z, context)
        else:
            trainset = data.TensorDataset(self.z)

        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            flow.parameters(), lr=lr, weight_decay=weight_decay
        )
        old_loss = float("inf")
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_iter):
            losses = []
            for batch in trainloader:
                if context is not None:
                    x_batch, ctx_batch = batch
                    loss = -flow(ctx_batch).log_prob(x_batch).mean()
                else:
                    x_batch = batch[0]
                    loss = -flow().log_prob(x_batch).mean()

                if reg_lambda > 0:
                    loss += (
                        reg_lambda
                        * torch.tensor(
                            [torch.sum(t**2) for t in list(flow.parameters())],
                            device=self.device
                        ).sum()
                    )

                if tail_lambda > 0 and t_df > 0:
                    loss -= tail_lambda * D.StudentT(df=t_df).log_prob(x_batch).sum()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.detach())

            # Calculate mean loss for this epoch
            mean_epoch_loss = torch.stack(losses).mean()

            if verbose:
                print(
                    f"Epoch {epoch}: mean loss = {mean_epoch_loss.item()}, best = {best_loss}"
                )
                print(f"Patience counter: {patience_counter}/{patience}")

            # Check if loss improved
            if mean_epoch_loss < best_loss - tol:
                best_loss = mean_epoch_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch} due to no improvement for {patience} epochs"
                        )
                    break

            old_loss = mean_epoch_loss.item()

        self.flow = flow

        # Only process grid if it exists (1D case only)
        if self.grid is not None:
            if self.prescaled:
                grid_centered = torch.tensor(self.grid)
            else:
                grid_centered = torch.tensor(self.rescale.forward(self.grid))
        else:
            grid_centered = None

        # Only create grid estimator if requested
        if make_grid_estimator and self.features == 1:
            if self.prescaled:
                grid_centered = torch.tensor(self.grid, device="cpu")
            else:
                grid_centered = torch.tensor(
                    self.rescale.forward(self.grid), device="cpu"
                )

            if context is not None:
                # For conditional models, compute grid estimates for each context point
                log_pdfs = []
                for ctx in context:
                    # Move context to device, compute density, then move results back to CPU
                    ctx_reshaped = ctx.reshape(1, -1).to(self.device)
                    with torch.no_grad():  # No need to track gradients for inference
                        log_pdf = (
                            self.flow(ctx_reshaped)
                            .log_prob(
                                grid_centered.reshape(self.grid_points, 1).to(
                                    self.device
                                )
                            )
                            .cpu()
                        )
                    log_pdfs.append(log_pdf)
                    # Clear GPU cache after each iteration
                    torch.cuda.empty_cache()

                # Store all conditional grid estimates
                log_pdfs = torch.stack(log_pdfs)
                pdfs = torch.exp(log_pdfs).numpy()
                # Apply smoothing to each estimate
                self.pdf = np.array([uniform_filter1d(pdf, size=500) for pdf in pdfs])
            else:
                # For unconditional model
                with torch.no_grad():
                    log_pdfs = (
                        self.flow()
                        .log_prob(
                            grid_centered.reshape(self.grid_points, 1).to(self.device)
                        )
                        .cpu()
                    )
                pdfs = uniform_filter1d(torch.exp(log_pdfs).numpy(), size=500)
                self.pdf = pdfs
                torch.cuda.empty_cache()
        else:
            self.pdf = None

    def get_cdf(self, n=10000, context=None):
        """Returns CDF

        Parameters
        ----------
        n : int
            number of samples from flow with which to estimate CDF
        context : tensor, optional
            Context variables for conditional CDF estimation. Must be provided if model was trained
            with context.

        Returns
        -------
        xcdf : array
            CDF evaluated along points

        Raises
        ------
        ValueError
            If the model was trained with context but no context is provided
        NotImplementedError
            If called on multivariate data (features > 1)
        """
        if self.features > 1:
            raise NotImplementedError(
                f"CDF estimation not supported for multivariate data (features={self.features}). "
                "Use p_value() for multivariate tail probabilities, or sample() for empirical distributions."
            )

        if hasattr(self, "training_context") and context is None:
            raise ValueError(
                "This model was trained with context, but no context was provided for CDF estimation. "
                "Please provide a specific context vector."
            )

        # Handle device for context
        if context is not None:
            if isinstance(context, torch.Tensor):
                context = context.to(self.device)
            else:
                # Convert numpy arrays or other array-like inputs to tensors
                context = torch.tensor(context, dtype=torch.float32, device=self.device)

        x = self.sample(n, context=context)

        # Convert grid to numpy if it's a tensor for comparison
        grid_np = (
            self.grid.cpu().numpy()
            if isinstance(self.grid, torch.Tensor)
            else self.grid
        )

        xcdf = np.zeros(len(grid_np))
        for i in range(len(grid_np)):
            xcdf[i] = (np.sum(x < grid_np[i]) + 1) / (len(x) + 1)

        return xcdf

    def log_prob(self, x, context=None):
        """Returns log probability

        Parameters
        ----------
        x : float
            data point to score
        context : tensor, optional
            Context variables for conditional probability

        Returns
        -------
        flow.log_prob : float
            log of probability density function value for x
        """
        if context is not None:
            return self.flow(context).log_prob(x)
        return self.flow().log_prob(x)

    def sample(self, n=10000, context=None, batch_size=1000000):
        """Samples points from flow with true batched context support

        Parameters
        ----------
        n : int
            Number of samples per group (if context is batched) or total samples
        context : tensor, optional
            Context variables. If 2D with shape [num_groups, context_features],
            generates n samples for each group using vectorized operations
        batch_size : int
            Maximum number of samples to generate at once per group

        Returns
        -------
        x_hat : array
            Samples from flow. Shape depends on context:
            - No context: [n, features] (or [n] if features=1)
            - Single context: [n, features] (or [n] if features=1)
            - Batched context: [n, num_groups, features] (or [n, num_groups] if features=1)
        """

        # Handle device for context
        if context is not None:
            if isinstance(context, torch.Tensor):
                context = context.to(self.device)
            else:
                # Convert numpy arrays or other array-like inputs to tensors
                context = torch.tensor(context, dtype=torch.float32, device=self.device)

        # Determine if we have batched contexts
        is_batched = context is not None and context.ndim >= 2 and context.shape[0] > 1

        # Generate samples in batches to avoid memory issues
        samples = []
        remaining = n

        while remaining > 0:
            current_batch_size = min(batch_size, remaining)

            if context is not None:
                # This handles both single and batched contexts via torch's broadcasting
                batch_samples = (
                    self.flow(context)
                    .sample((current_batch_size,))
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                batch_samples = (
                    self.flow().sample((current_batch_size,)).detach().cpu().numpy()
                )

            samples.append(batch_samples)
            remaining -= current_batch_size

        # Concatenate all batches along sample dimension
        if not samples:
            if is_batched:
                empty_shape = (
                    (0, context.shape[0], self.features)
                    if self.features > 1
                    else (0, context.shape[0])
                )
            else:
                empty_shape = (0, self.features) if self.features > 1 else (0,)
            return np.empty(empty_shape)

        x_hat_centered = np.concatenate(samples, axis=0)

        # Apply reverse scaling if needed
        if not self.prescaled:
            # Handle batched case - need to be careful about broadcasting
            if is_batched and self.features > 1:
                # Reshape for broadcasting: x_hat_centered is [n, num_groups, features]
                original_shape = x_hat_centered.shape
                x_hat_flat = x_hat_centered.reshape(-1, self.features)
                x_hat_scaled = self.rescale.reverse(x_hat_flat)
                x_hat = x_hat_scaled.reshape(original_shape)
            else:
                x_hat = self.rescale.reverse(x_hat_centered)
        else:
            x_hat = x_hat_centered

        # Apply clipping
        min_support = (
            self.min_support.cpu().numpy()
            if isinstance(self.min_support, torch.Tensor)
            else self.min_support
        )
        max_support = (
            self.max_support.cpu().numpy()
            if isinstance(self.max_support, torch.Tensor)
            else self.max_support
        )

        x_hat = np.clip(x_hat, min_support, max_support)

        # Handle feature=1 case by squeezing last dimension only
        if self.features == 1:
            x_hat = x_hat.squeeze(-1)

        return x_hat

    def p_value(
        self, x, greater_than=True, n=1000000, context=None, batch_size=1000000
    ):
        """Samples points from flow to compute tail probability with true batched analysis

        Parameters
        ----------
        x : float, array-like, or tensor
            Data point(s) for estimating tail probability
        greater_than : bool
            Computes tail probability as greater than (True) or less than (False)
        n : int
            Number of samples from flow for computing p-value
        context : tensor, optional
            Context variables. If batched, computes p-value for each group simultaneously
        batch_size : int
            Maximum number of samples to generate at once to avoid memory issues

        Returns
        -------
        p_value : float or array
            Tail probability. If context is batched, returns array of p-values (one per group)
        """

        # Handle device for context
        if context is not None:
            if isinstance(context, torch.Tensor):
                context = context.to(self.device)
            else:
                # Convert numpy arrays or other array-like inputs to tensors
                context = torch.tensor(context, dtype=torch.float32, device=self.device)

        # Determine if we have batched contexts
        is_batched = context is not None and context.ndim >= 2 and context.shape[0] > 1

        if is_batched:
            num_groups = context.shape[0]

            # Handle x input for batched case
            x_array = np.asarray(x)
            if x_array.ndim == 0:  # Scalar x
                if self.features == 1:
                    x_broadcast = np.full(num_groups, x_array.item())
                else:
                    x_broadcast = np.tile(x_array, (num_groups, 1))
            elif self.features == 1 and x_array.ndim == 1:
                if len(x_array) != num_groups:
                    raise ValueError(
                        f"For batched context, x length ({len(x_array)}) "
                        f"must match number of groups ({num_groups}) or be scalar"
                    )
                x_broadcast = x_array
            elif self.features > 1:
                if x_array.ndim == 1 and len(x_array) == self.features:
                    x_broadcast = np.tile(x_array[None, :], (num_groups, 1))
                elif x_array.ndim == 2 and x_array.shape == (num_groups, self.features):
                    x_broadcast = x_array
                else:
                    raise ValueError(f"Invalid x shape for batched context")

            # Generate samples in batches for memory management, accumulate counts
            counts = np.zeros(num_groups)
            total = 0
            remaining = n

            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                batch_samples = self.sample(current_batch_size, context=context)

                # batch_samples shape: [current_batch_size, num_groups] or [current_batch_size, num_groups, features]

                if self.features == 1:
                    # batch_samples: [current_batch_size, num_groups], x_broadcast: [num_groups]
                    if greater_than:
                        comparison = batch_samples >= x_broadcast[None, :]
                    else:
                        comparison = batch_samples <= x_broadcast[None, :]
                    # Sum over sample dimension, accumulate per group
                    counts += np.sum(comparison, axis=0)
                else:
                    # batch_samples: [current_batch_size, num_groups, features]
                    # x_broadcast: [num_groups, features]
                    if greater_than:
                        comparison = batch_samples >= x_broadcast[None, :, :]
                    else:
                        comparison = batch_samples <= x_broadcast[None, :, :]
                    # For multivariate: all features must satisfy condition
                    joint_comparison = np.all(
                        comparison, axis=2
                    )  # Shape: [current_batch_size, num_groups]
                    counts += np.sum(joint_comparison, axis=0)

                total += current_batch_size
                remaining -= current_batch_size

            # Bayesian smoothing per group
            p_values = (counts + 1) / (total + 1)
            return p_values

        else:
            # Single context case - generate samples in batches for memory management
            count = 0
            total = 0
            remaining = n

            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                batch_samples = self.sample(current_batch_size, context=context)

                # batch_samples will be [current_batch_size] or [current_batch_size, features] for single context

                if greater_than:
                    if self.features == 1:
                        # Convert x to numpy array to handle tensor inputs
                        x_array = np.asarray(x)
                        count += np.sum(batch_samples >= x_array)
                    else:
                        x_array = np.asarray(x)
                        comparison = batch_samples >= x_array
                        count += np.sum(np.all(comparison, axis=1))
                else:
                    if self.features == 1:
                        # Convert x to numpy array to handle tensor inputs
                        x_array = np.asarray(x)
                        count += np.sum(batch_samples <= x_array)
                    else:
                        x_array = np.asarray(x)
                        comparison = batch_samples <= x_array
                        count += np.sum(np.all(comparison, axis=1))

                total += len(batch_samples)
                remaining -= current_batch_size

            return (count + 1) / (total + 1)

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        self.flow = self.flow.to(device)
        if isinstance(self.x, torch.Tensor):
            self.x = self.x.to(device)
        if isinstance(self.grid, torch.Tensor):
            self.grid = self.grid.to(device)
        return self
