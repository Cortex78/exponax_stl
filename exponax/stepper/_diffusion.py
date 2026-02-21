from typing import TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._base_stepper import BaseStepper
from ..nonlin_fun import ZeroNonlinearFun

D = TypeVar("D")


class Diffusion(BaseStepper):
    diffusivity: Float[Array, "D D"]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: Float[Array, "D D"] | Float[Array, "D"] | float = 0.01,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) diffusion equation
        on periodic boundary conditions.

        In 1d, the diffusion equation is given by

        ```
            uₜ = ν uₓₓ
        ```

        with `ν ∈ ℝ` being the diffusivity.

        In higher dimensions, the diffusion equation can written using the
        Laplacian operator.

        ```
            uₜ = ν Δu
        ```

        More generally speaking, there can be anisotropic diffusivity given by a
        `A ∈ ℝᵈ ˣ ᵈ` sandwiched between the gradient and divergence operators.

        ```
            uₜ = ∇ ⋅ (A ∇u)
        ```

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `diffusivity` (keyword-only): The diffusivity `ν`. In higher
            dimensions, this can be a scalar (=float), a vector of length `d`,
            or a matrix of shape `d ˣ d`. If a scalar is given, the diffusivity
            is assumed to be the same in all spatial dimensions. If a vector (of
            length `d`) is given, the diffusivity varies across dimensions (=>
            diagonal diffusion). For a matrix, there is fully anisotropic
            diffusion. In this case, `A` must be symmetric positive definite
            (SPD). Default: `0.01`.

        **Notes:**

        - The stepper is unconditionally stable, no matter the choice of
            any argument because the equation is solved analytically in Fourier
            space.
        - A `ν > 0` leads to stable and decaying solutions (i.e., energy is
            removed from the system). A `ν < 0` leads to unstable and growing
            solutions (i.e., energy is added to the system).
        - Ultimately, only the factor `ν Δt / L²` affects the characteristic
            of the dynamics. See also
            [`exponax.stepper.generic.NormalizedLinearStepper`][] with
            `normalized_coefficients = [0, 0, alpha_2]` with `alpha_2 =
            diffusivity * dt / domain_extent**2`.
        """
        if num_spatial_dims not in {1, 2, 3}:
            raise ValueError(
                f"num_spatial_dims must be 1, 2, or 3, got {num_spatial_dims}"
            )
        if domain_extent <= 0:
            raise ValueError(f"domain_extent must be positive, got {domain_extent}")
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        diffusivity = jnp.asarray(diffusivity)
        if diffusivity.ndim == 0:
            diffusivity = jnp.diag(jnp.ones(num_spatial_dims) * diffusivity)
        elif diffusivity.ndim == 1:
            if diffusivity.shape != (num_spatial_dims,):
                raise ValueError(
                    f"diffusivity as a vector must have shape ({num_spatial_dims},), "
                    f"got {diffusivity.shape}"
                )
            diffusivity = jnp.diag(diffusivity)
        elif diffusivity.ndim == 2:
            if diffusivity.shape != (num_spatial_dims, num_spatial_dims):
                raise ValueError(
                    f"diffusivity as a matrix must have shape ({num_spatial_dims}, "
                    f"{num_spatial_dims}), got {diffusivity.shape}"
                )
            if not jnp.allclose(diffusivity, diffusivity.T):
                raise ValueError("diffusivity matrix must be symmetric")
            eigenvalues = jnp.linalg.eigvalsh(diffusivity)
            if jnp.any(eigenvalues < 0):
                raise ValueError(
                    "diffusivity matrix must be positive semi-definite (all eigenvalues >= 0)"
                )
        else:
            raise ValueError(
                f"diffusivity must be a scalar, 1D array, or 2D array, "
                f"got {diffusivity.ndim}D array"
            )
        self.diffusivity = diffusivity
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        laplace_outer_producct = (
            derivative_operator[:, None] * derivative_operator[None, :]
        )
        linear_operator = jnp.einsum(
            "ij,ij...->...",
            self.diffusivity,
            laplace_outer_producct,
        )
        # Add the necessary singleton channel axis
        linear_operator = linear_operator[None, ...]
        return linear_operator

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
        )
