import jax.numpy as jnp
import pytest
from exponax.stepper import AdvectionDiffusion, Advection, Diffusion

def test_valid_init():
    # Should not raise any error
    AdvectionDiffusion(
        num_spatial_dims=2,
        domain_extent=1.0,
        num_points=16,
        dt=0.1,
        velocity=jnp.array([1.0, 2.0]),
        diffusivity=jnp.array([[0.1, 0.0], [0.0, 0.1]])
    )

def test_invalid_spatial_dims():
    with pytest.raises(ValueError, match="num_spatial_dims must be 1, 2, or 3"):
        AdvectionDiffusion(num_spatial_dims=4, domain_extent=1.0, num_points=16, dt=0.1)

def test_invalid_domain_extent():
    with pytest.raises(ValueError, match="domain_extent must be positive"):
        AdvectionDiffusion(num_spatial_dims=2, domain_extent=-1.0, num_points=16, dt=0.1)

def test_invalid_num_points():
    with pytest.raises(ValueError, match="num_points must be positive"):
        AdvectionDiffusion(num_spatial_dims=2, domain_extent=1.0, num_points=0, dt=0.1)

def test_invalid_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        AdvectionDiffusion(num_spatial_dims=2, domain_extent=1.0, num_points=16, dt=0.0)

def test_invalid_velocity_shape():
    with pytest.raises(ValueError, match="velocity must have shape"):
        AdvectionDiffusion(
            num_spatial_dims=2,
            domain_extent=1.0,
            num_points=16,
            dt=0.1,
            velocity=jnp.array([1.0, 2.0, 3.0])
        )

def test_invalid_diffusivity_shape():
    with pytest.raises(ValueError, match="diffusivity as a matrix must have shape"):
        AdvectionDiffusion(
            num_spatial_dims=2,
            domain_extent=1.0,
            num_points=16,
            dt=0.1,
            diffusivity=jnp.eye(3)
        )

def test_invalid_diffusivity_non_symmetric():
    with pytest.raises(ValueError, match="diffusivity matrix must be symmetric"):
        AdvectionDiffusion(
            num_spatial_dims=2,
            domain_extent=1.0,
            num_points=16,
            dt=0.1,
            diffusivity=jnp.array([[0.1, 0.2], [0.1, 0.1]])
        )

def test_invalid_diffusivity_negative_eigenvalues():
    with pytest.raises(ValueError, match="diffusivity matrix must be positive semi-definite"):
        AdvectionDiffusion(
            num_spatial_dims=2,
            domain_extent=1.0,
            num_points=16,
            dt=0.1,
            diffusivity=jnp.array([[0.1, 0.2], [0.2, 0.1]])
            # Eigenvalues are 0.1 + 0.2 = 0.3 and 0.1 - 0.2 = -0.1
        )

def test_scalar_velocity_conversion():
    stepper = AdvectionDiffusion(
        num_spatial_dims=2,
        domain_extent=1.0,
        num_points=16,
        dt=0.1,
        velocity=2.5
    )
    assert stepper.velocity.shape == (2,)
    assert jnp.all(stepper.velocity == 2.5)

def test_scalar_diffusivity_conversion():
    stepper = AdvectionDiffusion(
        num_spatial_dims=2,
        domain_extent=1.0,
        num_points=16,
        dt=0.1,
        diffusivity=0.05
    )
    assert stepper.diffusivity.shape == (2, 2)
    assert jnp.allclose(stepper.diffusivity, jnp.diag(jnp.array([0.05, 0.05])))

def test_vector_diffusivity_conversion():
    stepper = AdvectionDiffusion(
        num_spatial_dims=2,
        domain_extent=1.0,
        num_points=16,
        dt=0.1,
        diffusivity=jnp.array([0.1, 0.2])
    )
    assert stepper.diffusivity.shape == (2, 2)
    assert jnp.allclose(stepper.diffusivity, jnp.diag(jnp.array([0.1, 0.2])))

def test_advection_valid():
    Advection(num_spatial_dims=2, domain_extent=1.0, num_points=16, dt=0.1, velocity=1.0)

def test_advection_invalid_velocity():
    with pytest.raises(ValueError, match="velocity must have shape"):
        Advection(num_spatial_dims=2, domain_extent=1.0, num_points=16, dt=0.1, velocity=jnp.array([1.0, 2.0, 3.0]))

def test_diffusion_valid():
    Diffusion(num_spatial_dims=2, domain_extent=1.0, num_points=16, dt=0.1, diffusivity=0.01)

def test_diffusion_invalid_diffusivity():
    with pytest.raises(ValueError, match="diffusivity matrix must be symmetric"):
        Diffusion(num_spatial_dims=2, domain_extent=1.0, num_points=16, dt=0.1, diffusivity=jnp.array([[0.1, 0.2], [0.1, 0.1]]))
