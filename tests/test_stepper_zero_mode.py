
import jax.numpy as jnp
import pytest
from exponax.stepper.generic import GeneralNonlinearStepper, NormalizedNonlinearStepper

def test_normalized_stepper_instantiation():
    # Just check if it accepts the argument
    stepper = NormalizedNonlinearStepper(
        num_spatial_dims=1,
        num_points=64,
        zero_mode_fix=False,
    )
    assert stepper.zero_mode_fix == False

    from exponax.stepper.generic import DifficultyNonlinearStepper
    stepper2 = DifficultyNonlinearStepper(
        num_spatial_dims=1,
        num_points=64,
        zero_mode_fix=False,
    )
    assert stepper2.zero_mode_fix == False

def run_simulation(zero_mode_fix: bool):
    num_points = 64
    num_spatial_dims = 1
    domain_extent = 2 * jnp.pi
    dt = 0.01

    # Coefficients: (b0, b1, b2)
    # We want b2=1.0, others 0.
    stepper = GeneralNonlinearStepper(
        num_spatial_dims=num_spatial_dims,
        domain_extent=domain_extent,
        num_points=num_points,
        dt=dt,
        linear_coefficients=(0.0,), # No linear term
        nonlinear_coefficients=(0.0, 0.0, 1.0), # Only gradient norm
        dealiasing_fraction=1.0,
        zero_mode_fix=zero_mode_fix,
    )

    # Initial condition: sin(x)
    x = jnp.linspace(0, domain_extent, num_points, endpoint=False)
    u = jnp.sin(x)
    u = u[None, :] # Add channel dimension (1, N)

    # Initial mean should be 0
    initial_mean = jnp.mean(u)

    # Step
    u_next = stepper.step(u)

    next_mean = jnp.mean(u_next)

    return initial_mean, next_mean

def test_zero_mode_fix_enabled():
    initial_mean, next_mean = run_simulation(zero_mode_fix=True)
    print(f"Enabled - Initial: {initial_mean}, Next: {next_mean}")
    assert jnp.abs(next_mean) < 1e-6, f"Expected zero mean change with zero_mode_fix=True, got {next_mean}"

def test_zero_mode_fix_disabled():
    initial_mean, next_mean = run_simulation(zero_mode_fix=False)
    print(f"Disabled - Initial: {initial_mean}, Next: {next_mean}")

    # u_t = 0.5 * (u_x)^2 = 0.5 * cos^2(x) = 0.25 + 0.25 cos(2x)
    # Mean(u_t) = 0.25
    # dt = 0.01
    # Mean change should be 0.0025

    # BUT wait, coefficients are negated in GeneralNonlinearFun!
    # GradientNormNonlinearFun: return -scale * ...
    # scale = -nonlinear_coefficients[2] = -1.0
    # return -(-1.0) * ... = 1.0 * ...
    # So u_t = + 0.5 * (u_x)^2.
    # So mean should INCREASE by 0.0025.

    expected_change = 0.0025
    change = next_mean - initial_mean

    assert jnp.abs(change - expected_change) < 1e-4, f"Expected mean change ~{expected_change}, got {change}"

if __name__ == "__main__":
    test_normalized_stepper_instantiation()
    test_zero_mode_fix_enabled()
    test_zero_mode_fix_disabled()
