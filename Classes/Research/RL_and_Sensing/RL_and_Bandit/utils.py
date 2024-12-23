import jax
import jax.numpy as jnp

def bresenham_line(tx, rx, map_size=20):
    """
    Compute a Bresenham line mask between two points on a 2D grid, JIT-enabled.

    Args:
        tx (tuple): Coordinates of the starting point (x1, y1).
        rx (tuple): Coordinates of the ending point (x2, y2).
        map_size (int): Size of the grid (default is 20).

    Returns:
        jax.numpy.ndarray: A 2D mask of shape (map_size, map_size) with 1s along the Bresenham line.
    """
    x1, y1 = tx
    x2, y2 = rx
    dx = jnp.abs(x2 - x1)
    dy = jnp.abs(y2 - y1)
    sx = jax.lax.cond(x1 < x2, lambda: 1, lambda: -1)
    sy = jax.lax.cond(y1 < y2, lambda: 1, lambda: -1)
    err = dx - dy
    initial_state = (x1, y1, err, jnp.zeros((map_size, map_size)))

    def step_fn(carry, _):
        x, y, err, mask = carry

        # Update mask for the current point
        mask = mask.at[y, x].set(1)

        # Compute e2
        e2 = 2 * err

        # Update x and y conditionally
        new_x = jax.lax.cond(e2 > -dy, lambda: x + sx, lambda: x)
        new_y = jax.lax.cond(e2 < dx, lambda: y + sy, lambda: y)

        # Update err
        new_err = jax.lax.cond(e2 > -dy, lambda: err - dy, lambda: err)
        new_err = jax.lax.cond(e2 < dx, lambda: new_err + dx, lambda: new_err)

        return (new_x, new_y, new_err, mask), None

    # Number of steps to include both start and end points
    num_steps = max(dx, dy) + 1

    # Run lax.scan
    final_state, _ = jax.lax.scan(step_fn, initial_state, None, length=num_steps)

    # Extract the mask
    _, _, _, mask = final_state
    return mask

if __name__ == "__main__":
    mask = bresenham_line((0, 0), (6, 6), map_size=10)
    print(mask)
