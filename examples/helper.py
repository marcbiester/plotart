import numpy as np


def create_random_config(
    grid,
    n_vort=-1,
    n_source_sink=-1,
    v_strength=50,
    ss_strength=5,
    n_streamtraces=-1,
    seed_setting=[],
    max_iter=-1,
    seed_radius=-1,
    max_sink_dist=-1,
) -> dict:
    """
    function to create a random vonfig that is later used to create a streamline plot. Negative parameter values will create up to 20 random inputs.
    """
    params = {
        "n_streamtraces": n_streamtraces,
        "max_iter": max_iter,
        "vortices": [],
        "source_sink": [],
        "seed_setting": seed_setting,
        "seed_radius": seed_radius,
        "max_sink_dist": max_sink_dist,
    }

    if n_streamtraces < 0:
        params["n_streamtraces"] = np.random.randint(100, 300)

    if max_iter < 0:
        params["max_iter"] = np.random.randint(200, 1000)

    if max_sink_dist < 0:
        params["max_sink_dist"] = np.random.uniform(0.5, 3.0)

    if seed_radius < 0:
        params["seed_radius"] = np.random.uniform(0.2, 1.0)

    if n_vort < 0:
        n_vort = np.random.randint(1, 20)
    if n_source_sink < 0:
        n_source_sink = np.random.randint(1, 20)

    if not seed_setting:
        params["seed_setting"] = np.random.choice(
            ["random", "sources", "grid"], np.random.randint(1, 4)
        ).tolist()

    for i in range(n_vort):
        x, y = np.random.uniform(grid["x_start"], grid["x_end"]), np.random.uniform(
            grid["y_start"], grid["y_end"]
        )
        vortex_strength = np.random.uniform(-v_strength, v_strength)
        source_sink_strength = np.random.uniform(-ss_strength, ss_strength)
        if abs(source_sink_strength) < 3:
            source_sink_strength = (
                source_sink_strength / abs(source_sink_strength) * 5
            )  # avoid too little source or think so that streamlines just circle. Keep direction

        params["vortices"].append((vortex_strength, x, y))
        params["source_sink"].append((source_sink_strength, x, y))

    for i in range(n_source_sink):
        x, y = np.random.uniform(grid["x_start"], grid["x_end"]), np.random.uniform(
            grid["y_start"], grid["y_end"]
        )
        source_sink_strength = np.random.uniform(-ss_strength, ss_strength)
        params["source_sink"].append((source_sink_strength, x, y))

    return params
