import pickle
from pathlib import Path
from time import time

import gpytorch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
from ablation_model import AblationModel

import src


def get_map(cfg):
    with np.load(f"../../data/arrays/{cfg.map.geo_coordinate}.npz") as data:
        map = data["arr_0"]
    print(f"Loaded map with shape {map.shape}.")
    return map


def get_sensor(cfg, map, rng):
    rate = cfg.sensor.sensing_rate
    noise_scale = cfg.sensor.noise_scale
    sensor = src.sensors.PointSensor(
        matrix=map,
        env_extent=cfg.map.env_extent,
        rate=rate,
        noise_scale=noise_scale,
        rng=rng,
    )
    print(f"Initialized sensor with rate {rate} and noise scale {noise_scale}.")
    return sensor


def get_robot(cfg, sensor):
    init_state = np.array([cfg.map.task_extent[1], cfg.map.task_extent[2], -np.pi])
    robot = src.robots.DiffDriveRobot(
        sensor=sensor,
        state=init_state,
        control_rate=cfg.robot.control_rate,
        max_lin_vel=cfg.robot.max_lin_vel,
        max_ang_vel=cfg.robot.max_ang_vel,
        goal_radius=cfg.robot.goal_radius,
    )
    print(f"Initialized robot.")
    return robot


def get_planner(cfg, rng):
    if cfg.planner.name == "max_entropy":
        planner = src.planners.MaxEntropyPlanner(
            cfg.map.task_extent, rng, cfg.planner.num_candidates
        )
    else:
        raise ValueError(f"Unknown planner: {cfg.planner.name}")
    print(f"Initialized planner {cfg.planner.name}.")
    return planner


def get_visualizer(cfg, map):
    visualizer = src.utils.Visualizer(
        cfg.map.env_extent, cfg.map.task_extent, cfg.plot_robot_interval
    )
    vmin, vmax = np.min(map), np.max(map)
    visualizer.vmins[1], visualizer.vmaxs[1] = vmin, vmax
    visualizer.plot_image(
        index=0, matrix=map, title="Ground Truth", vmin=vmin, vmax=vmax
    )
    print(f"Initialized visualizer.")
    return visualizer


def get_evaluator(cfg, sensor):
    evaluator = src.utils.Evaluator(sensor, cfg.map.task_extent, cfg.eval_grid)
    print(f"Initialized evaluator.")
    return evaluator


def pilot_survey(cfg, robot, rng):
    bezier_planner = src.planners.BezierPlanner(cfg.map.task_extent, rng)
    goals = bezier_planner.plan(num_points=cfg.num_bezier_points)
    robot.goals = goals
    while len(robot.goals) > 0:
        robot.step()
    x_init, y_init = robot.commit_samples()
    print(f"Collected {len(x_init)} samples in pilot survey.")
    return x_init, y_init


def get_scalers(cfg, x_init, y_init):
    x_scaler = src.scalers.MinMaxScaler()
    x_scaler.fit(x_init)
    y_scaler = src.scalers.StandardScaler()
    y_scaler.fit(y_init)
    return x_scaler, y_scaler


def get_kernel(cfg, x_init):
    if cfg.kernel.name == "rbf":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        kernel.base_kernel.lengthscale = cfg.kernel.lengthscale
        kernel.outputscale = cfg.kernel.outputscale
    elif cfg.kernel.name == "ak":
        kernel = gpytorch.kernels.ScaleKernel(
            src.kernels.AttentiveKernel(
                dim_input=x_init.shape[1],
                dim_hidden=cfg.kernel.dim_hidden,
                dim_output=cfg.kernel.dim_output,
                min_lengthscale=cfg.kernel.min_lengthscale,
                max_lengthscale=cfg.kernel.max_lengthscale,
            )
        )
    else:
        raise ValueError(f"Unknown kernel: {cfg.kernel}")
    print(f"Initialized kernel {cfg.kernel.name}.")
    return kernel


def get_model(cfg, x_init, y_init, x_scaler, y_scaler, kernel):
    model = AblationModel(
        num_inducing=cfg.model.num_inducing,
        learn_inducing=cfg.model.learn_inducing,
        learn_variational=cfg.model.learn_variational,
        x_train=x_init,
        y_train=y_init,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        kernel=kernel,
        noise_variance=cfg.likelihood.noise_variance,
        batch_size=cfg.model.batch_size,
        jitter=cfg.model.jitter,
        use_online_elbo=cfg.model.use_online_elbo,
    )
    print(f"Initialized model {cfg.model.name}.")
    return model


def model_update(cfg, model, evaluator):
    print("Updating model...")
    start = time()
    model.update_inducing(name=cfg.model.name)
    if not cfg.model.learn_variational:
        if "ssgp" in cfg.model.name:
            model.update_variational("ssgp")
        else:
            model.update_variational()
    losses = model.optimize(num_steps=cfg.num_train_steps)
    end = time()
    evaluator.training_times.append(end - start)
    evaluator.losses.extend(losses)


def evaluation(model, evaluator):
    print("Evaluating model...")
    start = time()
    mean, std = model.predict(evaluator.eval_inputs)
    end = time()
    evaluator.prediction_times.append(end - start)
    evaluator.compute_metrics(mean, std)


def visualization(visualizer, evaluator, x_inducing=None):
    print(f"Visualizing results...")
    visualizer.plot_prediction(evaluator.mean, evaluator.std, evaluator.abs_error)
    visualizer.plot_data(evaluator.x_train)
    if x_inducing is not None:
        visualizer.plot_inducing_inputs(x_inducing)
    visualizer.plot_metrics(evaluator)


def information_gathering(robot, model, planner, visualizer=None):
    while True:
        print("Planning...")
        goal = planner.plan(model, robot.state[:2])
        if visualizer is not None:
            visualizer.plot_goal(goal)
            visualizer.pause()
        robot.goals = goal
        plot_counter = 0
        print("Sampling...")
        while robot.has_goals:
            plot_counter += 1
            robot.step()
            if visualizer is None:
                continue
            if visualizer.interval > 0 and plot_counter % visualizer.interval == 0:
                visualizer.plot_robot(robot.state, scale=4)
                visualizer.pause()
        if len(robot.sampled_observations) > 0:
            x_new, y_new = robot.commit_samples()
            return x_new, y_new


def save_evaluator(evaluator, save_path):
    with open(save_path, "wb") as file:  # Overwrites any existing file.
        pickle.dump(evaluator, file, pickle.HIGHEST_PROTOCOL)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # experiment setup
    rng = src.utils.set_random_seed(cfg.seed)
    map = get_map(cfg)
    sensor = get_sensor(cfg, map, rng)
    robot = get_robot(cfg, sensor)
    planner = get_planner(cfg, rng)

    # visualization and evaluation
    if cfg.visualize:
        visualizer = get_visualizer(cfg, map)
    evaluator = get_evaluator(cfg, sensor)

    # pilot survey
    x_init, y_init = pilot_survey(cfg, robot, rng)
    evaluator.add_data(x_init, y_init)
    x_scaler, y_scaler = get_scalers(cfg, x_init, y_init)
    kernel = get_kernel(cfg, x_init)
    model = get_model(cfg, x_init, y_init, x_scaler, y_scaler, kernel)
    # First optimization takes longer time, which affects the training time evaluation.
    model.optimize(num_steps=1)
    model_update(cfg, model, evaluator)
    evaluation(model, evaluator)
    if cfg.visualize:
        visualization(visualizer, evaluator)
        print("Press any key to continue and [ESC] to exit...")
        plt.waitforbuttonpress()

    # main loop
    decision_epoch = 0
    start_time = time()
    while True:
        num_samples = len(evaluator.y_train)
        if num_samples >= cfg.max_num_samples:
            break
        time_elapsed = time() - start_time
        decision_epoch += 1

        print(
            f"Decision epoch: {decision_epoch} | "
            + f" Time used: {time_elapsed:.2f} seconds | "
            + f"Number of samples: {num_samples} / {cfg.max_num_samples}"
        )

        x_new, y_new = information_gathering(
            robot, model, planner, visualizer if cfg.visualize else None
        )

        evaluator.add_data(x_new, y_new)
        model.add_data(x_new, y_new)
        model_update(cfg, model, evaluator)
        evaluation(model, evaluator)

        if not cfg.visualize:
            continue
        visualizer.clear()
        visualizer.plot_title(decision_epoch, time_elapsed)
        visualization(
            visualizer,
            evaluator,
            model.x_inducing if hasattr(model, "x_inducing") else model.x_train,
        )
        if cfg.kernel.name == "ak":
            visualizer.plot_lengthscales(
                model, evaluator, cfg.kernel.min_lengthscale, cfg.kernel.max_lengthscale
            )
        visualizer.pause()
    print("Done!")
    if cfg.visualize:
        visualizer.show()

    # save results
    if cfg.save_results:
        if cfg.kernel.name == "ak":
            lenscale = model.get_ak_lengthscales(evaluator.eval_inputs).reshape(
                *evaluator.eval_grid
            )
            evaluator.lenscale = lenscale
        if hasattr(model, "x_inducing"):
            evaluator.x_inducing = model.x_inducing
        run_id = f"{cfg.map.geo_coordinate}_{cfg.model.name}_{cfg.seed}"
        save_path = f"{cfg.output_folder}/{run_id}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_evaluator(evaluator, f"{save_path}/evaluator.pkl")
        OmegaConf.save(cfg, f"{save_path}/config.yaml")
        print(f"Results saved to {save_path}.")


if __name__ == "__main__":
    main()
