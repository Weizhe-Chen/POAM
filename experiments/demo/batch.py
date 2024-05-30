import numpy as np
import pyvista as pv
import torch
import vtk
import src
import gpytorch
from matplotlib import pyplot as plt

seed = 0
num_train_points = 5000
env_name = "n44w111"
env_extent = [-31.0, 31.0, -31.0, 31.0]
task_extent = [-30, 30, -30, 30]
eval_grid = [100, 100]
sensing_rate = 3
noise_scale = 1.0
control_rate = 10.0
max_lin_vel = 1.0
max_ang_vel = 1.0
goal_radius = 0.1
num_waypoints = 35
image_path = f"~/Projects/RSS2024/paper/figures/images"
# camera
camera_position = (30, -82.6, 116.2)
camera_focal_point = (-4.25, 4.33, 2.12)
camera_up = (-0.07, 0.78, 0.63)
# kernel
dim_hidden = 10
dim_output = 10
min_lengthscale = 0.05
max_lengthscale = 0.5
# model
num_inducing = 500
learn_inducing = True
learn_variational = True
noise_variance = 1.0
batch_size = 256
jitter = 1e-6


def visualize_env(env: np.ndarray, save_path: str = None) -> None:
    plotter = pv.Plotter()
    env = env[:, :, np.newaxis]
    x_range = env_extent[1] - env_extent[0]
    y_range = env_extent[3] - env_extent[2]
    z_range = np.max(env) - np.min(env)
    resolution = [x_range / env.shape[0], y_range / env.shape[1], z_range]
    origin = np.array([env_extent[0], env_extent[2], 0])
    image_data = pv.ImageData(dimensions=env.shape, origin=origin, spacing=resolution)
    image_data.point_data["values"] = env.flatten()
    terrain = image_data.warp_by_scalar()
    plotter.add_mesh(terrain, cmap="jet", lighting=True, show_scalar_bar=False)
    plotter.add_axes()
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.up = camera_up
    plotter.show()
    if save_path is not None:
        plotter.screenshot(save_path)


def visualize_points(env, points, save_path: str = None) -> None:
    plotter = pv.Plotter()
    # plot lawnmower path
    max_z = np.max(points[:, 2])
    waypoints = np.hstack([points[:, :2], max_z * np.ones((len(points), 1))])
    spline = pv.Spline(waypoints, 1500)
    spline["scalars"] = np.arange(spline.n_points)
    tube = spline.tube(radius=0.1)
    plotter.add_mesh(tube, color="grey", lighting=True)
    # plot robot
    robot_mesh = pv.read("../../data/meshes/robot.stl")
    robot_mesh = robot_mesh.scale(5).translate([0, 0, max_z])
    robot_mesh = robot_mesh.rotate_z(90).translate(
        (waypoints[-1, 0], waypoints[-1, 1], 0)
    )
    plotter.add_mesh(robot_mesh, color="orange", lighting=True)

    # plot environment
    env = env[:, :, np.newaxis]
    x_range = env_extent[1] - env_extent[0]
    y_range = env_extent[3] - env_extent[2]
    z_range = np.max(env) - np.min(env)
    resolution = [x_range / env.shape[0], y_range / env.shape[1], z_range]
    origin = np.array([env_extent[0], env_extent[2], 0])
    image_data = pv.ImageData(dimensions=env.shape, origin=origin, spacing=resolution)
    image_data.point_data["values"] = env.flatten()
    terrain = image_data.warp_by_scalar()
    plotter.add_mesh(
        terrain,
        color="grey",
        lighting=False,
        opacity=0.1,
        style="wireframe",
    )

    # plot data points
    plotter.add_points(
        points,
        point_size=5.0,
        render_points_as_spheres=True,
        scalars=points[:, 2],
        cmap="jet",
        show_scalar_bar=False,
    )
    plotter.add_axes()
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.up = camera_up
    plotter.show()
    if save_path is not None:
        plotter.screenshot(save_path)


def visualize(
    matrix: np.ndarray,
    points: np.ndarray = None,
    save_path: str = None,
    show_bar: bool = False,
    clim: tuple = None,
) -> None:
    plotter = pv.Plotter()
    matrix = matrix[:, :, np.newaxis]
    x_range = env_extent[1] - env_extent[0]
    y_range = env_extent[3] - env_extent[2]
    z_range = np.max(matrix) - np.min(matrix)
    resolution = [x_range / matrix.shape[0], y_range / matrix.shape[1], z_range]
    origin = np.array([env_extent[0], env_extent[2], 0])
    image_data = pv.ImageData(
        dimensions=matrix.shape, origin=origin, spacing=resolution
    )
    image_data.point_data["values"] = matrix.flatten()
    terrain = image_data.warp_by_scalar()
    if clim is None:
        clim = [0, 25]
    sargs = dict(
        height=0.25,
        vertical=True,
        position_x=0.85,
        position_y=0.05,
        fmt="%.1f",
        title="",
    )
    plotter.add_mesh(
        terrain,
        cmap="jet",
        lighting=True,
        show_scalar_bar=show_bar,
        scalar_bar_args=sargs,
        clim=clim,
    )
    if points is not None:
        plotter.add_points(
            points,
            point_size=5.0,
            render_points_as_spheres=True,
            color="black",
            show_scalar_bar=False,
        )
    plotter.add_axes()
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.up = camera_up
    plotter.show()
    if save_path is not None:
        plotter.screenshot(save_path)


def main() -> None:
    rng = src.utils.set_random_seed(seed)

    with np.load(f"../../data/arrays/{env_name}.npz") as data:
        env = data["arr_0"]
    print(f"Loaded environment {env_name} with shape {env.shape}.")


    sensor = src.sensors.PointSensor(
        matrix=env,
        env_extent=env_extent,
        rate=sensing_rate,
        noise_scale=noise_scale,
        rng=rng,
    )
    print(f"Initialized sensor with rate {sensing_rate} and noise scale {noise_scale}.")

    robot = src.robots.DiffDriveRobot(
        sensor=sensor,
        state=np.array([task_extent[0], task_extent[2], np.pi / 2]),
        control_rate=control_rate,
        max_lin_vel=max_lin_vel,
        max_ang_vel=max_ang_vel,
        goal_radius=goal_radius,
    )
    print(f"Initialized robot with control rate {control_rate}.")

    planner = src.planners.LawnmowerPlanner(task_extent, rng)
    goals = planner.plan(num_points=num_waypoints)
    robot.goals = goals
    while len(robot.goals) > 0:
        robot.step()
    x_train, y_train = robot.commit_samples()
    print(f"Collected {len(x_train)} samples.")

    # x_train = np.random.uniform(
    #     low=[task_extent[0], task_extent[2]],
    #     high=[task_extent[1], task_extent[3]],
    #     size=(num_train_points, 2),
    # )
    # y_train = sensor.sense(x_train)
    # indices = np.argsort(x_train[:, 0])
    # x_train = x_train[indices]
    # y_train = y_train[indices]
    num_train_per_batch = int(np.ceil(num_train_points / 4))

    evaluator = src.utils.Evaluator(sensor, task_extent, eval_grid)
    x_test = evaluator.eval_inputs
    y_test = evaluator.eval_outputs

    x_scaler = src.scalers.MinMaxScaler()
    x_scaler.fit(x_test)
    y_scaler = src.scalers.StandardScaler()
    y_scaler.fit(y_test)

    kernel = gpytorch.kernels.ScaleKernel(
        src.kernels.AttentiveKernel(
            dim_input=x_train.shape[1],
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            min_lengthscale=min_lengthscale,
            max_lengthscale=max_lengthscale,
        )
    )
    mean_scaler = src.scalers.MinMaxScaler(0.0, 25.0, np.min(env), np.max(env))
    std_scaler = src.scalers.MinMaxScaler(0.0, 25.0)
    error_scaler = src.scalers.MinMaxScaler(0.0, 25.0)
    lenscale_scaler = src.scalers.MinMaxScaler(
        30 * min_lengthscale, 30 * max_lengthscale, min_lengthscale, max_lengthscale
    )
    env_plot = mean_scaler.preprocess(env)

    for i in range(4):
        batch_start = i * num_train_per_batch
        batch_end = (i + 1) * num_train_per_batch
        if i == 3:
            x_batch = x_train[batch_start:]
            y_batch = y_train[batch_start:]
        else:
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

        y_batch_plot = mean_scaler.preprocess(y_batch)
        points = np.hstack((x_batch, y_batch_plot))
        visualize_points(env_plot, points, f"{image_path}/demo_batch_data_{i}.png")

        if i == 0:
            model = src.models.PAMModel(
                num_inducing,
                x_batch,
                y_batch,
                x_scaler,
                y_scaler,
                kernel,
                noise_variance,
                batch_size,
                jitter,
            )
            model.update_variational()
        else:
            model.add_data(x_batch, y_batch)

        all_losses = []
        for _ in range(20):
            losses = model.optimize(num_steps=50)
            all_losses.extend(losses)
            model.update_inducing()
            model.update_variational()

        _, ax = plt.subplots()
        ax.plot(all_losses)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        plt.show()

        mean, std = model.predict(x_test)
        mean_plot = mean_scaler.preprocess(mean).reshape(*eval_grid)
        inducing_points = np.hstack([model.x_inducing, 25 * np.ones((num_inducing, 1))])
        visualize(
            mean_plot,
            inducing_points,
            save_path=f"{image_path}/demo_batch_pam_mean_{i}.png",
        )
        std_scaler.fit(std)
        std_plot = std_scaler.preprocess(std).reshape(*eval_grid)
        visualize(
            std_plot,
            save_path=f"{image_path}/demo_batch_pam_std_{i}.png",
        )
        abs_error = np.abs(mean - y_test)
        error_scaler.fit(abs_error)
        abs_error_plot = error_scaler.preprocess(abs_error).reshape(*eval_grid)
        visualize(
            abs_error_plot,
            save_path=f"{image_path}/demo_batch_pam_error_{i}.png",
        )
        lenscale = model.get_ak_lengthscales(x_test)
        lenscale_plot = lenscale_scaler.preprocess(lenscale).reshape(*eval_grid)
        visualize(
            lenscale_plot,
            show_bar=True,
            clim=[lenscale_scaler.expected_min, lenscale_scaler.expected_max],
            save_path=f"{image_path}/demo_batch_pam_lenscale_{i}.png",
        )


if __name__ == "__main__":
    main()
