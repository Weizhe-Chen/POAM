import os
import pickle
import pyvista as pv
import numpy as np
import src
# from scipy.interpolate import RegularGridInterpolator


env_extent = [-31.0, 31.0, -31.0, 31.0]
min_lengthscale = 0.05
max_lengthscale = 0.5
camera_position = (30, -82.6, 116.2)
camera_focal_point = (-4.25, 4.33, 2.12)
camera_up = (-0.07, 0.78, 0.63)
plot_min = 0.0
plot_max = 25.0
image_path = os.path.expanduser("~/Projects/RSS2024/paper/figures/images")


def get_results(run_id):
    with open(f"./outputs/{run_id}/evaluator.pkl", "rb") as f:
        evaluator = pickle.load(f)
    mean = evaluator.mean
    lenscale = evaluator.lenscale
    std = evaluator.std
    abs_error = evaluator.abs_error
    x = evaluator.x_train
    z = evaluator.x_inducing
    return mean, lenscale, std, abs_error, x, z


def visualize_map_path(map, points, save_path: str = None) -> None:
    plotter = pv.Plotter()
    # plot lawnmower path
    max_z = np.max(map)
    waypoints = np.hstack([points, max_z * np.ones((len(points), 1))])
    spline = pv.Spline(waypoints[:-8], 1500)
    spline["scalars"] = np.arange(spline.n_points)
    tube = spline.tube(radius=0.2)
    plotter.add_mesh(tube, color="grey", lighting=True)
    # plot robot
    rotate_degree = 180 / np.pi * np.arctan2(
        waypoints[-1, 1] - waypoints[-2, 1], waypoints[-1, 0] - waypoints[-2, 0]
    )
    robot_mesh = pv.read("../../data/meshes/robot.stl")
    robot_mesh = robot_mesh.scale(5).translate([0, 0, max_z])
    robot_mesh = robot_mesh.rotate_z(rotate_degree).translate(
        (waypoints[-1, 0], waypoints[-1, 1], 0)
    )
    plotter.add_mesh(robot_mesh, color="orange", lighting=True)
    # plot environment
    map = map[:, :, np.newaxis]
    x_range = env_extent[1] - env_extent[0]
    y_range = env_extent[3] - env_extent[2]
    z_range = np.max(map) - np.min(map)
    resolution = [x_range / map.shape[0], y_range / map.shape[1], z_range]
    origin = np.array([env_extent[0], env_extent[2], 0])
    image_data = pv.ImageData(dimensions=map.shape, origin=origin, spacing=resolution)
    image_data.point_data["values"] = map.flatten()
    terrain = image_data.warp_by_scalar()
    plotter.add_mesh(terrain, cmap="jet", lighting=True, show_scalar_bar=False)
    # plot data points
    # plotter.add_points(
    #     points,
    #     point_size=5.0,
    #     render_points_as_spheres=True,
    #     scalars=points[:, 2],
    #     cmap="jet",
    #     show_scalar_bar=False,
    # )
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
            point_size=8.0,
            render_points_as_spheres=True,
            color="black",
            show_scalar_bar=False,
        )
    # plotter.add_axes()
    plotter.add_bounding_box(line_width=5.0, color="black")
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.up = camera_up
    plotter.show()
    if save_path is not None:
        plotter.screenshot(save_path)


def main():
    fix_scaler = src.scalers.MinMaxScaler(plot_min, plot_max, 0, 255)
    # fit_scaler = src.scalers.MinMaxScaler(plot_min, plot_max)
    ls_ranges = 30 * np.array([min_lengthscale, max_lengthscale])
    ls_scaler = src.scalers.MinMaxScaler(
        *ls_ranges, min_lengthscale, max_lengthscale
    )
    coords = ["n44w111", "n17e073", "n47w124", "n35w107"]
    models = ["ssgp++", "ovc", "ovc++", "poam"]
    for coord in coords:
        for model in models:
            mean, ls, std, err, x, z = get_results(f"{coord}_{model}_1")
            mean_plot = fix_scaler.preprocess(mean)
            visualize_map_path(mean_plot, x, save_path=f"{image_path}/mean_{coord}_{model}.png")
            ls_plot = ls_scaler.preprocess(ls)
            inducing_points = np.hstack([z, 30 * max_lengthscale * np.ones((len(z), 1))])
            visualize(ls_plot, inducing_points, show_bar=True, clim=ls_ranges, save_path=f"{image_path}/ls_{coord}_{model}.png")

            # fit_scaler.fit(std)
            # std_plot = fit_scaler.preprocess(std)
            # # visualize(std_plot)
            # fit_scaler.fit(err)
            # err_plot = fit_scaler.preprocess(err)
            # visualize(err_plot)


if __name__ == "__main__":
    main()
