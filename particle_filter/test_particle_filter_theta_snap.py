"""
Test out the particle Filter
1. Visualize particles 
2. Initialize particles in environment 
3. Move the particles with delta action 
4. Updated particles with a fake noisy observation
"""

import matplotlib.pyplot as plt 
import numpy as np 
from particle_filter_theta_snap import LocalizationFilter_theta_snap

def visualize_particles(true_robot_position, particles, particle_weights, name): 
    """
    Visualize the particles
    """
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.scatter(particles[:, 0], particles[:, 1], c=particle_weights)
    plt.scatter(true_robot_position[0], true_robot_position[1], c='r', marker='x')
    plt.title(name)
    plt.show()
    return


if __name__ == "__main__": 
    x_grid = [-10, 10]
    y_grid = [-10, 10]

    init_robot_position = [1, 2]

    filter = LocalizationFilter_theta_snap(init_robot_position=init_robot_position, 
                                num_particles=2000)

    visualize_particles(init_robot_position, filter.particles, filter.particle_weights, "init")

    true_robot_state = init_robot_position
    for timestep in range(100):

        true_action = np.random.random(2)
        true_action = (true_action * 2) - 1
        true_action[0] *= 2
        true_action[1] *= 2

        # predict step 
        filter.predict_step(true_action, delta_theta=0.1)

        true_robot_state += true_action

        # visualize particles
        visualize_particles(true_robot_state, filter.particles, filter.particle_weights, name=f'Predict: {timestep}: {true_action}')

        # update step
        obs_noise = np.random.multivariate_normal(filter.obs_mean, filter.obs_cov)
        observed_state = true_robot_state + obs_noise
        filter.update_step(observed_state, theta_position=0.5)

        # visualize particles
        print(filter.particle_weights)
        update_name = f'Update: {timestep}: Observed: {observed_state}, True: {true_robot_state}, Estimated: {filter.get_robot_position()}'
        visualize_particles(true_robot_state, filter.particles, filter.particle_weights, name=update_name)
        