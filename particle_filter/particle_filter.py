import numpy as np 
from copy import deepcopy 
from scipy.stats import multivariate_normal
import math 

"""
Methodology: 
1. Initialize particles 
2. Predict step: Motion model step the particles
3. Update step: Use the april tag positions to update the particle locations
4. Resample step: resample the particles. 


NOTE: add theta later? 
"""
class LocalizationFilter:
    def __init__(self, init_robot_position, 
                #  known_tag_ids, init_known_tag_positions, 
                 num_particles): 
        
        self.init_robot_position = init_robot_position 
        # self.known_tag_ids = known_tag_ids
        # self.init_known_tag_positions = init_known_tag_positions
        self.num_particles = num_particles

        self.particles = None 
        self.particle_weights = None 

        # Parameters to change
        self.grid_size = [[-5, 5], [-5, 5]] # [[x start, x end], [y start, y end]], []

        self.motion_mean = np.array([0, 0])   # observation noise mean 
        motion_var = 1e-2
        self.motion_cov = np.array([[motion_var, 0],  # observation noise covariance
                                 [0, motion_var]])
        self.motion_cov_det = np.linalg.det(self.motion_cov)
        self.inv_motion_cov = np.linalg.inv(self.motion_cov)

        self.obs_mean = np.array([0, 0])   # observation noise mean 
        obs_var = 2
        self.obs_cov = np.array([[obs_var, 0],  # observation noise covariance
                                 [0, obs_var]])
        self.obs_cov_det = np.linalg.det(self.obs_cov)
        self.inv_obs_cov = np.linalg.inv(self.obs_cov)
        
        self.init_particles()
        return 

    def reinit_particles(self):
        self.init_particles()
        return

    def init_particles(self): 
        """
        Initialize particles
        """
        initialization_type = "uniform"
        self.particles = []
        for particle_num in range(self.num_particles): 
            if initialization_type == "uniform": 
                x = np.random.uniform(self.grid_size[0][0], self.grid_size[0][1])
                y = np.random.uniform(self.grid_size[1][0], self.grid_size[1][1])

                theta = np.random.uniform(0, 2*np.pi)
                self.particles.append([x, y])
            else: 
                raise NotImplementedError("Initialization type not implemented")

        self.particles = np.array(self.particles)
        self.particle_weights = np.ones(self.num_particles) / self.num_particles
        return 

    def predict_step(self, delta_action): 
        """
        Motion model step the particles
        args: 
            - delta_action: [delta x, delta y, delta theta ?]
        """

        all_noise = np.random.multivariate_normal(self.motion_mean, self.motion_cov, size=self.num_particles)
        self.particles = np.array(self.particles) + delta_action.reshape((1, -1)) + all_noise

        # INEFFICIENT: but working 
        # for particle_num in range(len(self.particles)): 
        #     self.particles[particle_num][0] += delta_action[0]
        #     self.particles[particle_num][1] += delta_action[1]
        #     # self.particles[particle_num][2] = (self.particles[particle_num][2] + delta_action[2]) % (2*np.pi) # NOTE: theta

        #     # inject motion noise 
        #     self.particles[particle_num] = np.random.multivariate_normal(np.array(self.particles[particle_num]) + self.motion_mean, self.motion_cov)

        return
    
    def update_step(self, estimated_robot_position): 
        """
        Use the robot position meand from the estimated april tags to update the particle weights 
        and then update the particles via resampling
        """

        pdf_normalizer = 1/(2*np.pi*np.sqrt(self.obs_cov_det))
        pdf_exp = -0.5 * np.einsum('ij,ij->i', np.einsum('ik,kj->ij', self.particles - estimated_robot_position.reshape((1, -1)) ,  self.inv_obs_cov), (self.particles - estimated_robot_position.reshape((1, -1))))
        obs_weight_update_multipliers = pdf_normalizer * np.exp(pdf_exp)
        updated_particle_weights = self.particle_weights * obs_weight_update_multipliers

        # INEFFICIENT: but working
        # updated_particle_weights = deepcopy(self.particle_weights)
        # for particle_num in range(len(self.particles)): 
            
        #     obs_weight_update_multiplier = get_pdf(observed_position=np.array(estimated_robot_position), 
        #                                                 particle_mean=np.array(self.particles[particle_num]), 
        #                                                 particle_inv_cov=self.inv_obs_cov, 
        #                                                 particle_obs_cov_det=self.obs_cov_det)
        #     updated_particle_weights[particle_num] *= obs_weight_update_multiplier
            

        # normalize the weights 
        updated_particle_weights = np.array(updated_particle_weights)
        updated_particle_weights /= np.sum(updated_particle_weights)

        self.particle_weights = deepcopy(updated_particle_weights)

        # Resampling 
        tmp1=[val**2 for val in self.particle_weights]
        Neff=1/(np.array(tmp1).sum())

        if Neff < self.num_particles/3: # resample 
            # first resampling approach - resampling according to the probabilities stored in the weights
            resampledStateIndex=np.random.choice(np.arange(self.num_particles), self.num_particles, p=self.particle_weights, replace=True)

            # second resampling approach - systematic resampling
            # resampledStateIndex=systematicResampling(self.particle_weights)

            new_particles = self.particles[resampledStateIndex]
            new_particle_weights = self.particle_weights[resampledStateIndex]
            # normalize new particle weights
            new_particle_weights = new_particle_weights/np.sum(new_particle_weights)
        
            self.particles = deepcopy(new_particles)
            self.particle_weights = deepcopy(new_particle_weights)

        return

    ############## Interface ################

    def get_robot_position(self): 
        """
        Get the robot position from the particles
        """
        pos = np.array([0.0, 0.0])
        for particle_num in range(len(self.particles)): 
            pos += self.particles[particle_num] * self.particle_weights[particle_num]
        return pos 
        # return np.mean(self.particles*self.particle_weights.reshape((-1, 1)), axis=0)

    def predict_update_position(self, delta_action, estimated_robot_position):
        """
        Predict and update the robot position
        """
        self.predict_step(delta_action)
        self.update_step(estimated_robot_position)
        return self.get_robot_position()

############### Helper Functions ################
def get_pdf(observed_position, particle_mean, particle_inv_cov, particle_obs_cov_det): 
    """
    Get the probability density function of the particle position given the mean and covariance
    """
    distrib = multivariate_normal(mean=particle_mean, cov=particle_inv_cov)

    # testthing = 1/(2*np.pi*np.sqrt(particle_obs_cov_det)) * np.exp(-0.5 * (observed_position - particle_mean).T @ particle_inv_cov @ (observed_position - particle_mean))
    # return testthing

    output =  distrib.pdf(observed_position)
    return output
    

def systematicResampling(weightArray):
    # N is the total number of samples
    N=len(weightArray)
    # cummulative sum of weights
    cValues=[]
    cValues.append(weightArray[0])
 
    for i in range(N-1):
        cValues.append(cValues[i]+weightArray[i+1])
 
    # starting random point
    startingPoint=np.random.uniform(low=0.0, high=1/(N))
     
    # this list stores indices of resampled states
    resampledIndex=[]
    for j in range(N):
        currentPoint=startingPoint+(1/N)*(j)
        s=0
        while (currentPoint>cValues[s]):
            s=s+1
             
        resampledIndex.append(s)
 
    return resampledIndex
