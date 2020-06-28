/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

#define EPS 0.00001
#define DEBUG false

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;

  // Init distributions
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Gen particles
  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Init distributions
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for (Particle& p : particles) {
    if (fabs(yaw_rate) < EPS) {
      // Yaw didnt change
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    } else {
      // Yaw changed
      p.x += velocity / yaw_rate * ( sin( p.theta + yaw_rate * delta_t ) -
                                                sin( p.theta ) );
      p.y += velocity / yaw_rate * ( cos( p.theta ) -
                                                cos( p.theta + yaw_rate * delta_t ) );
      p.theta += yaw_rate * delta_t;
    }

    // Add noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (LandmarkObs& observation : observations) {

    double minPredDist = std::numeric_limits<double>::max();
    int minPredId = -1;

    for (LandmarkObs prediction : predicted) {
      double xDist = observation.x - prediction.x;
      double yDist = observation.y - prediction.y;

      double dist = xDist*xDist + yDist*yDist;

      if (dist < minPredDist) {
        minPredDist = dist;
        minPredId = prediction.id;
      }
    }

    observation.id = minPredId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double stdLmRange_Bearing = std_landmark[0]*std_landmark[1];
  double stdLmRange_2 = std_landmark[0]*std_landmark[0];
  double stdLmBearing_2 = std_landmark[1]*std_landmark[1];

  for (Particle& p : particles) {
    // Find landmarks in particles range
    double sensor_range_2 = sensor_range * sensor_range;
    vector<LandmarkObs> inRangeLandmarks;

    for (Map::single_landmark_s lm : map_landmarks.landmark_list) {
      double dX = p.x - lm.x_f;
      double dY = p.y - lm.y_f;
      if (dX*dX + dY*dY <= sensor_range_2) {
        inRangeLandmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }

    // Transform observation car coords to map coords
    vector<LandmarkObs> mappedObservations;
    for (LandmarkObs lmo: observations) {
      double mapX = cos(p.theta)*lmo.x - sin(p.theta)*lmo.y + p.x;
      double mapY = sin(p.theta)*lmo.x + cos(p.theta)*lmo.y + p.y;
      mappedObservations.push_back(LandmarkObs{lmo.id, mapX, mapY});
    }

    dataAssociation(inRangeLandmarks, mappedObservations);

    // Update weight
    if (DEBUG)
      std::cout << p.id << ": old_weight=" << p.weight;
    p.weight = 1.0;
    for (LandmarkObs obs : mappedObservations) {
      // Find associated landmark
      // TODO: use hashmap or direct assignment of landmark to observation in dataAssociation()
      // to achieve O(1) speed instead of O(n).
      LandmarkObs associated_lm;
      for (LandmarkObs lm : inRangeLandmarks) {
        if (lm.id == obs.id) {
          associated_lm = lm;
          break;
        }
      }

      double dX = obs.x - associated_lm.x;
      double dY = obs.y - associated_lm.y;
      double lm_weight = ( exp( -( dX*dX/(2*stdLmRange_2) + (dY*dY/(2*stdLmBearing_2)) ) /
                             (2*M_PI*stdLmRange_Bearing)) );
      if (lm_weight == 0)
        p.weight *= EPS;
//        continue;
      else
        p.weight *= lm_weight;
    }

    if (DEBUG)
      std::cout << "; new_weight=" << p.weight << std::endl;
  }

}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Get max weight.
  double maxWeight = std::numeric_limits<double>::min();
  for(Particle p : particles) {
    if ( p.weight > maxWeight ) {
      maxWeight = p.weight;
    }
  }

  // Creating distributions.
  std::default_random_engine gen;
  std::uniform_real_distribution<double> distDouble(0.0, maxWeight * 2);
  std::uniform_int_distribution<int> distInt(0, num_particles - 1);

  // init resampling wheel
  int index = distInt(gen);
  double beta = 0.0;

  // rotate the wheel
  vector<Particle> resampledParticles;
  for(int i = 0; i < num_particles; i++) {
    beta += distDouble(gen);
    while( beta > particles[index].weight) {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    if (DEBUG)
      std::cout << "chosen particle: " << particles[index].id << "(w=" << particles[index].weight << ")" << std::endl;
    Particle new_p = {.id=i,
                      .x = particles[index].x,
                      .y = particles[index].y,
                      .theta = particles[index].theta,
                      .weight = particles[index].weight};
    resampledParticles.push_back(new_p);
  }

  particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
