/**
 * Classical Pendulum Simulation
 * 
 * This program simulates the motion of a simple pendulum using the Euler method.
 * It calculates the angular position and velocity over time, saving the results
 * to a CSV file and displaying them in the console.
 * 
 * The simulation uses the following parameters:
 * - Length (L): 1.0 meters
 * - Initial angle (theta): PI/6 radians (30 degrees)
 * - Initial angular velocity (omega): 0.0 rad/s
 * - Time step (dt): 0.01 seconds
 * - Total simulation time: 10.0 seconds
 * 
 * Output is saved to 'pendulum_output.csv' in the format: time,theta,omega
 */

#include <iostream>
#include <cmath>
#include <fstream>

const double PI = 3.14159265358979323846;
const double g = 9.81;  // gravity in m/s^2

int main() {
    double L = 1.0;             // Length of the pendulum (meters)
    double theta = PI / 6;      // Initial angle (30 degrees in radians)
    double omega = 0.0;         // Initial angular velocity
    double dt = 0.01;           // Time step (seconds)
    double t_max = 10.0;        // Total simulation time (seconds)

    std::ofstream file("pendulum_output.csv");
    file << "time,theta,omega\n";

    for (double t = 0; t <= t_max; t += dt) {
        // Euler method
        double alpha = - (g / L) * theta;  // Angular acceleration
        omega += alpha * dt;
        theta += omega * dt;

        file << t << "," << theta << "," << omega << "\n";

        // Optional: print to console
        std::cout << "t=" << t << "s, theta=" << theta << " rad, omega=" << omega << " rad/s\n";
    }

    file.close();
    std::cout << "\nSimulation complete! Data saved to pendulum_output.csv\n";

    return 0;
}