/**
 * @file mass_spring_damper_sim.cpp
 * @brief Simulates a mass-spring-damper system using numerical integration
 * 
 * This program simulates the motion of a mass-spring-damper system by solving
 * the second-order differential equation: m*x'' + c*x' + k*x = 0
 * where:
 * - m is the mass of the object
 * - c is the damping coefficient
 * - k is the spring constant
 * - x is the displacement from equilibrium
 * 
 * The simulation uses a simple numerical integration method (Euler integration)
 * to compute the position and velocity of the mass over time. Results are
 * output both to the console and to a CSV file for further analysis.
 * 
 * Output file format:
 * - Column 1: Time (seconds)
 * - Column 2: Position (meters)
 * - Column 3: Velocity (meters/second)
 */

#include <iostream>
#include <fstream>

int main() {
    // Parameters
    const double m = 1.0;   // Mass (kg)
    const double k = 10.0;  // Spring constant (N/m)
    const double c = 0.5;   // Damping coefficient (kg/s)

    // Initial conditions
    double x = 1.0;         // Initial position (m)
    double v = 0.0;         // Initial velocity (m/s)
    double dt = 0.01;       // Time step (s)
    double t_max = 10.0;    // Total simulation time (s)

    std::ofstream file("spring_damper_output.csv");
    file << "time,x,v\n";

    for (double t = 0; t <= t_max; t += dt) {
        double a = (-c * v - k * x) / m;  // Acceleration
        v += a * dt;
        x += v * dt;

        file << t << "," << x << "," << v << "\n";
        std::cout << "t=" << t << "s, x=" << x << " m, v=" << v << " m/s\n";
    }

    file.close();
    std::cout << "\nSimulation complete! Data saved to spring_damper_output.csv\n";

    return 0;
}