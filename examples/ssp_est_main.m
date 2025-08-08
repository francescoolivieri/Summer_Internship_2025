clc
close all;

% --- 1. Create and Configure the Simulation ---
sim = bellhop.Simulation('Bottom Parameter Estimation Example');

% Set environment properties
sim.Environment.Depth = 100; % meters

% Set source properties
sim.Source.Frequency = 4000; % Hz
sim.Source.Depth = 10;       % meters

% --- 2. Define the Estimation Task ---
% We want to estimate two parameters on the Bottom object
estimation = bellhop.Estimation(...
    {'Bottom.SoundSpeed', 'Bottom.Density'}, ... % Parameters to estimate
    [1650, 1.6], ...                            % Prior Means
    diag([50^2, 0.2^2]) ...                     % Prior Covariance
);

% Link the estimation task to the simulation
sim.Estimation = estimation;

% --- 3. Setup the "True" Environment ---
% This step uses the prior distribution from the estimation object
% to sample the "true" parameter values that the simulation will use.
sim.Estimation.setup(sim); 

fprintf('--- True Parameters ---\n');
fprintf('True Bottom Sound Speed: %.2f m/s\n', sim.Environment.Bottom.SoundSpeed);
fprintf('True Bottom Density:     %.2f g/cm^3\n', sim.Environment.Bottom.Density);
fprintf('-----------------------\n\n');

% --- 4. Run the Simulation (Placeholder) ---
% This will eventually run Bellhop to generate data
sim.run();

% --- 5. Main Estimation Loop (To be refactored) ---
% The old loop will be adapted to the new object-oriented structure.
% We will need to update the UKF and data generation functions to work
% with the `sim` and `estimation` objects.

% disp('--- Starting Estimation Loop ---');
% for n = 1:20 % Let's say 20 measurements
%     fprintf('\n=== Iteration nr %d ===\n', n);
% 
%     % 1. Get Next Measurement Position (NBV Logic)
%     % next_position = pos_next_measurement(sim, estimation);
% 
%     % 2. Generate a Measurement
%     % measurement = generate_data(sim, next_position);
% 
%     % 3. Update the Estimate
%     % estimation.update(measurement);
% 
%     % 4. Display Results
%     % plot_result(estimation);
% end

disp('Example script finished.');
