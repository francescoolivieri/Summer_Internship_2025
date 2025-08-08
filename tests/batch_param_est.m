clc
close all;

% Number of runs
numRuns = 20;

% Preallocate results structure
results(numRuns) = struct('th', [], 'th_est', [], 'Sigma_est', []);

% Loop over multiple runs
for runIdx = 1:numRuns
    fprintf('\n========== Run %d ==========\n', runIdx);

    % Define parameters to estimate (you can randomize if needed)
    params.names = {'sound_speed_sediment', 'sound_speed_sediment'};
    params.mu = [1600; 1700];  % Prior means
    params.Sigma = diag([20 20].^2);  % Prior covariances

    % Setup simulation environment
    [data, s, ~] = setupUnderwaterSimulation(...
        'Parameters', params, ...
        'Units', 'km', ...
        'ExtraOutput', false);

    % Run simulation
    for n = 2:s.N
        fprintf('\n=== Iteration nr %d ===\n', n)

        data = pos_next_measurement(data, s);
        data = generate_data(data, s);
        data = ukf(data, s);
    end

    % Save results for this run
    results(runIdx).th        = data.th;
    results(runIdx).th_est    = data.th_est;
    results(runIdx).Sigma_est = data.Sigma_est;
end

% Save all results to file
save('batch_simulation_results.mat', 'results');
