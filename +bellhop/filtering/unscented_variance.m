function var_y = unscented_variance(f, p_vals, mu, Sigma)
    % Computes variance of y(p) for a vector of p values using the Unscented Transform.
    %
    % Inputs:
    %   f       - Function handle y = f(p, x)
    %   p_vals  - Matrix of p values (each row is a different p)
    %   mu      - Mean vector of x
    %   Sigma   - Covariance matrix of x
    %
    % Output:
    %   var_y   - Vector of variance values for each p

    % Define UT parameters
    n = length(mu); % Dimension of x
    alpha = 1e-3;   % Spread of sigma points
    kappa = 0;      % Secondary scaling parameter
    beta = 2;       % Optimal for Gaussian distributions
    
    lambda = alpha^2 * (n + kappa) - n;
    gamma = sqrt(n + lambda);

    % Compute sigma points
    [U, S, V] = svd(Sigma); % Ensure positive definiteness
    sqrtSigma = U * sqrt(S) * V'; % Compute square root of Sigma

    sigmaPoints = [mu, mu + gamma * sqrtSigma, mu - gamma * sqrtSigma];

    % Compute weights
    Wm = [lambda / (n + lambda), repmat(1 / (2 * (n + lambda)), 1, 2 * n)];
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta); % Adjust first weight

    % Number of p values
    num_p = size(p_vals, 1); % Number of rows in p_vals

    % Initialize variance output
    var_y = zeros(num_p, 1);

    % Loop over each p value
    for i = 1:num_p
        p = p_vals(i, :); % Extract current p vector

        % Transform sigma points through f(p, x)
        yPoints = arrayfun(@(j) f(p, sigmaPoints(:, j)), 1:size(sigmaPoints, 2));

        % Compute mean of transformed points
        mu_y = sum(Wm .* yPoints);

        % Compute variance of y for this p
        var_y(i) = sum(Wc .* (yPoints - mu_y).^2);
    end
end
