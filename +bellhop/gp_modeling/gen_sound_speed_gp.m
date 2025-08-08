function Csample = gen_sound_speed_gp(gridX,gridY,gridZ)
%SAMPLE_GP_SOUND_SPEED Sample a 3D sound speed field from a GP
%
% Csample = SAMPLE_GP_SOUND_SPEED(X,Y,Z) returns a synthetic sound speed
% field on a 3D grid defined by X,Y,Z (same size), using a GP with a
% mean function from CTD data and a squared exponential kernel.
%
% Optional hyperparameters:
%  'ell_h'     - horizontal length scale (m)
%  'ell_v'     - vertical length scale   (m)
%  'sigma_f'   - standard deviation of fluctuations (4 m/s)

% Parse optional parameters
ell_h = 300;
ell_v = 20;
sigma_f =1;

% Load measured profile
S  = load('CTD.mat');
fn = fieldnames(S);
raw = S.(fn{1});
z_raw = raw(:,1);
c_raw = raw(:,2);

% Remove duplicate depths
[z_tr,~,grp] = unique(z_raw,'stable');
c_tr = accumarray(grp, c_raw, [], @mean);

% Mean function mu(z)
mu_fun = @(z) interp1(z_tr, c_tr, z, 'linear', 'extrap');

% Grid coordinates (M×3)
Xstar = [gridX(:), gridY(:), gridZ(:)];
M = size(Xstar,1);

% Compute covariance matrix K (SE kernel)
K = zeros(M,M);
for i = 1:M
    dX = Xstar(i,:) - Xstar;
    dx2 = dX(:,1).^2 + dX(:,2).^2;
    dz2 = dX(:,3).^2;
    K(i,:) = exp(-0.5 * dx2 / ell_h^2 - 0.5 * dz2 / ell_v^2);
end
K = sigma_f^2 * K;

% Sample from the GP
mu = mu_fun(Xstar(:,3));
L = chol(K + 1e-6*eye(M), 'lower');  % Add jitter for stability
c_sample = mu + L * randn(M,1);

% Reshape to grid
Csample = reshape(c_sample, size(gridX));


end
