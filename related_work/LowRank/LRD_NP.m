function [A, E, N] = LRD_NP(D, P, alpha, beta, mu, tol, itemax)
    % - D image
    % - P defect prior
    % - G gradient information
    % - alpha regularization parameter for E (spaese term), default = 1/sqrt(max(N,M))
    % - beta regularization parameter for N (noise term), default = 2*alpha
    % - mu - the augmented lagrangian parameter, default = 10*alpha
    % - tol - reconstruction error tolerance, default = 1e-6
    % - max_iter - maximum number of iterations, default = 2000
    % ADM algorithm to solve the following matrix decomposotion problem:
    %       min  ||A||_* + alpha * ||W*E||_1 + beta/2 * ||N||_F^2
    %       s.t. D = A + E + N

    [m, n] = size(D);

    if nargin < 6
        tol = 1e-7;
    end

    if nargin < 7
        itemax = 2000;
    end

    % the observed matrix (image) D can contain nan
    unobserved = isnan(D);
    D(unobserved) = 0;

    % initialize
    norm_two = svds(D, 1);
    norm_inf = norm(D(:), inf) / alpha;
    dual_norm = max(norm_two, norm_inf);
    Y = D / dual_norm;

    A = zeros(m, n);
    E = zeros(m, n);
    N = zeros(m, n);

    if nargin < 5
        % mu = 1.e-4;
        mu = 1.25 / norm_two; % this one can be tuned
    end

    mu_max = mu * 1e6;
    rho = 1.05; % this one can be tuned
    d_norm = norm(D, 'fro');

    % defect prior P
    W = exp(-P);

    for ii = 1:itemax
        E = soft_thresh(D - A - N + Y / mu, (alpha / mu) .* W);

        N = (mu * (D - A - E) + Y) .* (1 / (beta + mu));

        A = sig_thresh(D - E - N + Y / mu, (1 / mu));

        Y = Y + mu * (D - A - E - N);

        mu = min(mu * rho, mu_max);

        % error = norm(D - A - E - N, inf);
        error = norm(D - A - E - N, 'fro') / d_norm;

        if (ii == 1) || (mod(ii, 100) == 0) || (error < tol)
            fprintf(1, 'iter: %05d\t err: %f\t rank(A): %d\t card(E): %d\n', ...
                ii, error, rank(A), nnz(E));
        end

        if error < tol
            disp('tolerance reached, stop early!');
            break;
        end

    end

end

function [Y] = soft_thresh(X, T)
    Y = sign(X) .* max(abs(X) - T, 0);
end

function [A] = sig_thresh(X, T)
    [U, S, V] = svd(X, 'econ');
    % A = U * soft_thresh(S, T) * V';
    S = diag(S);
    ind = find(S > T);
    S = diag(S(ind) - T);
    A = U(:, ind) * S * V(:, ind)';
end
