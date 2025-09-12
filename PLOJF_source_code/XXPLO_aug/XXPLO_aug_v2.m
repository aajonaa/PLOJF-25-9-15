function [best_pos, Convergence_curve] = XXPLO(N, MaxFEs, lb, ub, dim, fobj)
% XXPLO: High-performance PLO variant combining three ideas
% 1) Parent-child separation (EPM Strategy 1)
% 2) FDB-based survivor selection (EPM Strategy 2)
% 3) Simple success-history adaptation on LS/GS mixing weights
%
% Signature matches other algorithms: [best_pos, Convergence_curve] = XXPLO(N, MaxFEs, lb, ub, dim, fobj)
%
% Notes:
% - BoundaryControl retains feasible solutions (reflect-or-regen style).
% - Success rate is tracked over a sliding window and adjusts an adaptive factor that
%   scales LS/GS mix to better balance exploration vs exploitation.

%% Initialization
FEs = 0;
it = 1;
Convergence_curve = [];

% Initialize population
X = initialization(N, dim, ub, lb);
V = ones(N, dim);
Fitness = inf * ones(N, 1);
for i = 1:N
    Fitness(i) = fobj(X(i, :));
    FEs = FEs + 1;
end
[Fitness, idx] = sort(Fitness);
X = X(idx, :);
best_pos = X(1, :);
bestFitness = Fitness(1);
Convergence_curve(it) = bestFitness;

% Initialize personal best archive (EPM-inspired)
Pbest = X;            % per-individual best position
PbestF = Fitness;     % per-individual best fitness

% PLO parameters
phi_qKR = 0.25 + 0.55 * ((0 + ((1:N)/N)) .^ 0.5);

% Success-history adaptation (gentle)
succ_win = 30;            % sliding window length (iterations)
succ_hist = false(1, succ_win);
succ_ptr = 1;
target_succ = 0.25;       % target success rate
adapt_gain = 0.30;        % adaptation gain (0..1)
adapt_scale = 1.0;        % multiplicative factor for LS/GS

% Partnered global direction coefficients (EPM-inspired)
c_mean    = 0.50;  % towards population mean
c_pbest   = 0.80;  % towards personal best
c_gbest   = 0.80;  % towards global best
c_partner = 0.60;  % towards partner

% Early vs late partner schedule
early_ratio = 0.20;    % first 20% of FEs use deterministic cross-half pairing


%% Main loop
while FEs < MaxFEs
    % Compute global stats and base weights
    X_mean = mean(X, 1);
    w1 = tansig((FEs/MaxFEs)^4);
    w2 = exp(-(2*FEs/MaxFEs)^3);

    % Apply success-history adaptation to LS/GS blend
    curr_succ_rate = mean(succ_hist);
    if isnan(curr_succ_rate)
        curr_succ_rate = 0;
    end
    adapt_scale = max(0.5, min(1.5, adapt_scale * (1 + adapt_gain * (target_succ - curr_succ_rate))));

    % Candidate generation: one candidate per individual
    newX = zeros(N, dim);
    newF = inf(N, 1);

    for i = 1:N
        a = rand()/2 + 1;
        V(i, :) = exp((1 - a)/100 * FEs);  % local search scale
        LS = V(i, :);
        GS = Levy(dim) .* (X_mean - X(i, :) + (lb + rand(1, dim) .* (ub - lb))/2);

        % Partner index (EPM-inspired: early deterministic cross-half pairing; late random within half)
        if FEs <= early_ratio * MaxFEs
            if i <= N/2, partner = i + floor(N/2); else, partner = i - floor(N/2); end
        else
            if i <= N/2, partner = randi([1, floor(N/2)]); else, partner = randi([floor(N/2)+1, N]); end
        end

        % Hybrid direction combining mean, personal best, global best, and partner
        H = c_mean*(X_mean - X(i,:)) + c_pbest*(Pbest(i,:) - X(i,:)) + c_gbest*(best_pos - X(i,:)) + c_partner*(Pbest(partner,:) - X(i,:));

        % Two candidate proposals: exploitation vs exploration
        ori_value = rand(1, dim);
        cand1 = X(i, :) + (adapt_scale*w1)*LS .* ori_value + H;           % exploit-biased
        cand2 = X(i, :) + (1/adapt_scale*w2)*GS .* ori_value;             % explore-biased

        % Racy Cauchy jump (rare)
        if rand < 0.05
            cand2 = X(i, :) + (X(i, :) - best_pos) .* tan((ori_value - 0.5) * pi);
        end

        % Evaluate both, choose the better (1 extra eval amortized; still budget-neutral vs prior)
        cand1 = BoundaryControl(cand1, lb, ub);
        cand2 = BoundaryControl(cand2, lb, ub);
        f1 = inf; f2 = inf;
        if FEs < MaxFEs, f1 = fobj(cand1); FEs = FEs + 1; end
        if FEs < MaxFEs, f2 = fobj(cand2); FEs = FEs + 1; end
        if f1 <= f2, cand = cand1; f_cand = f1; else, cand = cand2; f_cand = f2; end

        % Update personal best (archive)
        if f_cand < PbestF(i)
            Pbest(i,:) = cand; PbestF(i) = f_cand;
        end

        newX(i, :) = cand; newF(i) = f_cand;
    end

    % FDB selection using old (X,Fitness) and new (newX,newF) — no extra evals
    comb_X = [X; newX];
    comb_F = [Fitness; newF];
    survivors = fdb_survivor_selection(comb_X, comb_F, N);
    X_next = comb_X(survivors, :);
    F_next = comb_F(survivors, :);
    [F_next, sidx] = sort(F_next);
    X_next = X_next(sidx, :);

    % Note: no elite-only extra passes; keep selection purely from (X,newX) via FDB

    % Track success for adaptation (improvements over previous best)
    improved = F_next(1) < bestFitness - eps;
    succ_hist(succ_ptr) = improved;
    succ_ptr = succ_ptr + 1; if succ_ptr > succ_win, succ_ptr = 1; end

    if F_next(1) < bestFitness
        bestFitness = F_next(1);
        best_pos = X_next(1, :);
    end

    X = X_next; Fitness = F_next;

    it = it + 1;
    Convergence_curve(it) = bestFitness;

    % Optional progress print (comment out if noisy)
    % fprintf('XXPLO Iter %d, FEs %d, Best = %.4e, succ=%.2f, scale=%.2f\n', it-1, FEs, bestFitness, mean(succ_hist), adapt_scale);
end

end

%% Helper: Initialization
function X = initialization(N, dim, ub, lb)
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    X = rand(N, dim) .* (ub - lb) + lb;
end

%% Helper: Boundary control (reflect-or-regen style)
function X = BoundaryControl(X, lb, ub)
    [N, dim] = size(X);
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    for i = 1:N
        for j = 1:dim
            k = rand < rand;
            if X(i, j) < lb(j)
                if k, X(i, j) = lb(j); else, X(i, j) = rand * (ub(j) - lb(j)) + lb(j); end
            end
            if X(i, j) > ub(j)
                if k, X(i, j) = ub(j); else, X(i, j) = rand * (ub(j) - lb(j)) + lb(j); end
            end
        end
    end
end

%% Helper: Levy flight step
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    o = step;
end

%% Helper: FDB-based survivor selection (Strategy 2)
function idx_newpop = fdb_survivor_selection(Bpop, Fpop, N)
    [total_pop_size, ~] = size(Bpop);
    [~, indexes] = sort(Fpop);
    forbidden_list = false(1, total_pop_size);
    idx_newpop = zeros(1, N);

    % Step 1: Fitness-based selection for the first half
    sayac = 0;
    for i = 1:total_pop_size
        if ~forbidden_list(indexes(i)) && sayac < floor(N/2)
            sayac = sayac + 1;
            idx_newpop(sayac) = indexes(i);
            forbidden_list(indexes(i)) = true;
        end
        if sayac >= floor(N/2), break; end
    end

    % Step 2: FDB-based selection for the second half
    sayac2 = sayac;
    Dpop = pdist2(Bpop, Bpop, 'euclidean');
    [~, best_idx] = min(Fpop);
    f_best = Fpop(best_idx);
    d_best = Dpop(:, best_idx) + eps;
    % Normalize fitness and distance
    f_norm = (Fpop - min(Fpop)) / (max(Fpop) - min(Fpop) + eps);
    d_norm = d_best / (max(d_best) + eps);
    % Fitness-Distance Balance score (lower is better)
    FDB = f_norm + d_norm;
    [~, order] = sort(FDB);
    for i = 1:total_pop_size
        idx = order(i);
        if ~forbidden_list(idx)
            sayac2 = sayac2 + 1;
            idx_newpop(sayac2) = idx;
            forbidden_list(idx) = true;
            if sayac2 >= N, break; end
        end
    end
end

