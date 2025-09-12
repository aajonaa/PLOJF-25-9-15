function [best_pos, Convergence_curve] = XXPLO_S(N, MaxFEs, lb, ub, dim, fobj)
% XXPLO_S: Simplified PLO variant with adaptive exploration via success rate
% - Minimal parameters, strong adaptivity
% - Focuses on parent selection and solution generation
% - One candidate per individual per iteration; FDB selection for diversity
%
% Signature: [best_pos, Convergence_curve] = XXPLO_S(N, MaxFEs, lb, ub, dim, fobj)

%% Init
FEs = 0; it = 1; Convergence_curve = [];
X = initialization(N, dim, ub, lb);
Fitness = inf(N,1);
for i = 1:N, Fitness(i) = fobj(X(i,:)); FEs = FEs + 1; end
[Fitness, idx] = sort(Fitness); X = X(idx,:);
best_pos = X(1,:); bestFitness = Fitness(1);
Convergence_curve(it) = bestFitness;

% Success-rate adaptation (single driver for exploration)
succ_win = 20;                    % short window
succ_hist = false(1, succ_win);
succ_ptr = 1;
target_succ = 0.25;               % desired success

%% Main loop
while FEs < MaxFEs
    X_mean = mean(X,1);
    w1 = tansig((FEs/MaxFEs)^4);      % exploitation weight
    w2 = exp(-(2*FEs/MaxFEs)^3);      % exploration weight

    % Exploration pressure from success ratio
    curr_succ_rate = mean(succ_hist); if isnan(curr_succ_rate), curr_succ_rate = 0; end
    p_explore = min(0.9, max(0.0, 1 - curr_succ_rate / max(target_succ, eps)));
    p_cauchy  = min(0.35, 0.05 + 0.25 * p_explore);

    % One candidate per individual
    newX = zeros(N, dim); newF = inf(N,1);
    halfN = floor(N/2);
    for i = 1:N
        a = rand()/2 + 1;                        % local scale factor
        LS = exp((1 - a)/100 * FEs);             % local search scale (scalar -> broadcast)
        GS = Levy(dim) .* (X_mean - X(i,:) + (lb + rand(1,dim).*(ub-lb))/2);

        % Adaptive partner selection (explore from worse half; exploit from better half)
        if rand < p_explore
            partner = randi([halfN+1, N]);       % explore partner (worse half)
        else
            partner = randi([1, max(1,halfN)]);  % exploit partner (better half)
        end

        % Candidate generation (simple, adaptive blend)
        ori = rand(1,dim);
        cand = X(i,:) ...
             + (1 - p_explore) * (w1 .* LS) .* ori ...          % exploit toward mean/best via LS
             + p_explore       * (w2 .* GS) .* ori ...          % explore via GS
             + p_explore * 0.1 * (X(partner,:) - X(i,:));       % partner-driven drift

        % Occasional Cauchy jump around best (more frequent under high p_explore)
        if rand < p_cauchy
            cand = X(i,:) + (X(i,:) - best_pos) .* tan((ori - 0.5) * pi);
        end

        cand = BoundaryControl(cand, lb, ub);
        if FEs < MaxFEs, newX(i,:) = cand; newF(i) = fobj(cand); FEs = FEs + 1; end
    end

    % FDB survivor selection for diversity (parent+offspring)
    comb_X = [X; newX]; comb_F = [Fitness; newF];
    survivors = fdb_survivor_selection(comb_X, comb_F, N);
    X_next = comb_X(survivors,:); F_next = comb_F(survivors,:);
    [F_next, sidx] = sort(F_next); X_next = X_next(sidx,:);

    % Update success measures and best
    improved = F_next(1) < bestFitness - eps;
    succ_hist(succ_ptr) = improved; succ_ptr = succ_ptr + 1; if succ_ptr > succ_win, succ_ptr = 1; end
    if improved, bestFitness = F_next(1); best_pos = X_next(1,:); end

    X = X_next; Fitness = F_next;
    it = it + 1; Convergence_curve(it) = bestFitness;
end

end

%% Helpers
function X = initialization(N, dim, ub, lb)
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    X = rand(N, dim) .* (ub - lb) + lb;
end

function X = BoundaryControl(X, lb, ub)
    [N, dim] = size(X);
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    for i = 1:N
        for j = 1:dim
            k = rand < rand;
            if X(i,j) < lb(j)
                if k, X(i,j) = lb(j); else, X(i,j) = rand * (ub(j)-lb(j)) + lb(j); end
            end
            if X(i,j) > ub(j)
                if k, X(i,j) = ub(j); else, X(i,j) = rand * (ub(j)-lb(j)) + lb(j); end
            end
        end
    end
end

function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma; v = randn(1, d);
    o = u ./ abs(v).^(1 / beta);
end

function idx_newpop = fdb_survivor_selection(Bpop, Fpop, N)
    [total_pop_size, ~] = size(Bpop);
    [~, indexes] = sort(Fpop);
    forbidden_list = false(1, total_pop_size);
    idx_newpop = zeros(1, N);
    sayac = 0;
    for i = 1:total_pop_size
        if ~forbidden_list(indexes(i)) && sayac < floor(N/2)
            sayac = sayac + 1; idx_newpop(sayac) = indexes(i); forbidden_list(indexes(i)) = true;
        end
        if sayac >= floor(N/2), break; end
    end
    sayac2 = sayac;
    Dpop = pdist2(Bpop, Bpop, 'euclidean');
    [~, best_idx] = min(Fpop); d_best = Dpop(:, best_idx) + eps;
    f_norm = (Fpop - min(Fpop)) / (max(Fpop) - min(Fpop) + eps);
    d_norm = d_best / (max(d_best) + eps);
    FDB = f_norm + d_norm; [~, order] = sort(FDB);
    for i = 1:total_pop_size
        idx = order(i);
        if ~forbidden_list(idx)
            sayac2 = sayac2 + 1; idx_newpop(sayac2) = idx; forbidden_list(idx) = true;
            if sayac2 >= N, break; end
        end
    end
end

