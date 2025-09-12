function [best_pos, Convergence_curve] = XXPLO_PB_RTR_A(N, MaxFEs, lb, ub, dim, fobj)
% XXPLO_PB_RTR_A: XXPLO_PB_RTR + Archive A only (no acc-rate, no adaptive kRTR, no FDB fallback)
% - Proposal: current-to-pbest/1, r2 from archive with prob pA
% - Replacement: Restricted Tournament Replacement (RTR)
% - Success history: boolean global-improvement
% - kRTR: fixed

%% Init
FEs = 0; it = 1; Convergence_curve = [];
X = initialization(N, dim, ub, lb);
Fitness = inf(N,1);
for i = 1:N, Fitness(i) = fobj(X(i,:)); FEs = FEs + 1; end
[Fitness, idx] = sort(Fitness); X = X(idx,:);
best_pos = X(1,:); bestFitness = Fitness(1);
Convergence_curve(it) = bestFitness;

% Success-history adaptation
succ_win = 20; succ_hist = false(1, succ_win); succ_ptr = 1;
target_succ = 0.25;

% p-best and step scale
p_init = 0.25; p_min = 0.10; p_max = 0.35;
F0 = 0.5; kF = 0.4; Fmin = 0.2; Fmax = 0.9;

% Archive A
A = zeros(0, dim); A_cap = N; pA = 0.30;

% RTR and aging
kRTR = min(5, N);
age = zeros(N,1);
age_limit = 12;

%% Main loop
while FEs < MaxFEs
    X_mean = mean(X,1);

    succ_rate = mean(double(succ_hist)); if isnan(succ_rate), succ_rate = 0; end
    press = max(0, 1 - succ_rate / max(target_succ, eps));

    p_frac = clamp(p_init + 0.20 * (target_succ - succ_rate), p_min, p_max);
    F_like = clamp(F0 + kF * press, Fmin, Fmax);
    p_cauchy = min(0.25, 0.05 + 0.20 * press);

    % Diversity (normalized std)
    pop_std = std(X, 0, 1);
    rng_span = max(1e-12, mean(abs(ub - lb)));
    norm_div = mean(pop_std) / rng_span;

    topK = max(1, ceil(p_frac * N));

    % Prepare trial set
    newX = zeros(N, dim); newF = inf(N,1);
    for i = 1:N
        pbest_idx = randi(topK);
        r1 = i; while r1 == i, r1 = randi(N); end
        useA = rand < pA && ~isempty(A);
        if useA
            r2vec = A(randi(size(A,1)), :);
        else
            r2 = r1; while r2 == i || r2 == r1, r2 = randi(N); end
            r2vec = X(r2, :);
        end
        V = X(i,:) + F_like * (X(pbest_idx,:) - X(i,:)) + F_like * (X(r1,:) - r2vec);
        if rand < p_cauchy
            V = V + (V - best_pos) .* tan((rand(1, dim) - 0.5) * pi);
        end
        U = BoundaryControl(V, lb, ub);
        if FEs < MaxFEs, newX(i,:) = U; newF(i) = fobj(U); FEs = FEs + 1; end
    end

    % RTR replacement
    improved_global = false;
    idx_order = randperm(N);
    for t = 1:N
        i = idx_order(t);
        cand_idx = randperm(N, kRTR);
        d = sum((X(cand_idx,:) - newX(i,:)).^2, 2); [~, mpos] = min(d);
        j = cand_idx(mpos);
        if newF(i) < Fitness(j)
            % push parent to archive (bounded)
            if rand < 0.7
                A = [A; X(j,:)];
                if size(A,1) > A_cap
                    excess = size(A,1) - A_cap; drop_idx = randperm(size(A,1), excess);
                    keep = true(size(A,1),1); keep(drop_idx) = false; A = A(keep, :);
                end
            end
            X(j,:) = newX(i,:); Fitness(j) = newF(i); age(j) = 0;
            if Fitness(j) < bestFitness
                bestFitness = Fitness(j); best_pos = X(j,:); improved_global = true;
            end
        else
            age(j) = age(j) + 1;
        end
    end

    % micro-restart for aged when diversity is low and no global improve
    if ~improved_global && norm_div < 0.01
        aged = find(age > age_limit);
        if ~isempty(aged)
            m = max(1, round(0.10 * numel(aged)));
            aged = aged(randperm(numel(aged), m));
            X_opp = (best_pos + X_mean) - X(aged, :);
            X_opp = BoundaryControl(X_opp, lb, ub);
            for kk = 1:numel(aged)
                if FEs < MaxFEs
                    f_opp = fobj(X_opp(kk, :)); FEs = FEs + 1;
                    if f_opp < Fitness(aged(kk))
                        Fitness(aged(kk)) = f_opp; X(aged(kk), :) = X_opp(kk, :);
                        age(aged(kk)) = 0;
                        if f_opp < bestFitness
                            bestFitness = f_opp; best_pos = X(aged(kk), :);
                        end
                    end
                end
            end
        end
    end

    % success-history update (boolean global improvement)
    succ_hist(succ_ptr) = improved_global; succ_ptr = succ_ptr + 1; if succ_ptr > succ_win, succ_ptr = 1; end

    it = it + 1; Convergence_curve(it) = bestFitness;
    [Fitness, idx] = sort(Fitness); X = X(idx, :); age = age(idx);
end

end

%% Utility: clamp
function y = clamp(x, lo, hi)
    y = min(hi, max(lo, x));
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

