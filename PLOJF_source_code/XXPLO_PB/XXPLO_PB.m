function [best_pos, Convergence_curve] = XXPLO_PB(N, MaxFEs, lb, ub, dim, fobj)
% XXPLO_PB: PLO variant with current-to-pbest/1 guidance + tiny archive (JADE/SHADE-inspired)
% - One proposal per individual per iteration (budget parity)
% - Success-history adapts step size and p-best set size
% - External archive injects diversity in difference vectors
% - Survivor selection via FDB
%
% Signature: [best_pos, Convergence_curve] = XXPLO_PB(N, MaxFEs, lb, ub, dim, fobj)

%% Initialize
FEs = 0; it = 1; Convergence_curve = [];
X = initialization(N, dim, ub, lb);
Fitness = inf(N,1);
for i = 1:N, Fitness(i) = fobj(X(i,:)); FEs = FEs + 1; end
[Fitness, idx] = sort(Fitness); X = X(idx,:);
best_pos = X(1,:); bestFitness = Fitness(1);
Convergence_curve(it) = bestFitness;

% Success-history adaptation
succ_win = 20; succ_hist = false(1, succ_win); succ_ptr = 1;
target_succ = 0.25;                 % desired success rate

% p-best fraction and step scale (adaptive)
p_init = 0.25; p_min = 0.10; p_max = 0.35;
F0 = 0.5; kF = 0.4; Fmin = 0.2; Fmax = 0.9;

% External archive (stores parent solutions replaced by offspring)
A = zeros(0, dim);      % empty; rows will be appended
A_cap = N;              % maximum rows
pA = 0.30;              % probability to draw r2 from archive

% Rare micro-restart (opposition around mean+best) when stalled and collapsed
no_improv_cnt = 0; no_improv_limit = 12;

%% Main loop
while FEs < MaxFEs
    X_mean = mean(X, 1);
    % Success pressure
    succ_rate = mean(succ_hist); if isnan(succ_rate), succ_rate = 0; end
    press = max(0, 1 - succ_rate / max(target_succ, eps));

    % Adaptive p-best set size and step scale
    p_frac = clamp(p_init + 0.20 * (target_succ - succ_rate), p_min, p_max);
    F_like = clamp(F0 + kF * press, Fmin, Fmax);

    % Diversity metric (normalized std)
    pop_std = std(X, 0, 1);
    rng_span = max(1e-12, mean(abs(ub - lb)));
    norm_div = mean(pop_std) / rng_span;

    % Occasional Cauchy jump probability (higher when stalled)
    p_cauchy = min(0.30, 0.05 + 0.20 * press);

    % Proposal per individual
    newX = zeros(N, dim); newF = inf(N, 1);
    topK = max(1, ceil(p_frac * N));

    for i = 1:N
        % Choose p-best index from topK
        pbest_idx = randi(topK);
        
        % r1 from population distinct from i
        r1 = i; while r1 == i, r1 = randi(N); end
        % r2 from archive with prob pA (if available), else population
        useA = rand < pA && ~isempty(A);
        if useA
            r2vec = A(randi(size(A,1)), :);
        else
            r2 = r1; while r2 == i || r2 == r1, r2 = randi(N); end
            r2vec = X(r2, :);
        end

        % Mutation: current-to-pbest/1 with archive option
        V = X(i, :) + F_like * (X(pbest_idx, :) - X(i, :)) + F_like * (X(r1, :) - r2vec);

        % Rare Cauchy perturbation around best to escape local minima
        if rand < p_cauchy
            V = V + (V - best_pos) .* tan((rand(1, dim) - 0.5) * pi);
        end

        U = BoundaryControl(V, lb, ub);
        if FEs < MaxFEs, newX(i, :) = U; newF(i) = fobj(U); FEs = FEs + 1; end
    end

    % Survivor selection via FDB on parents + offspring
    comb_X = [X; newX]; comb_F = [Fitness; newF];
    survivors = fdb_survivor_selection(comb_X, comb_F, N);

    % Push parents replaced by offspring into archive
    A = archive_update(A, A_cap, survivors, X, newX);

    X_next = comb_X(survivors, :); F_next = comb_F(survivors, :);
    [F_next, sidx] = sort(F_next); X_next = X_next(sidx, :);

    % Success tracking
    improved = F_next(1) < bestFitness - eps;
    succ_hist(succ_ptr) = improved; succ_ptr = succ_ptr + 1; if succ_ptr > succ_win, succ_ptr = 1; end

    if improved
        bestFitness = F_next(1); best_pos = X_next(1, :);
        no_improv_cnt = 0;
    else
        no_improv_cnt = no_improv_cnt + 1;
    end

    % Optional micro-restart: opposition around best+mean for worst tail
    if no_improv_cnt >= no_improv_limit && norm_div < 0.01
        m = max(1, round(0.10 * N));
        idx_tail = (N - m + 1):N;             % worst m after sort
        X_opp = (best_pos + X_mean) - X_next(idx_tail, :);
        X_opp = BoundaryControl(X_opp, lb, ub);
        for k = 1:m
            if FEs < MaxFEs
                f_opp = fobj(X_opp(k, :)); FEs = FEs + 1;
                if f_opp < F_next(idx_tail(k))
                    F_next(idx_tail(k)) = f_opp;
                    X_next(idx_tail(k), :) = X_opp(k, :);
                end
            end
        end
        [F_next, sidx2] = sort(F_next); X_next = X_next(sidx2, :);
        no_improv_cnt = 0;
    end

    X = X_next; Fitness = F_next;
    it = it + 1; Convergence_curve(it) = bestFitness;
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

%% Helper: Archive update from FDB survivor list
function A = archive_update(A, A_cap, survivors, X, newX)
    % survivors indexes refer to [X; newX]. For any offspring chosen (idx>N), push its parent into archive.
    N = size(X, 1);
    sel_off = survivors(survivors > N) - N;  % offspring indices 1..N
    if ~isempty(sel_off)
        parents = X(sel_off, :);
        A = [A; parents];
        % Cap the archive by random removal if needed
        if size(A,1) > A_cap
            % remove excess rows randomly
            excess = size(A,1) - A_cap;
            drop_idx = randperm(size(A,1), excess);
            keep_mask = true(size(A,1),1); keep_mask(drop_idx) = false;
            A = A(keep_mask, :);
        end
    end
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

