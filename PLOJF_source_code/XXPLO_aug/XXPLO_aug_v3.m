function [best_pos, Convergence_curve] = XXPLO_PB_RTR(N, MaxFEs, lb, ub, dim, fobj)
% XXPLO_PB_RTR: PLO variant = current-to-pbest/1 guidance + restricted tournament replacement (niching)
% - One proposal per individual per iteration (budget parity)
% - Success-history adapts step size and p-best set size
% - Restricted Tournament Replacement (RTR) keeps multiple niches, avoiding takeover
% - Optional micro-restart via opposition around (best+mean) for aged individuals

%% Init
FEs = 0; it = 1; Convergence_curve = [];
X = initialization(N, dim, ub, lb);
Fitness = inf(N,1);
for i = 1:N, Fitness(i) = fobj(X(i,:)); FEs = FEs + 1; end
[Fitness, idx] = sort(Fitness); X = X(idx,:);
best_pos = X(1,:); bestFitness = Fitness(1);
Convergence_curve(it) = bestFitness;

% Success-history adaptation
succ_win = 20; succ_hist = zeros(1, succ_win); succ_ptr = 1;
target_succ = 0.25;

% p-best and step scale
p_init = 0.25; p_min = 0.10; p_max = 0.35;
F0 = 0.5; kF = 0.4; Fmin = 0.2; Fmax = 0.9;

    % External archive (JADE/SHADE-inspired) and stagnation counter
    A = zeros(0, dim);      % archive rows
    A_cap = N;              % archive capacity
    pA = 0.30;              % prob. to draw r2 from archive
    no_improv_cnt = 0;      % iterations since last global improvement
    no_improv_limit = 12;   % fallback trigger under low diversity

% RTR and aging
kRTR = min(5, N);           % RTR neighborhood size
age = zeros(N,1);           % iterations since last personal improvement
age_limit = 12;             % trigger micro-restart if exceeded under low diversity

%% Main loop
while FEs < MaxFEs
    X_mean = mean(X,1);

    succ_rate = mean(succ_hist); if isnan(succ_rate), succ_rate = 0; end
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
        % p-best index from topK
        pbest_idx = randi(topK);
        % choose r1 distinct and r2 from archive with prob pA (if available)
        r1 = i; while r1 == i, r1 = randi(N); end
        useA = rand < pA && ~isempty(A);
        if useA
            r2vec = A(randi(size(A,1)), :);
        else
            r2 = r1; while r2 == i || r2 == r1, r2 = randi(N); end
            r2vec = X(r2, :);
        end

        % Mutation: current-to-pbest/1 with archive option
        V = X(i,:) + F_like * (X(pbest_idx,:) - X(i,:)) + F_like * (X(r1,:) - r2vec);
        if rand < p_cauchy
            V = V + (V - best_pos) .* tan((rand(1, dim) - 0.5) * pi);
        end
        U = BoundaryControl(V, lb, ub);
        if FEs < MaxFEs, newX(i,:) = U; newF(i) = fobj(U); FEs = FEs + 1; end
    end

    % RTR replacement: offspring competes locally
    improved_global = false;
    replace_count = 0;   % for acceptance-rate based success history
    idx_order = randperm(N); % random visiting order to reduce bias
    for t = 1:N
        i = idx_order(t);
        % sample kRTR candidates from population and pick nearest to newX(i,:)
        cand_idx = randperm(N, kRTR);
        d = sum((X(cand_idx,:) - newX(i,:)).^2, 2); [~, mpos] = min(d);
        j = cand_idx(mpos);
        if newF(i) < Fitness(j)
            % before replacement, push parent to archive with some probability
            if rand < 0.7  % push-more bias keeps archive fresh but bounded later
                A = [A; X(j,:)];
                if size(A,1) > A_cap
                    excess = size(A,1) - A_cap;
                    drop_idx = randperm(size(A,1), excess);
                    keep = true(size(A,1),1); keep(drop_idx) = false;
                    A = A(keep, :);
                end
            end
            % replace and update age
            X(j,:) = newX(i,:); Fitness(j) = newF(i);
            age(j) = 0; % personal improvement
            replace_count = replace_count + 1;
            if Fitness(j) < bestFitness
                bestFitness = Fitness(j); best_pos = X(j,:); improved_global = true;
            end
        else
            age(j) = age(j) + 1;
        end
    end

    % Optional micro-restart for aged individuals when diversity is low
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

    % Success history based on acceptance rate (more informative)
    acc_rate = replace_count / N;
    succ_hist(succ_ptr) = acc_rate;  % record acceptance rate directly for adaptation
    succ_ptr = succ_ptr + 1; if succ_ptr > succ_win, succ_ptr = 1; end

    % Stagnation counter for fallback triggers
    if improved_global
        no_improv_cnt = 0;
    else
        no_improv_cnt = no_improv_cnt + 1;
    end

    % Optional: diversity-aware kRTR adaptation (mild)
    kRTR = max(3, min(12, round(min(5 + 20*max(0, 0.02 - norm_div), N))));

    % Periodic FDB fallback when stagnated under low diversity
    if no_improv_cnt >= no_improv_limit && norm_div < 0.01
        comb_X = [X; newX]; comb_F = [Fitness; newF];
        survivors = fdb_survivor_selection(comb_X, comb_F, N);
        X = comb_X(survivors, :); Fitness = comb_F(survivors, :);
        [Fitness, sidx] = sort(Fitness); X = X(sidx, :);
        best_pos = X(1,:); bestFitness = Fitness(1);
        no_improv_cnt = 0;    % reset after fallback
    end

    % Iteration bookkeeping
    it = it + 1; Convergence_curve(it) = bestFitness;

    % Optional: resort population by Fitness (not required by algorithm)
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
