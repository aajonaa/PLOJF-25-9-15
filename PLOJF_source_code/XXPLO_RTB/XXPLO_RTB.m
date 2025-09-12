function [best_pos, Convergence_curve] = XXPLO_RTB(N, MaxFEs, lb, ub, dim, fobj)
% XXPLO_RTB: Simplified, adaptive rand-to-best variant for PLO family
% - One trial per individual; DE/rand-to-best/1 style step with adaptive F, CR
% - Success-rate driven exploration vs exploitation (few parameters)
% - Survivor selection via FDB to retain diversity

%% Init
FEs = 0; it = 1; Convergence_curve = [];
X = initialization(N, dim, ub, lb);
Fitness = inf(N,1);
for i = 1:N, Fitness(i) = fobj(X(i,:)); FEs = FEs + 1; end
[Fitness, idx] = sort(Fitness); X = X(idx,:);
best_pos = X(1,:); bestFitness = Fitness(1);
Convergence_curve(it) = bestFitness;

% Success-rate adaptation
succ_win = 20; succ_hist = false(1, succ_win); succ_ptr = 1;
target_succ = 0.25;

% Base parameters (adapted at runtime)
F_base = 0.5; kF = 0.4;   % F in [0.3, 0.9]
CR_base = 0.2; kCR = 0.6; % CR in [0.1, 0.9]

%% Main loop
while FEs < MaxFEs
    curr_succ = mean(succ_hist); if isnan(curr_succ), curr_succ = 0; end
    press = max(0, 1 - curr_succ / max(target_succ, eps));
    F  = min(0.9, max(0.3, F_base  + kF  * press));
    CR = min(0.9, max(0.1, CR_base + kCR * press));
    p_cauchy = min(0.3, 0.05 + 0.25 * press);

    newX = zeros(N, dim); newF = inf(N,1);

    for i = 1:N
        % pick r1, r2 distinct from i
        r1 = i; while r1 == i, r1 = randi(N); end
        r2 = r1; while r2 == i || r2 == r1, r2 = randi(N); end

        % Mutation: rand-to-best/1
        V = X(i,:) + F * (best_pos - X(i,:)) + F * (X(r1,:) - X(r2,:));

        % Crossover: binomial with jrand
        U = X(i,:);
        jrand = randi(dim);
        for j = 1:dim
            if rand < CR || j == jrand
                U(j) = V(j);
            end
        end

        % Occasional Cauchy perturbation around best (helps escape)
        if rand < p_cauchy
            U = U + (U - best_pos) .* tan((rand(1,dim) - 0.5) * pi);
        end

        U = BoundaryControl(U, lb, ub);
        if FEs < MaxFEs, newX(i,:) = U; newF(i) = fobj(U); FEs = FEs + 1; end
    end

    % FDB selection between parents and trials
    comb_X = [X; newX]; comb_F = [Fitness; newF];
    survivors = fdb_survivor_selection(comb_X, comb_F, N);
    X_next = comb_X(survivors,:); F_next = comb_F(survivors,:);
    [F_next, sidx] = sort(F_next); X_next = X_next(sidx,:);

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

function idx_newpop = fdb_survivor_selection(Bpop, Fpop, N)
    [total_pop_size, ~] = size(Bpop);
    [~, indexes] = sort(Fpop);
    forbidden_list = false(1, total_pop_size);
    idx_newpop = zeros(1, N);

    % first half by best fitness
    sayac = 0;
    for i = 1:total_pop_size
        if ~forbidden_list(indexes(i)) && sayac < floor(N/2)
            sayac = sayac + 1; idx_newpop(sayac) = indexes(i); forbidden_list(indexes(i)) = true;
        end
        if sayac >= floor(N/2), break; end
    end

    % second half by Fitness-Distance Balance
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

