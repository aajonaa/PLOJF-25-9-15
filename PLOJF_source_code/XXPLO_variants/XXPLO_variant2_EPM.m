function [best_pos, Convergence_curve] = XXPLO_variant2_EPM(N, MaxFEs, lb, ub, dim, fobj)
% XXPLO_variant2_EPM: PLO-based variant with EPM Strategy 2 (FDB survivor selection) ONLY
% - Keep PLO-style generator: LS + GS + partner-guided hybrid direction H, two-candidate (cand1/cand2)
% - Introduce EPM Strategy 2: FDB-based survivor selection on [parents; offspring] (no external archive A)
% - Keep success-history adaptation on LS/GS, keep anti-stagnation micro-restart

%% Init
FEs = 0; it = 1; Convergence_curve = [];
X = initialization(N, dim, ub, lb);
V = ones(N, dim);
Fitness = inf(N,1);
for i = 1:N, Fitness(i) = fobj(X(i,:)); FEs = FEs + 1; end
[Fitness, idx] = sort(Fitness); X = X(idx,:);
best_pos = X(1,:); bestFitness = Fitness(1);
Convergence_curve(it) = bestFitness;

% Personal best archive (PLO-style)
Pbest = X; PbestF = Fitness;

% PLO parameters and adaptation
succ_win = 30; succ_hist = false(1, succ_win); succ_ptr = 1;
target_succ = 0.25; adapt_gain = 0.30; adapt_scale = 1.0;

c_mean=0.50; c_pbest=0.80; c_gbest=0.80; c_partner=0.60; early_ratio=0.20;
no_improv_cnt=0; no_improv_limit=8; imm_frac=0.15; div_thresh=0.01;

%% Main loop
while FEs < MaxFEs
    X_mean = mean(X,1);
    w1 = tansig((FEs/MaxFEs)^4); w2 = exp(-(2*FEs/MaxFEs)^3);

    curr_succ_rate = mean(succ_hist); if isnan(curr_succ_rate), curr_succ_rate=0; end
    adapt_scale = max(0.5, min(1.5, adapt_scale * (1 + adapt_gain*(target_succ - curr_succ_rate))));

    pop_std = std(X,0,1); rng_span = max(1e-12, mean(abs(ub - lb)));
    norm_div = mean(pop_std) / rng_span;

    low_div = max(0,(div_thresh - norm_div)/max(div_thresh,eps));
    low_succ= max(0,(target_succ - curr_succ_rate)/max(target_succ,eps));
    p_explore = min(0.7, 0.15 + 0.5*low_div + 0.25*low_succ);
    p_cauchy  = min(0.4, 0.05 + 0.2*low_div + 0.2*low_succ);
    gs_amp    = 1 + 0.8*low_div + 0.5*low_succ;

    newX = zeros(N, dim); newF = inf(N,1);
    for i = 1:N
        a = rand()/2 + 1; V(i,:) = exp((1-a)/100 * FEs); LS = V(i,:);
        GS = Levy(dim) .* (X_mean - X(i,:) + (lb + rand(1,dim).*(ub - lb))/2);

        if FEs <= early_ratio * MaxFEs
            if i <= N/2, partner = i + floor(N/2); else, partner = i - floor(N/2); end
        else
            if i <= N/2, partner = randi([1, floor(N/2)]); else, partner = randi([floor(N/2)+1, N]); end
        end

        c_partner_dyn = c_partner * (1 + 0.5*max(0,(div_thresh - norm_div)/div_thresh));
        H = c_mean*(X_mean - X(i,:)) + c_pbest*(Pbest(i,:) - X(i,:)) + c_gbest*(best_pos - X(i,:)) + c_partner_dyn*(Pbest(partner,:) - X(i,:));

        ori = rand(1,dim);
        cand1 = X(i,:) + (adapt_scale*w1)*LS .* ori + H;
        cand2 = X(i,:) + (gs_amp/adapt_scale*w2)*GS .* ori;
        if rand < p_cauchy
            cand2 = X(i,:) + (X(i,:) - best_pos) .* tan((ori - 0.5) * pi);
        end
        if rand < p_explore
            cand1 = cand2;
        end

        cand1 = BoundaryControl(cand1, lb, ub);
        cand2 = BoundaryControl(cand2, lb, ub);
        f1=inf; f2=inf;
        if FEs < MaxFEs, f1 = fobj(cand1); FEs = FEs + 1; end
        if FEs < MaxFEs, f2 = fobj(cand2); FEs = FEs + 1; end
        if f1 <= f2, newX(i,:) = cand1; newF(i) = f1; else, newX(i,:) = cand2; newF(i) = f2; end

        if newF(i) < PbestF(i), Pbest(i,:) = newX(i,:); PbestF(i) = newF(i); end
    end

    % EPM Strategy 2: FDB survivor selection on [X; newX]
    comb_X = [X; newX]; comb_F = [Fitness; newF];
    survivors = fdb_survivor_selection(comb_X, comb_F, N);
    X_next = comb_X(survivors, :); F_next = comb_F(survivors, :);
    [F_next, sidx] = sort(F_next); X_next = X_next(sidx, :);

    % Anti-stagnation immigration (same trigger)
    no_improv_cnt = no_improv_cnt + (~(F_next(1) < bestFitness));
    if no_improv_cnt >= no_improv_limit && norm_div < div_thresh
        m = max(1, round(imm_frac * N)); idx_tail = (N-m+1):N;
        X_next(idx_tail, :) = lb + (ub - lb) .* rand(m, dim);
        for ii = 1:m
            if FEs < MaxFEs, F_next(idx_tail(ii)) = fobj(X_next(idx_tail(ii), :)); FEs = FEs + 1; end
        end
        [F_next, sidx2] = sort(F_next); X_next = X_next(sidx2, :); no_improv_cnt = 0;
    end

    improved = F_next(1) < bestFitness - eps;
    succ_hist(succ_ptr) = improved; succ_ptr = succ_ptr + 1; if succ_ptr > succ_win, succ_ptr = 1; end
    if F_next(1) < bestFitness, bestFitness = F_next(1); best_pos = X_next(1,:); end
    X = X_next; Fitness = F_next;

    it = it + 1; Convergence_curve(it) = bestFitness;
end

end

%% Helper: Initialization
function X = initialization(N, dim, ub, lb)
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    X = rand(N, dim) .* (ub - lb) + lb;
end

%% Helper: Boundary control
function X = BoundaryControl(X, lb, ub)
    [N, dim] = size(X);
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    for i=1:N
        for j=1:dim
            k = rand < rand;
            if X(i,j) < lb(j)
                if k, X(i,j)=lb(j); else, X(i,j)=rand*(ub(j)-lb(j))+lb(j); end
            end
            if X(i,j) > ub(j)
                if k, X(i,j)=ub(j); else, X(i,j)=rand*(ub(j)-lb(j))+lb(j); end
            end
        end
    end
end

%% Helper: Levy flight step
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma; v = randn(1, d);
    step = u ./ abs(v).^(1 / beta); o = step;
end

%% Helper: FDB-based survivor selection
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

    sayac2 = sayac; Dpop = pdist2(Bpop, Bpop, 'euclidean'); [~, best_idx] = min(Fpop);
    d_best = Dpop(:, best_idx) + eps;
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

