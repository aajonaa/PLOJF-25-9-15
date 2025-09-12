function [best_pos, Convergence_curve] = PLOJF(N, MaxFEs, lb, ub, dim, fobj)
% Polar Lights Optimization with Event-Driven JADE Integration and adaptive population management through Fitness-Distance Balanced Selection

%% JADE
% J. Zhang and A. C. Sanderson, "JADE: Adaptive Differential Evolution With Optional External Archive," in IEEE Transactions on Evolutionary Computation, vol. 13, no. 5, pp. 945-958, Oct. 2009
%% EPM
% Furkan Üstünsoy, Hamdi Tolga Kahraman, H. Hüseyin Sayan, Yusuf Sönmez, Evolutionary population management for the design of metaheuristic search algorithms: Three improved algorithms, real-time charge scheduling problems, optimal solutions and stability analysis, Knowledge-Based Systems, Volume 328, 2025

%% Init
FEs = 0; it = 1; Convergence_curve = [];
X = initialization(N, dim, ub, lb);
V = ones(N, dim);
Fitness = inf(N,1);
for i = 1:N, Fitness(i) = fobj(X(i,:)); FEs = FEs + 1; end
[Fitness, idx] = sort(Fitness); X = X(idx,:);
best_pos = X(1,:); bestFitness = Fitness(1);
Convergence_curve(it) = bestFitness;

% Personal best archive
Pbest = X; PbestF = Fitness;

% PLO adaptation
succ_win = 30; succ_hist = false(1, succ_win); succ_ptr = 1;
target_succ = 0.25; adapt_gain = 0.30; adapt_scale = 1.0;

% Partner schedule/weights
c_mean=0.50; c_pbest=0.80; c_gbest=0.80; c_partner=0.60; early_ratio=0.20;

% Anti-stagnation
no_improv_cnt=0; div_thresh=0.01;

% Adaptive FDB trigger
base_no_improv = 12; alpha_lim = 0.7; base_div_thr = div_thresh; beta_div = 0.5;
fdb_on_t = 0.40; pmax_fdb = 0.6; cooldown_iters = 3; cooldown = 0;

% JADE (event-driven, orthogonal, bottom-only)
A = zeros(0, dim); A_cap = N;             % archive (bounded)
K_after_fdb = 2;                           % window length (generations)
boost_left = 0;                            % remaining boosted generations after FDB
q_bottom = 0.30;                           % fraction of bottom individuals to apply JADE
pA = 0.10;                                 % small usage prob
eta = 0.15;                                % small injection magnitude

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

    % Determine bottom-q mask based on current Fitness
    [~, ord] = sort(Fitness, 'ascend');
    worst_count = max(1, round(q_bottom * N));
    worst_idx = ord(end-worst_count+1:end);
    bottom_mask = false(N,1); bottom_mask(worst_idx) = true;

    newX = zeros(N, dim); newF = inf(N,1);
    for i = 1:N
        % Generator: LS + GS + partner-guided H, two-candidate
        a = rand()/2 + 1; V(i,:) = exp((1-a)/100 * FEs); LS = V(i,:);
        GS = Levy(dim) .* (X_mean - X(i,:) + (lb + rand(1,dim).*(ub - lb))/2);

        % partner
        if FEs <= early_ratio * MaxFEs
            if i <= N/2, partner = i + floor(N/2); else, partner = i - floor(N/2); end
        else
            if i <= N/2, partner = randi([1, floor(N/2)]); else, partner = randi([floor(N/2)+1, N]); end
        end

        % hybrid direction H
        c_partner_dyn = c_partner * (1 + 0.5*max(0,(div_thresh - norm_div)/div_thresh));
        H = c_mean*(X_mean - X(i,:)) + c_pbest*(Pbest(i,:) - X(i,:)) + c_gbest*(best_pos - X(i,:)) + c_partner_dyn*(Pbest(partner,:) - X(i,:));

        ori = rand(1,dim);
        cand1 = X(i,:) + (adapt_scale*w1)*LS .* ori + H;                 % exploit
        cand2 = X(i,:) + (gs_amp/adapt_scale*w2)*GS .* ori;               % explore

        % Event-driven JADE injection: only if boost_left>0 and bottom individual
        if boost_left > 0 && bottom_mask(i) && ~isempty(A) && rand < pA
            r2vec = A(randi(size(A,1)), :);
            delta = eta * (X(i,:) - r2vec);
            % orthogonalize delta to H to avoid fighting exploitation direction
            denom = sum(H.^2) + eps;
            proj = (sum(delta.*H)/denom) * H;
            delta = delta - proj;
            cand2 = cand2 + delta;
        end

        % Cauchy jump for cand2
        if rand < p_cauchy
            cand2 = X(i,:) + (X(i,:) - best_pos) .* tan((ori - 0.5) * pi);
        end
        if rand < p_explore
            cand1 = cand2;
        end

        % evaluate and select
        cand1 = BoundaryControl(cand1, lb, ub);
        cand2 = BoundaryControl(cand2, lb, ub);
        f1=inf; f2=inf;
        if FEs < MaxFEs, f1 = fobj(cand1); FEs = FEs + 1; end
        if FEs < MaxFEs, f2 = fobj(cand2); FEs = FEs + 1; end
        if f1 <= f2, cand = cand1; f_cand = f1; else, cand = cand2; f_cand = f2; end

        % Accept-if-better; if replaced, push parent to archive (bounded)
        if f_cand < Fitness(i)
            A = [A; X(i,:)];
            if size(A,1) > A_cap
                excess = size(A,1) - A_cap; drop = randperm(size(A,1), excess);
                keep = true(size(A,1),1); keep(drop) = false; A = A(keep,:);
            end
            newX(i,:) = cand; newF(i) = f_cand;
        else
            newX(i,:) = X(i,:); newF(i) = Fitness(i);
        end

        if f_cand < PbestF(i), Pbest(i,:) = cand; PbestF(i) = f_cand; end
    end

    % success history
    succ_hist(succ_ptr) = any(newF < Fitness); succ_ptr = succ_ptr + 1; if succ_ptr > succ_win, succ_ptr = 1; end

    % Adaptive EPM: conditional FDB on [X; newX]
    if ~(min(newF) < bestFitness - eps), no_improv_cnt = no_improv_cnt + 1; else, no_improv_cnt = 0; end

    t = FEs / max(1, MaxFEs);
    if t >= fdb_on_t && cooldown == 0
        stall_lim = round(base_no_improv * max(0.3, 1 - alpha_lim * t));
        div_thr   = base_div_thr * (1 + beta_div * t);
        near_stall = no_improv_cnt >= max(1, stall_lim-1);
        low_div    = norm_div < div_thr;
        p_trig     = min(pmax_fdb, 0.2 + 0.4*t + 0.2*double(near_stall) + 0.2*double(low_div));
        if (no_improv_cnt >= stall_lim && low_div) || (near_stall && low_div && rand < p_trig)
            comb_X = [X; newX]; comb_F = [Fitness; newF];
            survivors = fdb_survivor_selection(comb_X, comb_F, N);
            X_sel = comb_X(survivors, :); F_sel = comb_F(survivors, :);
            [F_sel, sidx] = sort(F_sel); X_sel = X_sel(sidx, :);
            newX = X_sel; newF = F_sel;   % finalize with FDB-selected
            no_improv_cnt = 0; cooldown = cooldown_iters;  % start cooldown
            boost_left = K_after_fdb;      % enable JADE window for next K generations
        end
    end
    if cooldown > 0, cooldown = cooldown - 1; end
    if boost_left > 0, boost_left = boost_left - 1; end

    % finalize generation
    X = newX; Fitness = newF;
    [Fitness, sidx2] = sort(Fitness); X = X(sidx2,:);
    if Fitness(1) < bestFitness, bestFitness = Fitness(1); best_pos = X(1,:); end

    it = it + 1; Convergence_curve(it) = bestFitness;
end

end

function X = initialization(N, dim, ub, lb)
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    X = rand(N, dim) .* (ub - lb) + lb;
end

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
            sayac = sayac + 1; idx_newpop(sayac) = indexes(i); forbidden_list(indexes(i)) = true;
        end
        if sayac >= floor(N/2), break; end
    end

    % Step 2: FDB-based selection for the second half
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

