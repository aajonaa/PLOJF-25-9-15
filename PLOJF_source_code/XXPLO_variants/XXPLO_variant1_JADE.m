function [best_pos, Convergence_curve] = XXPLO_variant1_JADE(N, MaxFEs, lb, ub, dim, fobj)
% XXPLO_variant1_JADE: PLO-based variant with JADE/SHADE-style archive injection ONLY
% - Keep PLO-style generator: LS + GS + partner-guided hybrid direction H, two-candidate (cand1/cand2)
% - Introduce external archive A used to bias the exploration candidate cand2 (no EPM FDB survivor selection)
% - Selection: per-individual accept-if-better to keep PLO basis simple

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
phi_qKR = 0.25 + 0.55 * ((0 + ((1:N)/N)) .^ 0.5);
succ_win = 30; succ_hist = false(1, succ_win); succ_ptr = 1;
target_succ = 0.25; adapt_gain = 0.30; adapt_scale = 1.0;

% Partnered direction weights
c_mean=0.50; c_pbest=0.80; c_gbest=0.80; c_partner=0.60; early_ratio=0.20;

% Anti-stagnation controls (same style)
no_improv_cnt=0; no_improv_limit=8; imm_frac=0.15; div_thresh=0.01;

% JADE/SHADE-style archive (ONLY new addition in this variant)
A = zeros(0, dim); A_cap = N; pA = 0.25;  % cand2 archive injection rate

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

        % partner
        if FEs <= early_ratio * MaxFEs
            if i <= N/2, partner = i + floor(N/2); else, partner = i - floor(N/2); end
        else
            if i <= N/2, partner = randi([1, floor(N/2)]); else, partner = randi([floor(N/2)+1, N]); end
        end

        % hybrid direction
        c_partner_dyn = c_partner * (1 + 0.5*max(0,(div_thresh - norm_div)/div_thresh));
        H = c_mean*(X_mean - X(i,:)) + c_pbest*(Pbest(i,:) - X(i,:)) + c_gbest*(best_pos - X(i,:)) + c_partner_dyn*(Pbest(partner,:) - X(i,:));

        % two candidates
        ori = rand(1,dim);
        cand1 = X(i,:) + (adapt_scale*w1)*LS .* ori + H;
        % exploration candidate with archive injection (variant focus)
        if rand < pA && ~isempty(A)
            r2vec = A(randi(size(A,1)), :);
            cand2 = X(i,:) + (gs_amp/adapt_scale*w2)*GS .* ori + 0.5*(X(i,:) - r2vec);
        else
            cand2 = X(i,:) + (gs_amp/adapt_scale*w2)*GS .* ori;
        end
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
        if f1 <= f2, cand = cand1; f_cand = f1; else, cand = cand2; f_cand = f2; end

        % Accept-if-better selection per individual, push parent to archive when replaced
        if f_cand < Fitness(i)
            % push parent into archive
            A = [A; X(i,:)]; if size(A,1) > A_cap
                excess = size(A,1) - A_cap; drop = randperm(size(A,1), excess);
                keep = true(size(A,1),1); keep(drop) = false; A = A(keep,:);
            end
            newX(i,:) = cand; newF(i) = f_cand;
        else
            newX(i,:) = X(i,:); newF(i) = Fitness(i);
        end

        % update personal best
        if f_cand < PbestF(i), Pbest(i,:) = cand; PbestF(i) = f_cand; end
    end

    % finalize generation
    X = newX; Fitness = newF;
    [Fitness, sidx] = sort(Fitness); X = X(sidx,:);
    if Fitness(1) < bestFitness, bestFitness = Fitness(1); best_pos = X(1,:); end

    % success history
    improved = Fitness(1) < bestFitness - eps; %#ok<NASGU>
    succ_hist(succ_ptr) = any(newF < Fitness); succ_ptr = succ_ptr + 1; if succ_ptr > succ_win, succ_ptr = 1; end

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

