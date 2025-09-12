function idx_newpop = fdb_survivor_selection(Bpop, Fpop, N)
% FDB-based survivor selection (Strategy 2)
% Inputs:
%   Bpop: (M x dim) combined population
%   Fpop: (M x 1) fitness values (lower is better)
%   N:    target population size
% Output:
%   idx_newpop: (1 x N) indices of selected survivors (into Bpop)

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

