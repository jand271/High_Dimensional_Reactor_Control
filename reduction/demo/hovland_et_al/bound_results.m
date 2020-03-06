
clear all;

reduction_list = {
    'balanced_truncation', ...
    'carlberg', ...
    'modal', ...
    'pod', ...
    'buithanh', ...
    'grad_descent'};

rank_list = [1,2,3,4,5,6];
max_buithanh_rank = 3;

[FOM, Z, U] = randomSmallSystem();

plotControlErrorBounds(FOM, Z, U, reduction_list, rank_list, max_buithanh_rank);
