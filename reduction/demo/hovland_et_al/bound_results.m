
clear all;

reduction_list = {
    'balanced_truncation', ...
    'carlberg', ...
    'pod', ...
    'buithanh'};

rank_list = [1,2,3,4,5];
max_buithanh_rank = 5;

[FOM, Z, U] = randomSmallSystem3();

plotControlErrorBounds(FOM, Z, U, reduction_list, rank_list, max_buithanh_rank);
