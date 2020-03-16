
load('hovland_et_al_model.mat');

reduction_list = {
    'balanced_truncation', ...
    'carlberg', ...
    'pod', ...
    'buithanh'};

rank_list = [1,2,3,4,5];
max_buithanh_rank = 4;

F = dlqr(Ap,Bp,Q,R);

plotH2Norms(reduction_list, rank_list, max_buithanh_rank, Ap, Bp, F);
