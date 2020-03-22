clear all;

yalmip_path = '/home/jand271/rot/YALMIP-master';

load('hovland_et_al_model')
load('snapshots_of_hovland_el_al')

addpath('../../');
addpath(genpath(yalmip_path));

r = 5;
f = Bp * U(1,1);

[F,~,~] = dlqr(Ap,Bp,Q,R);

if r < 2
    [POD_Phi,~,~] = svd(X);
    V0 = POD_Phi(:,1:r);
else
    load(fullfile('output',['buithanh_rank_',num2str(r-1),'.mat']))
    V0 = zeros(size(X,1),r);
    V0(:,1:r-1) = V;
    clear V W compute_time_s;
    [POD_Phi,~,~] = svd(X);
    V0(:,r) = POD_Phi(:,1);
end

tic
V = buithanh(...
    eye(size(Ap,1)),...
    Ap, ...
    f, ...
    X,...
    r,...
    F,...
    1,...
    V0);
compute_time_s = toc;
W = V;
save(fullfile('output',['buithanh_rank_',num2str(r),'.mat']),...
     'W','V','compute_time_s');
