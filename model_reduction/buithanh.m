 function [V] = buithanh(M, K, f, U, m, C, beta)
%{
 Complete model reduction strategy by Bui-Thanh with YALMIP
    :param M: Mass Matrix
    :param K: Stiffness Matrix
    :param f: Affine Term
    :param U: Snapshots
    :param m: Number of DOFs for ROM
    :param C: state to output matrix
    :param beta: regularization parameter
 savemat('heat_exchanger_5x5_fem_model.mat', {'M': M,'K':K,'f':f, 'U':U})
%}

tic
yalmip('clear')
    
[nx, ns] = size(U);
H = C'*C;

% Initialize program decision variables
Phi = sdpvar(nx,m);
Gamma = sdpvar(ns,m);
alpha = sdpvar(m,ns);

% Assign initial value for decision variable Phi and Gamma
[POD_Phi,~,~] = svd(U);
assign(Phi, POD_Phi(:,1:m));
assign(Gamma, U \ POD_Phi(:,1:m))

% #### Constraints #####
constraints = [];

% helper variables
Mhat = Phi'*M*Phi;
Khat = Phi'*K*Phi;
fhat = Phi'*f;

% Initial time constraint
t0 = 1;
constraints = [constraints, Mhat*alpha(:,t0) == Phi'*M*U(:,t0)];

% dynamic time constraints
for ts = t0:ns-1
    constraints = [constraints, ...
        Mhat*alpha(:,ts+1) == Khat*alpha(:,ts) + fhat];
end

constraints = [constraints, Phi == U * Gamma];

% #### Objective ####
objective = 0;

for ts = 1:ns
    du = U(:,ts) - Phi*alpha(:,ts);
    objective = objective + du'*H*du;
end
    
for i = 1:m
    objective = objective + beta*(1 - Phi(:,i)'*Phi(:,i))^2;
end

for i = 1:m
    for j = 1:m
        if i == j
            continue;
        end
        objective = objective + beta*(Phi(:,i)'*Phi(:,j))^2;
    end
end

toc
disp('Completed Problem Setup')
tic

optimize(constraints,objective, sdpsettings(...
    'verbose',1,...
    'usex0',1,...
    'showprogress', 1,...
    'cachesolvers', 1));
toc

V = value(Phi);

% Orthogonal Procrustes Problem
[VV,~,WW] = svd(V);
[m, n] = size(V);
V = VV * eye(m,n) * WW';

end

