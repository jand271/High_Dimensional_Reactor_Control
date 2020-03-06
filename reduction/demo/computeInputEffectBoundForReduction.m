function [Dz, Du] = computeInputEffectBoundForReduction(FOM, W, V, Z, U)

n = size(V, 2);

A = W'*FOM.Af*V;
B = W'*FOM.Bf;
C = FOM.Cf*V;
H = FOM.Hf*V;
Q = V'*FOM.Qf*V; 
R = FOM.Rf;
S = 0;

ROM = struct('A', A, 'B', B, 'C', C, 'H', H, 'Q', Q, 'R', R, ...
                'V', V, 'W', W, 'S', S);
nf = size(FOM.Af,1);
m = size(FOM.Bf,2);

% Design controllers using LQR
[K,P,~] = dlqr(A, -B, Q, R);
[L,~,~] = dlqr(A', C', Q, FOM.RL);
L = L';
CTRL = struct('K', K, 'L', L, 'P', P);

% Compute constraint tightening amount via Lorenzetti et al. 2020
tau = 100;
Pp = eye(nf) - ROM.V*inv(ROM.W'*ROM.V)*ROM.W';
Ae = [FOM.Af, FOM.Bf*CTRL.K; CTRL.L*FOM.Cf, ROM.A + ROM.B*CTRL.K - CTRL.L*ROM.C];
Be = [Pp*FOM.Af*ROM.V, Pp*FOM.Bf; sparse(n, n), sparse(n, m)];
Ez = [Z.A*FOM.Hf*speye(nf), sparse(size(Z.A, 1), n)];
Eu = [sparse(size(U.A, 1), nf), U.A*CTRL.K*speye(n)];
ne = size(Ae,1);
ERROR = struct('Ae', Ae, 'Be', Be, 'Ez', Ez, 'Eu', Eu);
[Xbar] = computeXbar(ROM.A, ROM.B, ROM.H, Z, U, tau);
[Dz, Du] = computeInputEffectBound(ROM, ERROR, Z, U, Xbar, 0, 0, tau);
end