function [FOM, Z, U] = randomSmallSystem2()
% Simple random discrete time LTI SISO model, nf = 6
Af = [0.28, 0.25, -0.19, -0.22, 0.03, -0.50;
 0.25, -0.47, 0.30, 0.17, -0.11, -0.11;
 -0.19, 0.30, 0.46, 0.09, -0.02, -0.08;
 -0.22, 0.17, 0.09, 0.60, -0.06, 0.14;
 0.03, -0.11, -0.02, -0.06, 0.46, -0.13;
 -0.50, -0.11, -0.08, 0.14, -0.13, -0.23];

Bf = [1.0159; 0; 0.5988; 1.8641; 0; -1.2155];

Bfw = eye(6);

Cf = eye(6);

Hf = eye(6);

nf = size(Af, 1);
m = size(Bf, 2);
p = size(Cf, 1);
o = size(Hf,1);

% Constraints
zUB = [50, 50, 50, 50, 50, 50];
zLB = -zUB;
uUB = 20;
uLB = -uUB;
Z = rectanglePolytope(zUB, zLB);
U = rectanglePolytope(uUB, uLB);

% Controller design costs
Qf = 10*eye(nf);
Rf = 1*eye(m);
RL = 10*eye(p);

% Define data structures
FOM = struct('Af', Af, 'Bf', Bf, 'Bfw', Bfw, 'Cf', Cf, 'Hf', Hf, 'Qf', Qf, 'Rf', Rf, 'RL', RL);

end

