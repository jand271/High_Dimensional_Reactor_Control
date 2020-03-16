function H2 = compute_H2_norm_discrete(A, B, C, W, V)

Ar = W'*A*V;
Br = W'*B;
Cr = C*V;

A_aug = blkdiag(A, Ar);
B_aug = [B; Br];
C_aug = [C, -Cr];

state_space_model = ss(A_aug, B_aug, C_aug, [],[]);

H2 = norm(state_space_model);
end
