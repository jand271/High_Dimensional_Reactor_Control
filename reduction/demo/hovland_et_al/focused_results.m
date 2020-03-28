clear all;
close all;

load('snapshots_of_hovland_el_al.mat')

[FOM, Z, U] = randomSmallSystem2();

%[UU,~,~] = svd(X);
%POD_V = UU(:,1:3);
load('focused/carlberg_none.mat');
[Dz_POD, ~] = computeInputEffectBoundForReduction(FOM, W, V, Z, U); 

dz_norm_per_focus_state = zeros(7,1);

figure;
hold on;
for i = 1:6
    
load(['focused/carlberg_focused_on_',num2str(i),'.mat']);

[Dz, ~] = computeInputEffectBoundForReduction(FOM, W, V, Z, U);    

plot(Dz(1:2:end)/norm(Dz(1:2:end)));
dz_norm_per_focus_state(i) = norm(Dz(1:2:end));
end

plot(Dz_POD(1:2:end)/norm(Dz_POD(1:2:end)),'k')

title('Normalized Constraint Tightening per State')
xlabel('State Number')
legend('focus state 1','focus state 2', 'focus state 3', 'focus state 4',...
    'focus state 5', 'focus state 6', 'no focus state','location','best');
saveas(gcf, 'focused_states.png')

figure;
hold on;
load('focused/carlberg_focused_on_1_and_6.mat');
[Dz, Du] = computeInputEffectBoundForReduction(FOM, W, V, Z, U);    
plot(Dz(1:2:end)/norm(Dz(1:2:end)));

plot(Dz_POD(1:2:end)/norm(Dz_POD(1:2:end)),'k')

title('Normalized Constraint Tightening per State')
xlabel('State Number')
legend('focus state 1 and 6', 'no focus state','location','best');
saveas(gcf, 'focused_states_1_6.png')

dz_norm_per_focus_state