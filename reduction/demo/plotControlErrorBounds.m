function plotControlErrorBounds(FOM, Z, U, reduction_list, rank_list, max_buithanh_rank)

DuNorms = containers.Map();
DzNorms = containers.Map();

for reduction = reduction_list
    
    if strcmpi(reduction,'buithanh')
        modified_rank_list = 1:max_buithanh_rank;
    else
        modified_rank_list = rank_list;
    end
    
    du_norms = zeros(length(modified_rank_list), 1);
    dz_norms = zeros(length(modified_rank_list), 1);
    
    for rank = modified_rank_list
        
        directory = fullfile('output',strcat(reduction{1}, '_rank_', num2str(rank)));
        load(directory)
        
        try
            [Dz, Du] = computeInputEffectBoundForReduction(FOM, W, V, Z, U);
        catch
            warning(sprintf(...
                'Reduction strategy %s did not yeild results.', ...
                reduction{1}))
            Dz = nan;
            Du = nan;
        end
        
        du_norms(rank) = norm(Du);
        dz_norms(rank) = norm(Dz);
    end
    
    DuNorms(reduction{1}) = du_norms;
    DzNorms(reduction{1}) = dz_norms;
end

figure;
hold on;
for reduction = reduction_list
    plot(DuNorms(reduction{1}), '*-');
end
title('Norm of Du');
legend(reduction_list, 'Interpreter', 'none', 'location','best')
set(gca, 'YScale', 'log')
saveas(gcf, 'Du_norm.png');

figure;
hold on;
for reduction = reduction_list
    plot(DzNorms(reduction{1}), '*-');
end
title('Norm of Dz');
legend(reduction_list, 'Interpreter', 'none', 'location','best')
set(gca, 'YScale', 'log')
saveas(gcf, 'Dz_norm.png');

end

