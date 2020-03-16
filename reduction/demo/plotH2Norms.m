function plotH2Norms(reduction_list, rank_list, max_buithanh_rank, A, B, C)

H2Norms = containers.Map();

for reduction = reduction_list
    
    if strcmpi(reduction,'buithanh')
        modified_rank_list = 1:max_buithanh_rank;
    else
        modified_rank_list = rank_list;
    end
    
    h2_norms = zeros(length(modified_rank_list), 1);
    
    for rank = modified_rank_list
        
        directory = fullfile('output',strcat(reduction{1}, '_rank_', num2str(rank)));
        load(directory)
        
        h2_norms(rank) = compute_H2_norm_discrete(A, B, C, W, V);
    end
    
    H2Norms(reduction{1}) = h2_norms;
end

figure;
hold on;
for reduction = reduction_list
    plot(H2Norms(reduction{1}), '*-');
end
title('H2 Norms');
legend(reduction_list, 'Interpreter', 'none', 'location', 'best')
xlabel('rank')
set(gca, 'YScale', 'log')
saveas(gcf, 'H2_norm.png');

end