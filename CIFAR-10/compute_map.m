function [topN_idx, topN, map] = compute_map(dis_mtx, query_label, database_label)
tic;

q_num = length(query_label);
d_num = length(database_label);
map = zeros(q_num, 1);

database_label_mtx = repmat(database_label, 1, q_num);
sorted_database_label_mtx = database_label_mtx;

[~,idx_mtx] = sort(dis_mtx, 1);

for q = 1 : q_num
    sorted_database_label_mtx(:, q) = database_label_mtx(idx_mtx(:, q), q);
end

result_mtx = (sorted_database_label_mtx == repmat(query_label', d_num, 1));
topN = sum(result_mtx(1:10, :));
topN_idx = idx_mtx(1:10, :);

for q = 1 : q_num
    Qi = sum(result_mtx(:, q));
    map(q) = sum( ([1:Qi]') ./ (find(result_mtx(:, q) == 1)) ) / Qi;
end

map = mean(map);
toc;
end