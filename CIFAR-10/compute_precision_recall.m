function [precision_mtx] = compute_precision_recall(dis_mtx, query_label, database_label, recall)
tic;

q_num = length(query_label);
d_num = length(database_label);

precision_mtx = zeros(q_num, length(recall));
database_label_mtx = repmat(database_label, 1, q_num);
sorted_database_label_mtx = database_label_mtx;

[mtx idx_mtx] = sort(dis_mtx, 1);

for q = 1 : q_num
    sorted_database_label_mtx(:, q) = database_label_mtx(idx_mtx(:, q), q);
end

result_mtx = (sorted_database_label_mtx == repmat(query_label', d_num, 1));

for q = 1 : q_num
    if mod(q,1000)==0
        fprintf('query = %d\n', q);
    end
    starter=1;
    for r = 1 : length(recall)
        thres = recall(r)*sum(result_mtx(:, q));
        for k = starter : d_num
            if sum(result_mtx(1:k, q)) >= thres
                starter = k+1;
                break;
            end
        end
        precision_mtx(q, r) = sum(result_mtx(1:k, q))/k;
    end
end

toc;
end