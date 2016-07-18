clear all
clc
%%
fprintf('Loading data.\n');
f=fopen('code.dat','rb');
for i=1:10000
  B(i,:)=fread(f, 12, 'float');
end
B=sign(B);
fclose(f);

f=fopen('label.dat','rb');
label=fread(f,10000,'float');
fclose(f);

fprintf('Computing precision matrix.\n');
recall = 0:0.05:1;
dis_mtx=pdist2(B,B,'hamming');
pm=compute_precision_recall(dis_mtx, label, label, recall);

plot(recall(2:end), mean(pm(:,2:end),1), 'LineWidth', 3);
title('CIFAR-10', 'Fontsize', 30);
xlabel('Recall', 'Fontsize', 25);
ylabel('Precision', 'Fontsize', 25);
axis([0,1,0,0.8])
set(gca, 'xtick', 0:0.2:1.0, 'ytick', 0:0.2:1.0, 'FontSize', 25);
set(gca, 'YGrid', 'on');