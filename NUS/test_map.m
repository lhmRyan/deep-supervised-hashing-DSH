clearvars
nbit = 128;
nlabel = 21;
%%
f_code = fopen('code.dat','rb');
f_label = fopen('label.dat','rb');
B = zeros(10000, nbit);
label = zeros(10000, nlabel);
for i = 1:10000
  B(i,:) = fread(f_code, nbit, 'float');
  label(i,:) = fread(f_label, nlabel, 'float');
end
fclose(f_code);
fclose(f_label);

testlabel = label*(2.^[0:nlabel-1])';

B=sign(B);
[~, ~, map] = compute_map_multi(-B*B', testlabel, testlabel)