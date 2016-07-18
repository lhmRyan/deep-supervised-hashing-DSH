f=fopen('code.dat','rb');
for i=1:10000
  B(i,:)=fread(f, 12, 'float');
end
fclose(f);

f=fopen('label.dat','rb');
label=fread(f,10000,'float');
fclose(f);

B=sign(B);
[~, ~, map] = compute_map(-B*B', label, label)
