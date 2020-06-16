function [c,A,b] = Klee_Minty(n)
c = [];
A = zeros(n);
b = [];
for i = 1:n 
    c = [-10^(i-1) c];
    b = [b ; 100^(i-1)];
    for j = 1:i
        A(i,j) = 2*10^(i-j);
    end
end
A = A - eye(n);
