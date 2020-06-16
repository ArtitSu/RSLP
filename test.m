tic;
[x,val,FALG,output] = linprog(c,A,b,[],[],zeros(1,1000))
time = toc;