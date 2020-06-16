function [value,x] = RSLP_demo2(c,A,b,Aeq,beq)

N = length(c);
M1 = length(b);
M2 = length(beq);
basic_number = M1 + M2;
non_basic_number = N - M2;
c = [c zeros(1,M1)];
A = [A ; Aeq];
A = [A eye(basic_number)];
b = [b ; beq];
basic_index = N + 1 - M2 : N + M1 
non_basic_index = 1 : N - M2
B = [];
c_basic = [];
for i = basic_index 
    B = [B A(:,i)];
    c_basic = [c_basic c(i)];
end
it = 1;

phase_1 = false;
x_B = B\b
min_x_B = Inf;
for i = 1:length(x_B) 
    if x_B(i) < min_x_B 
        min_x_B = x_B(i);
        leave_index = i;
    end
end
% phase I
unbound = 0;
if min_x_B < 0
    phase_1 = true;
    disp('== phase I ==')
    a = basic_index(leave_index);
    basic_index(leave_index) = N + M1 + 1;
    
    c_1 = [zeros(1,N + M1 ) 1];
    for i = 1:basic_number 
        c_basic(i) = c_1(basic_index(i));
    end
    non_basic_index = [non_basic_index a];
    artificial = -1*ones(M1 + M2,1);
    artificial = B*artificial;
    B(:,leave_index) = artificial;
end 

infeasible = false;
while phase_1 == true
    disp(['== iteration ',num2str(it),' =='])
    x_B = B\b;
    min_c_d = Inf;
    min_P = ones(1,M1 + M2);
    
    for i = 1: N - M2 + 1
        if non_basic_index(i) == N + M1 + 1
            P = B\artificial;
        else
            P = B\A(:,non_basic_index(i));
        end
        c_d = c_1(non_basic_index(i)) - c_basic*P;
        if c_d <= min_c_d
            min_c_d = c_d;
            min_P = P;
            enter_index = i;
        end
    end
    if min_c_d >= 0
        it = it + 1;
        break
    
    else
        unbound = 1;
        for i = 1:length(min_P) 
            if min_P(i) > 0 
                unbound = 0;
                break
            end
        end
        if unbound == 1
            break
        end
        m = Inf;
        leave_index = 0;
        for i = 1: M1 + M2
            if min_P(i) > 0
                ratio = x_B(i) / min_P(i);
                if ratio < m
                    m = ratio;
                    leave_index = i;
                elseif ratio == m && m ~= Inf 
                    unbound = 1;
                    ratio;
                    m;
                end
            end
        end
        if leave_index == 0  
            unbound = 1
        end
        if unbound == 1
            break
        end
        x = basic_index(leave_index);
        basic_index(leave_index) = non_basic_index(enter_index);
        c_basic(leave_index) = c_1(non_basic_index(enter_index));
        non_basic_index(enter_index) = x;
        B(:,leave_index) = A(:,basic_index(leave_index));

    end
    it = it+1;
end

min_x_B = Inf;
for i = 1:length(x_B) 
    if x_B(i) < min_x_B 
        min_x_B = x_B(i);
        leave_index = i;
    end
end

for i = 1:basic_number
   if  basic_index(i) == N+M1+1
       infeasible = true
       disp('yo')
   end
end
if min_x_B < 0
    infeasible = true;
end
if infeasible == true
    disp('== infeasible solution ==')
else
    disp('== phase II ==')

    for i = 1:M1+M2
    
        c_basic(i) = c(basic_index(i));
    end

    for i = 1:length(non_basic_index)
        if non_basic_index(i) == N + M1 + 1
            for j = i+1:length(non_basic_index)
                non_basic_index(j-1) = non_basic_index(j);
            end
            break;
            non_basic_index = non_basic_index(1:end-1);
        end
    end
end
while  infeasible == false
    disp(['== iteration ',num2str(it),' =='])
    x_B = B\b;
    min_c_d = Inf;
    min_P = ones(1,N);
    for i = 1:N - M2
        
        P = B\A(:,non_basic_index(i))
        c_d = c(non_basic_index(i)) - c_basic*P
        if c_d <= min_c_d
            min_c_d = c_d;
            min_P = P;
            enter_index = i;
        end
       
    end
    
    if min_c_d >= 0
        break
    
    else
        unbound = 1;
        x_B
        for i = 1:length(x_B) 
            if x_B(i) > 0 
                unbound = 0;
                break
            end
        end
        if unbound == 1
            disp('x_B <= 0')
            break
        end
        m = Inf;
        leave_index = 0;
        min_P
        for i = 1: M1 + M2
            if min_P(i) > 0
                ratio = x_B(i) / min_P(i)
                if ratio < m
                    m = ratio;
                    leave_index = i;
                    unbound = 0;
                elseif ratio == m && m ~= Inf 
                    unbound = 1;
                    disp('repeated')
                end
            end
        end
        if leave_index == 0
           unbound = 1;
           disp('yo')
        end
        if unbound == 1
            disp('ratio(i) = m')
            break
        end
        
        x = basic_index(leave_index);
        basic_index(leave_index) = non_basic_index(enter_index);
        c_basic(leave_index) = c(non_basic_index(enter_index));
        non_basic_index(enter_index) = x;
        B(:,leave_index) = A(:,basic_index(leave_index));

    end
    it = it+1;
end
c = c(1:N);

x = zeros(N,1);
if min(basic_index) > N
    unbound = 1;
end
if unbound == 0 && infeasible == false
    disp('optimal solution')
    for i = 1:length(basic_index)
        if basic_index(i) <= N
            x(basic_index(i)) = x_B(i);
        end
    end
    value = c*x;
else
    if unbound == 1
        disp('unbound')
    end
    disp('no solution')
    value = 0;
end


